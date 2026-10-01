//! GPU encoder + classification head, dispatched through [`crate::gpu::Kernels`].
//!
//! The speaker cache stays on the CPU: it is a ranking over a few hundred frames,
//! not a GEMM. Each chunk uploads its encoder window, runs the 31 blocks and the
//! head, and reads the logits back.

use crate::config::{AudioConfig, HeadConfig, ModelConfig};
use crate::error::{DiarizationError, Result};
use crate::gpu::{gemm_bm, gemm_wave_bm, Kernels, StridedGemm, Uploader, GEMM_BM, GEMM_BN};
use crate::weights::Weights;

const MAX_SEQ: usize = 1024;
const P: &str = "model.audio_tower";

/// Op classes [`GpuEngine::bench_stages`] can switch on and off individually.
pub const S_LN: u32 = 1 << 0;
pub const S_QKV: u32 = 1 << 1;
pub const S_ROPE: u32 = 1 << 2;
pub const S_QK: u32 = 1 << 3;
pub const S_SM: u32 = 1 << 9;
pub const S_PV: u32 = 1 << 10;
pub const S_ATTN: u32 = S_QK | S_SM | S_PV;
pub const S_O: u32 = 1 << 4;
pub const S_ADD: u32 = 1 << 5;
pub const S_FC1: u32 = 1 << 6;
pub const S_GELU: u32 = 1 << 7;
pub const S_FC2: u32 = 1 << 8;
pub const S_ALL: u32 = S_LN | S_QKV | S_ROPE | S_ATTN | S_O | S_ADD | S_FC1 | S_GELU | S_FC2;

fn rope_tables(seq: usize, hd: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0f64 / theta.powf(2.0 * i as f64 / hd as f64)) as f32)
        .collect();
    let mut cos = vec![0.0f32; seq * hd];
    let mut sin = vec![0.0f32; seq * hd];
    for p in 0..seq {
        let pos = p as f32;
        for i in 0..half {
            let a = pos * inv[i];
            cos[p * hd + i] = a.cos();
            sin[p * hd + i] = a.sin();
            cos[p * hd + half + i] = cos[p * hd + i];
            sin[p * hd + half + i] = sin[p * hd + i];
        }
    }
    (cos, sin)
}

struct Ln {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
}

struct Linear {
    w: wgpu::Buffer,
    b: Option<wgpu::Buffer>,
    out: usize,
    inn: usize,
}

struct GpuBlock {
    ln1: Ln,
    /// Q, K and V stacked row-wise into one `[3h, h]` weight. They are mutually
    /// independent given the same A, so they run as one `n = 3h` GEMM: that is 12
    /// workgroups instead of three dispatches of 4, which is what actually fills
    /// six SMs (and all of them at `gy = 1`, i.e. every ultra-low-latency window).
    qkv: Linear,
    o: Linear,
    ln2: Ln,
    fc1: Linear,
    fc2: Linear,
}

/// Scratch sized for `MAX_SEQ` encoder frames.
struct Scratch {
    x: wgpu::Buffer,
    res: wgpu::Buffer,
    n: wgpu::Buffer,
    /// `[MAX_SEQ, 3*h]` holding Q, K and V as three contiguous column blocks.
    qkv: wgpu::Buffer,
    ctx: wgpu::Buffer,
    ff: wgpu::Buffer,
    proj: wgpu::Buffer,
    up: wgpu::Buffer,
    logits: wgpu::Buffer,
    cos: wgpu::Buffer,
    sin: wgpu::Buffer,
    stacked: wgpu::Buffer,
    /// `[frames, hh*3]` staging for the sub-pixel conv's im2col.
    conv_col: wgpu::Buffer,
    /// `[heads, seq, seq]` attention scores, reused in place as the probabilities.
    scores: wgpu::Buffer,
}

struct Unis {
    qkv: wgpu::Buffer,
    chk: wgpu::Buffer,
    o: wgpu::Buffer,
    fc1: wgpu::Buffer,
    fc2: wgpu::Buffer,
    ln: wgpu::Buffer,
    gelu: wgpu::Buffer,
    add: wgpu::Buffer,
    rope: wgpu::Buffer,
    rope_k: wgpu::Buffer,
    attn_qk: wgpu::Buffer,
    attn_sm: wgpu::Buffer,
    attn_pv: wgpu::Buffer,
    proj: wgpu::Buffer,
    conv: wgpu::Buffer,
    conv_gemm: wgpu::Buffer,
    dense: wgpu::Buffer,
    outp: wgpu::Buffer,
    relu: wgpu::Buffer,
}

struct LayerBg {
    ln1: wgpu::BindGroup,
    qkv: wgpu::BindGroup,
    o: wgpu::BindGroup,
    /// `o` and `fc2` are the two `n == hidden` GEMMs, and those are the only ones
    /// `gemm_wave_bm` ever re-tiles. A bind group is *exclusive* to the pipeline
    /// it was built from (wgpu rejects a mismatch with "Exclusive pipelines
    /// don't match"), so the 64x128 instantiation needs its own copies.
    o_m64: wgpu::BindGroup,
    ln2: wgpu::BindGroup,
    fc1: wgpu::BindGroup,
    fc2: wgpu::BindGroup,
    fc2_m64: wgpu::BindGroup,
}

struct SharedBg {
    input_ln: wgpu::BindGroup,
    final_ln: wgpu::BindGroup,
    rope_q: wgpu::BindGroup,
    rope_k: wgpu::BindGroup,
    attn_qk: wgpu::BindGroup,
    attn_sm: wgpu::BindGroup,
    attn_pv: wgpu::BindGroup,
    gelu: wgpu::BindGroup,
    add: wgpu::BindGroup,
    proj: wgpu::BindGroup,
    conv: wgpu::BindGroup,
    conv_gemm: wgpu::BindGroup,
    relu_up: wgpu::BindGroup,
    dense: wgpu::BindGroup,
    relu_d: wgpu::BindGroup,
    outp: wgpu::BindGroup,
}

/// QK^T and PV for one attention step, as two strided GEMMs with the head index in
/// `workgroup_id.z`.
///
/// `q`/`k`/`v` are `[seq, heads*hd]` row-major, so both operands carry a row stride
/// of `hidden` and the head offset is `z * head_dim`. The scores buffer is
/// `[heads, seq, seq]`, which is also the layout PV wants for its A operand.
/// `qkv_base` is the element offset of the K (resp. V) block inside the packed
/// `[seq, 3*hidden]` buffer; both the encoder and the self-checks feed the strided
/// kernels the same packed buffer, so the offset is part of the contract and is
/// covered by `ATTN_GEMM_CHECK` rather than assumed.
fn attn_gemms(seq: u32, heads: u32, hd: u32, hidden: u32) -> (StridedGemm, StridedGemm) {
    let tiles = seq.div_ceil(128);
    // Q, K and V are three *column blocks* of one `[seq, 3*hidden]` buffer, so the
    // row stride of every operand is `3*hidden` -- not `hidden` -- and the block
    // identity is a column offset, not a flat `seq*hidden` offset. Using the old
    // per-matrix numbers makes QK read K from the wrong row of every row but the
    // first, and the scores come out with the wrong sign and magnitude.
    let rs = 3 * hidden;
    let kbase = hidden;
    let vbase = 2 * hidden;
    let qk = StridedGemm {
        m: seq,
        n: seq,
        k: hd,
        a_rs: rs,
        a_base: 0,
        a_cs: 1,
        a_zs: hd,
        w_rs: rs,
        w_base: kbase,
        w_cs: 1,
        w_zs: hd,
        c_rs: seq,
        c_base: 0,
        c_zs: seq * seq,
        gx: tiles,
        gy: tiles,
        gz: heads,
    };
    let pv = StridedGemm {
        m: seq,
        n: hd,
        k: seq,
        a_rs: seq,
        a_base: 0,
        a_cs: 1,
        a_zs: seq * seq,
        w_rs: 1,
        w_base: vbase,
        w_cs: rs,
        w_zs: hd,
        c_rs: hidden,
        c_base: 0,
        c_zs: hd,
        gx: 1,
        gy: seq.div_ceil(GEMM_BM),
        gz: heads,
    };
    (qk, pv)
}

pub struct GpuEngine {
    k: Kernels,
    cfg: AudioConfig,
    head_cfg: HeadConfig,
    sub: usize,
    embed: Linear,
    input_ln: Ln,
    layers: Vec<GpuBlock>,
    final_ln: Ln,
    proj: Linear,
    conv_w: wgpu::Buffer,
    conv_b: wgpu::Buffer,
    dense: Linear,
    out: Linear,
    scratch: Scratch,
    unis: Unis,
    layer_bg: Vec<LayerBg>,
    shared_bg: SharedBg,
    /// Persistent host-to-device staging for the per-window encoder input.
    uploader: Uploader,
    cos_data: Vec<f32>,
    sin_data: Vec<f32>,
}

impl GpuEngine {
    pub fn load(model: &ModelConfig, w: &Weights) -> Result<Self> {
        let k = Kernels::new()?;
        let cfg = model.audio_config.clone();
        let head_cfg = model.head_config.clone();
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hh = head_cfg.hidden_size;
        let up = cfg.subsampling_factor;
        let ns = head_cfg.num_speakers;
        let kv = cfg.num_key_value_heads;
        let hd = h / cfg.num_attention_heads;

        let lin = |k: &Kernels, name: &str, out: usize, inn: usize, bias: Option<&str>| -> Result<Linear> {
            let wbuf = k.upload_new(name, w.tensor(name, &[out, inn])?);
            let b = match bias {
                Some(bn) => Some(k.upload_new(bn, w.tensor(bn, &[out])?)),
                None => None,
            };
            Ok(Linear { w: wbuf, b, out, inn })
        };
        // Q, K and V are independent, so concatenate the three `[out, inn]` weights
        // into one `[3*out, inn]` buffer and let a single GEMM produce all three.
        // `out` is the same for all three because this model is MHA, not GQA
        // (`num_key_value_heads == num_attention_heads`); the shape check inside
        // `tensor` is what keeps that assumption honest.
        let lin3 = |k: &Kernels, pfx: &str, out: usize, inn: usize| -> Result<Linear> {
            let mut data = vec![0f32; 3 * out * inn];
            for (i, which) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                let name = format!("{pfx}.self_attn.{which}.weight");
                let t = w.tensor(&name, &[out, inn])?;
                data[i * out * inn..(i + 1) * out * inn].copy_from_slice(t);
            }
            Ok(Linear {
                w: k.upload_new(&format!("{pfx}.self_attn.qkv"), &data),
                b: None,
                out: 3 * out,
                inn,
            })
        };
        let ln = |k: &Kernels, prefix: &str| -> Result<Ln> {
            Ok(Ln {
                w: k.upload_new(&format!("{prefix}.weight"), w.tensor(&format!("{prefix}.weight"), &[h])?),
                b: k.upload_new(&format!("{prefix}.bias"), w.tensor(&format!("{prefix}.bias"), &[h])?),
            })
        };

        let embed = lin(
            &k,
            &format!("{P}.embedder.projection.weight"),
            h,
            cfg.subsampling_factor * cfg.num_mel_bins,
            None,
        )?;
        let input_ln = ln(&k, &format!("{P}.input_layer_norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let b = format!("{P}.layers.{i}");
            layers.push(GpuBlock {
                ln1: ln(&k, &format!("{b}.layer_norm1"))?,
                qkv: lin3(&k, &b, kv * hd, h)?,
                o: lin(
                    &k,
                    &format!("{b}.self_attn.o_proj.weight"),
                    h,
                    h,
                    Some(&format!("{b}.self_attn.o_proj.bias")),
                )?,
                ln2: ln(&k, &format!("{b}.layer_norm2"))?,
                fc1: lin(
                    &k,
                    &format!("{b}.mlp.fc1.weight"),
                    inter,
                    h,
                    Some(&format!("{b}.mlp.fc1.bias")),
                )?,
                fc2: lin(
                    &k,
                    &format!("{b}.mlp.fc2.weight"),
                    h,
                    inter,
                    Some(&format!("{b}.mlp.fc2.bias")),
                )?,
            });
        }
        let final_ln = ln(&k, &format!("{P}.layer_norm"))?;
        let proj = lin(&k, "model.proj.weight", hh, h, Some("model.proj.bias"))?;
        let conv_w = k.upload_new(
            "model.upsampler.conv.weight",
            w.tensor("model.upsampler.conv.weight", &[hh * up, hh, 3])?,
        );
        let conv_b = k.upload_new(
            "model.upsampler.conv.bias",
            w.tensor("model.upsampler.conv.bias", &[hh * up])?,
        );
        let dense = lin(&k, "classifier.dense.weight", hh, hh, Some("classifier.dense.bias"))?;
        let out = lin(
            &k,
            "classifier.out_proj.weight",
            ns,
            hh,
            Some("classifier.out_proj.bias"),
        )?;

        let scratch = Scratch {
            x: k.storage("x", MAX_SEQ * h),
            res: k.storage("res", MAX_SEQ * h),
            n: k.storage("n", MAX_SEQ * h),
            qkv: k.storage("qkv", MAX_SEQ * 3 * h),
            ctx: k.storage("ctx", MAX_SEQ * h),
            ff: k.storage("ff", MAX_SEQ * inter),
            proj: k.storage("proj", MAX_SEQ * up * hh),
            up: k.storage("up", MAX_SEQ * up * hh),
            logits: k.storage("logits", MAX_SEQ * up * ns),
            cos: k.storage("cos", MAX_SEQ * hd),
            sin: k.storage("sin", MAX_SEQ * hd),
            stacked: k.storage("stacked", MAX_SEQ * cfg.subsampling_factor * cfg.num_mel_bins),
            scores: k.storage("scores", cfg.num_attention_heads * MAX_SEQ * MAX_SEQ),
            conv_col: k.storage("conv_col", MAX_SEQ * head_cfg.hidden_size * 3),
        };

        let hd = h / cfg.num_attention_heads;
        let (rope_cos, rope_sin) = rope_tables(MAX_SEQ, hd, cfg.rope_parameters.rope_theta);
        k.gpu().upload_f32(&scratch.cos, &rope_cos);
        k.gpu().upload_f32(&scratch.sin, &rope_sin);

        let unis = Unis {
            qkv: k.make_uniform("u_qkv"),
            chk: k.make_uniform("u_chk"),
            o: k.make_uniform("u_o"),
            fc1: k.make_uniform("u_fc1"),
            fc2: k.make_uniform("u_fc2"),
            ln: k.make_uniform("u_ln"),
            gelu: k.make_uniform("u_gelu"),
            add: k.make_uniform("u_add"),
            rope: k.make_uniform32("u_rope"),
            rope_k: k.make_uniform32("u_rope_k"),
            attn_qk: k.make_uniform64("u_attn_qk"),
            attn_sm: k.make_uniform("u_attn_sm"),
            attn_pv: k.make_uniform64("u_attn_pv"),
            proj: k.make_uniform("u_proj"),
            conv: k.make_uniform("u_conv"),
            conv_gemm: k.make_uniform("u_conv_gemm"),
            dense: k.make_uniform("u_dense"),
            outp: k.make_uniform("u_out"),
            relu: k.make_uniform("u_relu"),
        };
        let dummy = k.dummy_bias().clone();
        let gemm_p = k.pipes().gemm.clone();
        // Bind groups are exclusive to their pipeline, so the 64x128
        // instantiation needs its own copies of the two bind groups it may run.
        let gemm_m64_p = k.pipes().gemm_m64.clone();
        let ln_p = k.pipes().ln.clone();
        let unary_p = k.pipes().unary.clone();
        let rope_p = k.pipes().rope.clone();
        let col3_p = k.pipes().col3.clone();
        let layer_bg: Vec<LayerBg> = layers
            .iter()
            .map(|b| LayerBg {
                ln1: k.bind5(&ln_p, &scratch.x, &b.ln1.w, &b.ln1.b, &scratch.n, &unis.ln),
                qkv: k.bind5(&gemm_p, &scratch.n, &b.qkv.w, &dummy, &scratch.qkv, &unis.qkv),
                o: k.bind5(
                    &gemm_p,
                    &scratch.ctx,
                    &b.o.w,
                    b.o.b.as_ref().unwrap(),
                    &scratch.n,
                    &unis.o,
                ),
                o_m64: k.bind5(
                    &gemm_m64_p,
                    &scratch.ctx,
                    &b.o.w,
                    b.o.b.as_ref().unwrap(),
                    &scratch.n,
                    &unis.o,
                ),
                ln2: k.bind5(&ln_p, &scratch.x, &b.ln2.w, &b.ln2.b, &scratch.n, &unis.ln),
                fc1: k.bind5(
                    &gemm_p,
                    &scratch.n,
                    &b.fc1.w,
                    b.fc1.b.as_ref().unwrap(),
                    &scratch.ff,
                    &unis.fc1,
                ),
                fc2: k.bind5(
                    &gemm_p,
                    &scratch.ff,
                    &b.fc2.w,
                    b.fc2.b.as_ref().unwrap(),
                    &scratch.n,
                    &unis.fc2,
                ),
                fc2_m64: k.bind5(
                    &gemm_m64_p,
                    &scratch.ff,
                    &b.fc2.w,
                    b.fc2.b.as_ref().unwrap(),
                    &scratch.n,
                    &unis.fc2,
                ),
            })
            .collect();
        let gemm_m64_p = k.pipes().gemm_m64.clone();
        let shared_bg = SharedBg {
            input_ln: k.bind5(&ln_p, &scratch.res, &input_ln.w, &input_ln.b, &scratch.x, &unis.ln),
            final_ln: k.bind5(&ln_p, &scratch.x, &final_ln.w, &final_ln.b, &scratch.n, &unis.ln),
            rope_q: k.bind4(&rope_p, &scratch.qkv, &scratch.cos, &scratch.sin, &unis.rope),
            rope_k: k.bind4(&rope_p, &scratch.qkv, &scratch.cos, &scratch.sin, &unis.rope_k),
            attn_qk: k.bind5(
                &k.pipes().gemm_strided.clone(),
                &scratch.qkv,
                &scratch.qkv,
                &dummy,
                &scratch.scores,
                &unis.attn_qk,
            ),
            attn_sm: k.bind2(&k.pipes().attn_sm.clone(), &scratch.scores, &unis.attn_sm),
            attn_pv: k.bind5(
                &k.pipes().gemm_n64.clone(),
                &scratch.scores,
                &scratch.qkv,
                &dummy,
                &scratch.ctx,
                &unis.attn_pv,
            ),
            gelu: k.bind2(&unary_p, &scratch.ff, &unis.gelu),
            add: k.bind_add(&scratch.x, &scratch.n, &unis.add),
            proj: k.bind5(
                &gemm_p,
                &scratch.n,
                &proj.w,
                proj.b.as_ref().unwrap(),
                &scratch.proj,
                &unis.proj,
            ),
            conv: k.bind3(&col3_p, &scratch.proj, &scratch.conv_col, &unis.conv),
            conv_gemm: k.bind5(
                &gemm_p,
                &scratch.conv_col,
                &conv_w,
                &conv_b,
                &scratch.up,
                &unis.conv_gemm,
            ),
            relu_up: k.bind2(&unary_p, &scratch.up, &unis.relu),
            dense: k.bind5(
                &gemm_p,
                &scratch.up,
                &dense.w,
                dense.b.as_ref().unwrap(),
                &scratch.proj,
                &unis.dense,
            ),
            relu_d: k.bind2(&unary_p, &scratch.proj, &unis.relu),
            outp: k.bind5(
                &gemm_p,
                &scratch.proj,
                &out.w,
                out.b.as_ref().unwrap(),
                &scratch.logits,
                &unis.outp,
            ),
        };

        // big enough for the largest per-window window *and* a stacked mel chunk
        let uploader = Uploader::new(
            &k.gpu(),
            (MAX_SEQ * h.max(cfg.subsampling_factor * cfg.num_mel_bins) * 4) as u64,
            3,
        );

        Ok(Self {
            k,
            cfg,
            head_cfg,
            sub: up,
            embed,
            input_ln,
            layers,
            final_ln,
            proj,
            conv_w,
            conv_b,
            dense,
            out,
            scratch,
            unis,
            layer_bg,
            shared_bg,
            uploader,
            cos_data: rope_cos,
            sin_data: rope_sin,
        })
    }

    pub fn describe(&self) -> String {
        self.k.describe()
    }

    /// Isolated GEMM throughput for the encoder's `seq x 512 x 512` shape.
    /// Spin the GPU for ~2 s before returning, so the next measurement is not taken
    /// on the clock ramp. Measured with `nvidia-smi`: idle sits at 139 MHz (P8) and
    /// takes ~1.5 s to reach the 1721 MHz steady state, which is longer than a whole
    /// short benchmark -- that is what made a cold run look like a real speedup once.
    pub fn warm_gpu(&mut self) {
        let p = self.k.pipes().gemm.clone();
        let bg = self.layer_bg[0].qkv.clone();
        let t = std::time::Instant::now();
        while t.elapsed().as_secs_f64() < 2.0 {
            let mut enc = self.k.encoder();
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&p);
                pass.set_bind_group(0, &bg, &[]);
                for _ in 0..64 {
                    pass.dispatch_workgroups(12, 8, 1);
                }
            }
            self.k.submit(enc);
        }
        let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn bench_gemm(&mut self, seq: u32, iters: u32) -> (f64, f64) {
        let h = self.cfg.hidden_size as u32;
        self.k.write_uniform(&self.unis.qkv, seq, 3 * h, h, 0);
        let gx = (3 * h).div_ceil(GEMM_BN);
        // Must match the pipeline's row tile. Using the `GEMM_BM` constant here
        // dispatches a 128-row grid against the 64-row instantiation, which computes
        // only the first half of the output while the TFLOP/s figure still divides
        // by the full flop count -- an invented ~50% speedup. Every grid that feeds
        // a shared `gemm_p` pipeline needs this, not just the ones on the hot path.
        let gy = seq.div_ceil(gemm_bm());
        let gemm_p = self.k.pipes().gemm.clone();
        let bg = self.layer_bg[0].qkv.clone();
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &bg, &[]);
            for _ in 0..iters {
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        let t = std::time::Instant::now();
        self.k.submit(enc);
        let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // n is now 3*hidden (the packed QKV projection), so the flop count must be too
        let flop = 2.0 * seq as f64 * (3 * h) as f64 * h as f64 * iters as f64;
        (ms, flop / (ms * 1e9))
    }

    /// `embed_mel` (whole recording) vs the per-window `embed_mel_range` calls that
    /// `run_offline` actually makes, on the same input. They have to agree
    /// **bit for bit**: the offline path was rewritten to embed one window at a time
    /// so that a long recording no longer needs a whole-recording buffer, and that is
    /// only safe if a window's rows are the same products the whole-file form would
    /// have produced. `win` is the window size, i.e. `chunk_length + chunk_right_context`.
    pub fn check_embed_range(&mut self, num_frames: usize, win: usize) -> (f64, bool) {
        let mels = self.cfg.num_mel_bins;
        let mel: Vec<f32> = (0..num_frames * mels)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect();
        let whole = self.embed_mel(&mel, num_frames).expect("whole embed");
        let groups = num_frames.div_ceil(self.sub);
        let mut ranged: Vec<f32> = Vec::with_capacity(groups * self.cfg.hidden_size);
        let mut lo = 0usize;
        while lo < groups {
            let hi = (lo + win).min(groups);
            ranged.extend_from_slice(&self.embed_mel_range(&mel, num_frames, lo, hi).expect("range embed"));
            lo = hi;
        }
        if whole.len() != ranged.len() {
            return (f64::INFINITY, false);
        }
        let mut worst = 0.0f64;
        for (a, b) in whole.iter().zip(&ranged) {
            worst = worst.max((*a as f64 - *b as f64).abs());
        }
        (worst, worst == 0.0)
    }

    /// Max abs difference between the shared-memory GEMM and a plain CPU reference
    /// on deterministic inputs. Fast enough to run after every kernel edit.
    pub fn check_gemm(&mut self, m: u32, n: u32, k: u32) -> f64 {
        let val = |i: usize| (((i as f32 * 0.017).sin()) * 0.5 + ((i as f32 * 0.0031).cos())) as f32;
        let a: Vec<f32> = (0..(m * k) as usize).map(val).collect();
        let w: Vec<f32> = (0..(n * k) as usize).map(|i| val(i + 7)).collect();
        let bias: Vec<f32> = (0..n as usize).map(|i| val(i + 13) * 0.1).collect();
        let abuf = self.k.upload_new("chk_a", &a);
        let wbuf = self.k.upload_new("chk_w", &w);
        let bbuf = self.k.upload_new("chk_b", &bias);
        let cbuf = self.k.storage("chk_c", (m * n) as usize);
        self.k.write_uniform(&self.unis.chk, m, n, k, 1);
        let gemm_p = self.k.pipes().gemm.clone();
        let bg = self.k.bind5(&gemm_p, &abuf, &wbuf, &bbuf, &cbuf, &self.unis.chk);
        let gx = n.div_ceil(GEMM_BN);
        let gy = m.div_ceil(gemm_bm());
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.k.submit(enc);
        let got = self.k.gpu().readback_f32(&cbuf, (m * n) as usize).expect("gemm readback");
        let mut worst = 0.0f64;
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut acc = 0.0f32;
                for t in 0..k as usize {
                    acc += a[i * k as usize + t] * w[j * k as usize + t];
                }
                acc += bias[j];
                let g = got[i * n as usize + j];
                worst = worst.max((g - acc).abs() as f64);
                if std::env::var_os("GEMM_DUMP").is_some() && i < 8 && j < 8 {
                    println!("    C[{i},{j}] got {g:>12.5} want {acc:>12.5}");
                }
            }
        }
        worst
    }

    /// `QK^T` for every head, checked against a CPU triple loop:
    /// `C[i,j] = sum_d A[i*hidden + h*hd + d] * W[j*hidden + h*hd + d]`, with the
    /// head index riding in `workgroup_id.z` through `a_zs`/`w_zs`/`c_zs`. This
    /// path shares no code with `check_gemm` -- it is the strided instantiation
    /// with `k == head_dim` -- so it has to be checked on its own.
    pub fn check_gemm_qk(&mut self, seq: u32, heads: u32, hd: u32, hidden: u32) -> f64 {
        let val = |i: usize| (((i as f32 * 0.017).sin()) * 0.5 + ((i as f32 * 0.0031).cos())) as f32;
        let nn = (seq as usize) * (hidden as usize);
        let a: Vec<f32> = (0..nn).map(val).collect();
        let w: Vec<f32> = (0..nn).map(|i| val(i + 5)).collect();
        // Same packed `[seq, 3*hidden]` buffer the encoder produces, so the block
        // column offsets and the `3*hidden` row stride are exercised here rather
        // than only in production.
        let hn = nn / seq as usize;
        let mut aw = vec![0f32; 3 * nn];
        for i in 0..seq as usize {
            let (r, src) = (i * 3 * hn, i * hn);
            aw[r..r + hn].copy_from_slice(&a[src..src + hn]);
            aw[r + hn..r + 2 * hn].copy_from_slice(&w[src..src + hn]);
        }
        let qkvbuf = self.k.upload_new("chk_qkv", &aw);
        let cbuf = self.k.storage("chk_qc", (heads * seq * seq) as usize);
        let (qk, _) = attn_gemms(seq, heads, hd, hidden);
        self.k.write_uniform_g16(&self.unis.attn_qk, &qk);
        let p = self.k.pipes().gemm_strided.clone();
        let bg = self.k.bind5(&p, &qkvbuf, &qkvbuf, &qkvbuf, &cbuf, &self.unis.attn_qk);
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(qk.gx, qk.gy, qk.gz);
        }
        self.k.submit(enc);
        let got = self.k.gpu().readback_f32(&cbuf, (heads * seq * seq) as usize).expect("qk readback");
        let mut worst = 0.0f64;
        let (mut wh, mut wi, mut wj) = (0usize, 0usize, 0usize);
        let mut bad = 0usize;
        let mut first: Vec<String> = Vec::new();
        for h in 0..heads as usize {
            for i in 0..seq as usize {
                for j in 0..seq as usize {
                    let mut acc = 0.0f32;
                    for d in 0..hd as usize {
                        acc += a[i * hidden as usize + h * hd as usize + d]
                            * w[j * hidden as usize + h * hd as usize + d];
                    }
                    let g = got[(h * seq as usize + i) * seq as usize + j];
                    let e = (g - acc).abs() as f64;
                    if e > worst {
                        worst = e;
                        wh = h; wi = i; wj = j;
                    }
                    if e > 1e-3 {
                        bad += 1;
                        if first.len() < 6 {
                            first.push(format!("h{h} i{i} j{j} got {g:.3} want {acc:.3}"));
                        }
                    }
                }
            }
        }
        if std::env::var_os("GEMM_DUMP").is_some() {
            println!("    qk seq={seq}: {bad} bad of {}, worst {:.3} at h{wh} i{wi} j{wj}",
                     (seq * seq * heads) as usize, worst);
            for l in first.iter() { println!("      {l}"); }
        }
        worst
    }

    /// `P V` for every head: `C[i,h,d] = sum_t S[h,i,t] * V[t,h,d]`. The strided
    /// config expresses this as a `m=seq, n=hd, k=seq` GEMM whose `w` operand has
    /// `w_rs = 1` and `w_cs = hidden`, i.e. V is read transposed, and whose result
    /// lands back in the V buffer (`c_rs = hidden`, `c_zs = hd`).
    pub fn check_gemm_pv(&mut self, seq: u32, heads: u32, hd: u32, hidden: u32) -> f64 {
        let val = |i: usize| (((i as f32 * 0.021).sin()) * 0.5) as f32;
        let sv = (0..(heads * seq * seq) as usize).map(|i| val(i + 3) * 0.1).collect::<Vec<f32>>();
        let vv: Vec<f32> = (0..(seq as usize * hidden as usize)).map(val).collect();
        let nn = (seq as usize) * (hidden as usize);
        let mut vv3 = vec![0f32; 3 * nn];
        let hn = nn / seq as usize;
        for i in 0..seq as usize {
            vv3[i * 3 * hn + 2 * hn..(i + 1) * 3 * hn].copy_from_slice(&vv[i * hn..(i + 1) * hn]);
        }
        let sbuf = self.k.upload_new("chk_ps", &sv);
        let vbuf = self.k.upload_new("chk_pv", &vv3);
        let cbuf = self.k.storage("chk_pc", (seq * hidden) as usize);
        {
            let mut enc = self.k.encoder();
            enc.copy_buffer_to_buffer(&vbuf, (2 * nn * 4) as u64, &cbuf, 0, (seq * hidden) as u64 * 4);
            self.k.submit(enc);
        }
        let (_, pv) = attn_gemms(seq, heads, hd, hidden);
        self.k.write_uniform_g16(&self.unis.attn_pv, &pv);
        let p = self.k.pipes().gemm_n64.clone();
        let bg = self.k.bind5(&p, &sbuf, &vbuf, &vbuf, &cbuf, &self.unis.attn_pv);
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(pv.gx, pv.gy, pv.gz);
        }
        self.k.submit(enc);
        let got = self.k.gpu().readback_f32(&cbuf, (seq * hidden) as usize).expect("pv readback");
        let mut worst = 0.0f64;
        for h in 0..heads as usize {
            for i in 0..seq as usize {
                for d in 0..hd as usize {
                    let mut acc = 0.0f32;
                    for t in 0..seq as usize {
                        acc += sv[(h * seq as usize + i) * seq as usize + t]
                            * vv[t * hidden as usize + h * hd as usize + d];
                    }
                    let g = got[i * hidden as usize + h * hd as usize + d];
                    worst = worst.max((g - acc).abs() as f64);
                }
            }
        }
        worst
    }

    /// Cost of a single submit, separated from the work inside it. `GPU_PROFILE`
    /// on the streaming modes shows ~40 ms inside `device.poll(wait)` per window
    /// that barely moves between seq=4 and seq=19, and `bench_stages` shows the
    /// same constant while the tower count changes 40x — so the floor is per
    /// submit. This measures it directly: empty submits, submits with one trivial
    /// dispatch, and submits with one real tower.
    pub fn probe_submit_cost(&mut self) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        let n = 8usize;
        for label in ["empty submit", "1 trivial dispatch"] {
            let t = std::time::Instant::now();
            for _ in 0..n {
                let enc = self.k.encoder();
                self.k.submit(enc);
                let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
            }
            let per = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            let _ = label;
            out.push(("empty submit".to_string(), per));
            break;
        }
        let t = std::time::Instant::now();
        for _ in 0..n {
            let enc = self.k.encoder();
            self.k.submit(enc);
            let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
        }
        out.push(("empty submit".to_string(), t.elapsed().as_secs_f64() * 1e3 / n as f64));
        let t = std::time::Instant::now();
        for _ in 0..n {
            let mut enc = self.k.encoder();
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.k.pipes().add.clone());
                pass.set_bind_group(0, &self.shared_bg.add, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.k.submit(enc);
            let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
        }
        out.push(("1 trivial dispatch".to_string(), t.elapsed().as_secs_f64() * 1e3 / n as f64));
        // Is the floor per-dispatch? Same pipeline and bind group repeated, so any
        // growth here is wgpu/Vulkan's per-dispatch cost (pipeline barrier between
        // passes that share a buffer), not the kernel.
        for reps in [1usize, 4, 16, 64, 256] {
            let m = 4usize;
            let t = std::time::Instant::now();
            for _ in 0..m {
                let mut enc = self.k.encoder();
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.k.pipes().add.clone());
                    pass.set_bind_group(0, &self.shared_bg.add, &[]);
                    for _ in 0..reps {
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                self.k.submit(enc);
                let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
            }
            let per = t.elapsed().as_secs_f64() * 1e3 / m as f64;
            out.push((format!("{reps} trivial dispatches"), per));
        }
        for seq in [8u32, 64, 380] {
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = self.bench_stages(seq, seq, 0xFFFF, 1);
            }
            out.push((format!("1 tower seq={seq}"), t.elapsed().as_secs_f64() * 1e3 / n as f64));
        }
        out
    }

    /// Max abs difference between the in-place scaled softmax and a plain CPU
    /// reference on deterministic rows. `valid < cols` also checks that the masked
    /// tail is zeroed.
    pub fn check_attn_softmax(&mut self, rows: u32, cols: u32, valid: u32) -> f64 {
        let data: Vec<f32> = (0..(rows * cols) as usize)
            .map(|i| (((i as f32 * 0.021).sin()) * 3.0) as f32)
            .collect();
        let s = self.k.upload_new("chk_s", &data);
        let scale = 1.0 / 8.0f32;
        self.k.begin();
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.k.attn_softmax(&mut pass, &s, rows, cols, valid, scale);
        }
        self.k.submit(enc);
        let got = self.k.gpu().readback_f32(&s, (rows * cols) as usize).expect("softmax readback");
        let mut worst = 0.0f64;
        for r in 0..rows as usize {
            let base = r * cols as usize;
            let mut mx = f32::NEG_INFINITY;
            for j in 0..valid as usize {
                mx = mx.max(data[base + j]);
            }
            let mut sum = 0.0f32;
            let mut want = vec![0.0f32; cols as usize];
            for j in 0..valid as usize {
                let e = ((data[base + j] - mx) * scale).exp();
                want[j] = e;
                sum += e;
            }
            for j in 0..valid as usize {
                want[j] /= sum;
            }
            for j in 0..cols as usize {
                worst = worst.max((got[base + j] - want[j]).abs() as f64);
            }
        }
        worst
    }

    /// Can two independent GEMM dispatches in one pass actually overlap?
    ///
    /// This decides whether windows can be run concurrently. Vulkan guarantees a
    /// *memory* dependency between dispatches in a command buffer but not an
    /// *execution* one, so dispatches that touch disjoint buffers should be free
    /// to overlap -- and a `n = 512` GEMM only puts 4 workgroups on 6 SMs, leaving
    /// room for a second one. Three timings settle it:
    ///   solo  : one dispatch
    ///   pair  : two dispatches, disjoint buffers, one pass
    ///   chain : two dispatches, real data dependency, one pass
    /// If `pair` lands near `solo` and well under `chain`, overlap is real.
    pub fn probe_dispatch_concurrency(&mut self, seq: u32) -> (f64, f64, f64) {
        let h = self.cfg.hidden_size;
        let n = h as u32;
        let k = h as u32;
        let iters = 60;
        let val = |i: usize| ((i as f32 * 0.013).sin()) as f32;
        let a: Vec<f32> = (0..(seq as usize) * h as usize).map(val).collect();
        let wv: Vec<f32> = (0..(h * h) as usize).map(val).collect();
        let w = self.k.upload_new("cc_w", &wv);
        let a1 = self.k.upload_new("cc_a1", &a);
        let a2 = self.k.upload_new("cc_a2", &a);
        let c1 = self.k.storage("cc_c1", seq as usize * h);
        let c2 = self.k.storage("cc_c2", seq as usize * h);
        self.k.write_uniform(&self.unis.chk, seq, n, k, 0);
        let p = self.k.pipes().gemm.clone();
        let gx = n.div_ceil(GEMM_BN);
        let gy = seq.div_ceil(gemm_bm());
        let b1 = self.k.bind5(&p, &a1, &w, &self.k.dummy_bias(), &c1, &self.unis.chk);
        let b2 = self.k.bind5(&p, &a2, &w, &self.k.dummy_bias(), &c2, &self.unis.chk);
        // chained: a1 -> c1 -> a1' so the second dispatch must wait for the first
        let c3 = self.k.storage("cc_c3", seq as usize * h);
        let b3 = self.k.bind5(&p, &c1, &w, &self.k.dummy_bias(), &c3, &self.unis.chk);

        let run = |pass: &mut dyn FnMut(&mut wgpu::ComputePass)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut enc = self.k.encoder();
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    for _ in 0..iters {
                        pass(&mut cp);
                    }
                }
                let t = std::time::Instant::now();
                self.k.submit(enc);
                let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let solo = run(&mut |cp: &mut wgpu::ComputePass| {
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &b1, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        });
        let pair = run(&mut |cp: &mut wgpu::ComputePass| {
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &b1, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
            cp.set_bind_group(0, &b2, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        });
        let chain = run(&mut |cp: &mut wgpu::ComputePass| {
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &b1, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
            cp.set_bind_group(0, &b3, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        });
        (solo, pair, chain)
    }

    /// Max abs difference between the packed QKV projection and three independent
    /// CPU projections of the same layer.
    ///
    /// This is the only check that covers the *assembly*, not just the kernel: the
    /// `n = 3*hidden` GEMM, the row-concatenation of the three weight tensors, and
    /// the column-block layout the rest of the encoder indexes into all have to be
    /// right simultaneously. `GEMM_CHECK` pins the kernel at whatever `n` it is
    /// given (and never saw 1536), `ATTN_GEMM_CHECK` feeds the strided kernels a
    /// buffer it packed itself, and `ROPE_CHECK` owns its own buffer -- so when this
    /// merge was first done, every one of them was green while the encoder read the
    /// wrong halves. It failed end-to-end with `maxdiff 4.1e1` and no local signal.
    pub fn check_qkv(&mut self, seq: u32, w: &Weights) -> f64 {
        let h = self.cfg.hidden_size;
        let pfx = format!("{P}.layers.0.self_attn");
        let val = |i: usize| (((i as f32 * 0.017).sin()) * 0.5 + ((i as f32 * 0.0031).cos())) as f32;
        let a: Vec<f32> = (0..(seq as usize * h)).map(val).collect();
        let abuf = self.k.upload_new("chk_qkv_a", &a);
        self.k.write_uniform(&self.unis.qkv, seq, 3 * h as u32, h as u32, 0);
        let p = self.k.pipes().gemm.clone();
        let bg = self.k.bind5(
            &p,
            &abuf,
            &self.layers[0].qkv.w,
            &self.k.dummy_bias(),
            &self.scratch.qkv,
            &self.unis.qkv,
        );
        let gx = (3 * h as u32).div_ceil(GEMM_BN);
        let gy = seq.div_ceil(gemm_bm());
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.k.submit(enc);
        let got = self
            .k
            .gpu()
            .readback_f32(&self.scratch.qkv, seq as usize * 3 * h)
            .expect("qkv readback");
        // CPU: each of the three source tensors, placed at its own column block.
        let mut worst = 0.0f64;
        let mut bad_block = [0usize; 3];
        for (blk, which) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            let t = w.tensor(&format!("{pfx}.{which}.weight"), &[h, h]).expect("qkv weight");
            for i in 0..seq as usize {
                for j in 0..h {
                    let mut acc = 0.0f32;
                    for kk in 0..h {
                        acc += a[i * h + kk] * t[j * h + kk];
                    }
                    let d = (got[i * 3 * h + blk * h + j] - acc).abs() as f64;
                    if d > 1e-3 {
                        bad_block[blk] += 1;
                    }
                    worst = worst.max(d);
                }
            }
        }
        let per = (seq as usize) * h;
        if worst > 1e-3 {
            eprintln!(
                "    qkv block errors: q {}/{} k {}/{} v {}/{}",
                bad_block[0], per, bad_block[1], per, bad_block[2], per
            );
        }
        worst
    }

    /// QKV -> rope -> QK -> softmax -> PV on the *production* bind groups, against
    /// a CPU reference built from the same layer weights.
    ///
    /// The individual pieces are each covered elsewhere and were all green when the
    /// packed-QKV merge first broke the encoder end to end: `QKV_CHECK` covers the
    /// merged GEMM, `ROPE_CHECK` covers both blocks (on its own buffer) and
    /// `ATTN_GEMM_CHECK` covers both strided kernels (on a buffer it packed itself).
    /// What none of them crosses is the *composition* -- the same weights, the same
    /// offsets, the same bind groups the encoder uses. This does.
    pub fn check_attn_chain(&mut self, seq: u32, w: &Weights) -> f64 {
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads as u32;
        let hd = h as u32 / nh;
        let sq = seq as usize;
        let pfx = format!("{P}.layers.0.self_attn");
        let val = |i: usize| (((i as f32 * 0.017).sin()) * 0.5 + ((i as f32 * 0.0031).cos())) as f32;
        let a: Vec<f32> = (0..sq * h).map(val).collect();

        // ---- CPU reference: Q, K, V from the three source tensors
        let mut q = vec![0f32; sq * h];
        let mut k = vec![0f32; sq * h];
        let mut v = vec![0f32; sq * h];
        for (blk, dst) in [&mut q, &mut k, &mut v].into_iter().enumerate() {
            let which = ["q_proj", "k_proj", "v_proj"][blk];
            let t = w.tensor(&format!("{pfx}.{which}.weight"), &[h, h]).expect("attn chain weight");
            for i in 0..sq {
                for j in 0..h {
                    let mut acc = 0f32;
                    for kk in 0..h {
                        acc += a[i * h + kk] * t[j * h + kk];
                    }
                    dst[i * h + j] = acc;
                }
            }
        }
        let q_raw = q.clone();
        let k_raw = k.clone();
        // rope on Q and K only (V is not rotated)
        let half = hd as usize / 2;
        for src in [&mut q, &mut k] {
            for t in 0..sq {
                for hd_i in 0..nh as usize {
                    for i in 0..half {
                        let b = t * h + hd_i * hd as usize + i;
                        let c = t * hd as usize + i;
                        let x1 = src[b];
                        let x2 = src[b + half];
                        src[b] = x1 * self.cos_data[c] - x2 * self.sin_data[c];
                        src[b + half] = x2 * self.cos_data[c + half] + x1 * self.sin_data[c + half];
                    }
                }
            }
        }
        // attention
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut want = vec![0f32; sq * h];
        let mut sc = vec![0f32; sq];
        for z in 0..nh as usize {
            for i in 0..sq {
                let mut mx = f32::NEG_INFINITY;
                for j in 0..sq {
                    let mut acc = 0f32;
                    for d in 0..hd as usize {
                        acc += q[i * h + z * hd as usize + d] * k[j * h + z * hd as usize + d];
                    }
                    sc[j] = acc * scale;
                    if sc[j] > mx {
                        mx = sc[j];
                    }
                }
                let mut sum = 0f32;
                for j in 0..sq {
                    sc[j] = (sc[j] - mx).exp();
                    sum += sc[j];
                }
                for d in 0..hd as usize {
                    let mut acc = 0f32;
                    for j in 0..sq {
                        acc += sc[j] / sum * v[j * h + z * hd as usize + d];
                    }
                    want[i * h + z * hd as usize + d] = acc;
                }
            }
        }

        // ---- GPU: the production chain
        self.k.gpu().upload_f32(&self.scratch.n, &a);
        self.k.write_uniform(&self.unis.qkv, seq, 3 * h as u32, h as u32, 0);
        self.k.write_uniform6(&self.unis.rope, seq, nh, hd, 0, 3 * h as u32);
        self.k.write_uniform6(&self.unis.rope_k, seq, nh, hd, h as u32, 3 * h as u32);
        let (qk, pv) = attn_gemms(seq, nh, hd, h as u32);
        self.k.write_uniform_g16(&self.unis.attn_qk, &qk);
        self.k.write_uniform_g16(&self.unis.attn_pv, &pv);
        self.k.write_uniform(&self.unis.attn_sm, seq * nh, seq, seq, scale.to_bits());
        let gemm_p = self.k.pipes().gemm.clone();
        let (c, si) = (self.scratch.cos.clone(), self.scratch.sin.clone());
        let diff = |got: &[f32], want: &[f32]| -> f64 {
            got.iter().zip(want).map(|(g, wv)| (*g - *wv).abs() as f64).fold(0.0, f64::max)
        };
        // Three separate submits so each stage can be bisected: with everything in
        // one pass, reading back after the fact only tells you the composition is
        // wrong, not which link is.
        self.k.gpu().upload_f32(&self.scratch.n, &a);
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &self.layer_bg[0].qkv, &[]);
            pass.dispatch_workgroups((3 * h as u32).div_ceil(GEMM_BN), seq.div_ceil(gemm_bm()), 1);
        }
        self.k.submit(enc);
        // The GEMM writes `[seq, 3*h]` row-major, so the three blocks are
        // interleaved *within* each row, not concatenated. Stacking whole
        // matrices is the same values in the wrong order and reads as a huge
        // mismatch that looks exactly like a kernel bug.
        let interleave = |a: &[f32], b: &[f32], c: &[f32]| -> Vec<f32> {
            let mut o = vec![0f32; sq * 3 * h];
            for i in 0..sq {
                o[i * 3 * h..i * 3 * h + h].copy_from_slice(&a[i * h..(i + 1) * h]);
                o[i * 3 * h + h..i * 3 * h + 2 * h].copy_from_slice(&b[i * h..(i + 1) * h]);
                o[i * 3 * h + 2 * h..i * 3 * h + 3 * h].copy_from_slice(&c[i * h..(i + 1) * h]);
            }
            o
        };
        let packed = interleave(&q_raw, &k_raw, &v);
        let d_qkv = diff(
            &self.k.gpu().readback_f32(&self.scratch.qkv, sq * 3 * h).expect("qkv stage"),
            &packed,
        );

        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.k.rope(&mut pass, &self.scratch.qkv, &c, &si, seq, nh, hd, 0, 3 * h as u32);
            self.k.rope(&mut pass, &self.scratch.qkv, &c, &si, seq, nh, hd, h as u32, 3 * h as u32);
        }
        self.k.submit(enc);
        let roped = interleave(&q, &k, &v);
        let d_rope = diff(
            &self.k.gpu().readback_f32(&self.scratch.qkv, sq * 3 * h).expect("rope stage"),
            &roped,
        );

        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.k.pipes().gemm_strided.clone());
            pass.set_bind_group(0, &self.shared_bg.attn_qk, &[]);
            pass.dispatch_workgroups(qk.gx, qk.gy, qk.gz);
        }
        self.k.submit(enc);
        let gs = self
            .k
            .gpu()
            .readback_f32(&self.scratch.scores, nh as usize * sq * sq)
            .expect("scores stage");
        // reference must match whatever the GPU actually holds at this point
        let (qr, kr): (&Vec<f32>, &Vec<f32>) = (&q, &k);
        let mut d_scores = 0.0f64;
        for z in 0..nh as usize {
            for i in 0..sq.min(8) {
                for j in 0..sq {
                    let mut acc = 0f32;
                    for dd in 0..hd as usize {
                        acc += qr[i * h + z * hd as usize + dd] * kr[j * h + z * hd as usize + dd];
                    }
                    // QK emits the *raw* dot product; the 1/sqrt(d) scale is applied
                    // by the softmax stage, not here.
                    d_scores = d_scores.max((gs[z * sq * sq + i * sq + j] - acc).abs() as f64);
                }
            }
        }
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.k.pipes().attn_sm.clone());
            pass.set_bind_group(0, &self.shared_bg.attn_sm, &[]);
            pass.dispatch_workgroups((seq * nh).div_ceil(8), 1, 1);
            pass.set_pipeline(&self.k.pipes().gemm_n64.clone());
            pass.set_bind_group(0, &self.shared_bg.attn_pv, &[]);
            pass.dispatch_workgroups(pv.gx, pv.gy, pv.gz);
        }
        self.k.submit(enc);
        let d_attn = diff(
            &self.k.gpu().readback_f32(&self.scratch.ctx, sq * h).expect("attn stage"),
            &want,
        );
        println!("      stage qkv {d_qkv:.3e} | rope {d_rope:.3e} | qk {d_scores:.3e} | attn {d_attn:.3e}");
        d_attn
    }

    /// Max abs difference between the GPU RoPE and the CPU formula on
    /// deterministic input.
    ///
    /// Runs it over a packed `[seq, 2*hidden]` buffer, once at block 0 (Q) and once
    /// at block `seq*hidden` (K). `cfg.d` is new logic introduced by the packed-QKV
    /// layout, and a check that only ever rotates block 0 cannot see a wrong offset
    /// -- it would pass while the encoder rotated the wrong half of QKV.
    pub fn check_rope(&mut self, seq: u32) -> (f64, f64) {
        let nh = self.cfg.num_attention_heads as u32;
        let hd = (self.cfg.hidden_size / self.cfg.num_attention_heads) as u32;
        let h = self.cfg.hidden_size as u32;
        let half = hd / 2;
        let hn = h as usize;
        let rows = seq as usize;
        // `[seq, 2*hidden]`: two blocks *per row*, interleaved, with different
        // content per block so a skipped or mis-offset block cannot cancel out.
        let data: Vec<f32> = (0..rows * 2 * hn)
            .map(|i| {
                let blk = (i % (2 * hn)) / hn;
                ((i as f32 * 0.013).sin() + blk as f32 * 0.5) as f32
            })
            .collect();
        let x = self.k.upload_new("chk_x", &data);
        self.k.begin();
        for (col, _blk) in [(0u32, 0u32), (h, 1u32)] {
            let mut enc = self.k.encoder();
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                self.k.rope(
                    &mut pass,
                    &x,
                    &self.scratch.cos,
                    &self.scratch.sin,
                    seq,
                    nh,
                    hd,
                    col,
                    2 * h,
                );
            }
            self.k.submit(enc);
        }
        let got = self
            .k
            .gpu()
            .readback_f32(&x, rows * 2 * hn)
            .expect("rope readback");
        let mut worst = 0.0f64;
        let mut unwritten = 0usize;
        for t in 0..seq as usize {
            for hd_i in 0..nh as usize {
                for i in 0..half as usize {
                    let c = t * hd as usize;
                    for blk in [0usize, hn] {
                        let b = t * 2 * hn + blk + hd_i * hd as usize;
                        let x1 = data[b + i];
                        let x2 = data[b + half as usize + i];
                        let w0 = x1 * self.cos_data[c + i] - x2 * self.sin_data[c + i];
                        let w1 = x2 * self.cos_data[c + half as usize + i]
                            + x1 * self.sin_data[c + half as usize + i];
                        if !w0.is_finite() {
                            continue;
                        }
                        worst = worst
                            .max((got[b + i] - w0).abs() as f64)
                            .max((got[b + half as usize + i] - w1).abs() as f64);
                        if (got[b + i] - x1).abs() < 1e-9 && (got[b + half as usize + i] - x2).abs() < 1e-9 {
                            unwritten += 1;
                        }
                    }
                }
            }
        }
        (worst, unwritten as f64)
    }
    /// Embedder on the GPU: stack `subsampling_factor` log-mel frames and project
    /// to `hidden`. Done in `MAX_SEQ`-frame chunks because a whole recording can have
    /// more encoder groups than the scratch buffers hold.
    ///
    /// On the CPU this was ~50 ms of the offline run for 1.3 GFLOP; the same GEMM is
    /// about 2 ms here.
    /// Whole-recording embedder, kept for callers that want everything at once.
    /// Chunked through [`Self::embed_mel_range`], so it has no length limit either.
    pub fn embed_mel(&mut self, mel: &[f32], num_frames: usize) -> Result<Vec<f32>> {
        let groups = num_frames.div_ceil(self.sub);
        let mut out = Vec::with_capacity(groups * self.cfg.hidden_size);
        for lo in (0..groups).step_by(MAX_SEQ) {
            out.extend_from_slice(&self.embed_mel_range(mel, num_frames, lo, (lo + MAX_SEQ).min(groups))?);
        }
        Ok(out)
    }

    /// Encoder groups `lo..hi` only, which is what every caller actually needs.
    ///
    /// The range has to fit `MAX_SEQ` groups, so the whole working set is the
    /// pre-existing `stacked` and `x` scratch — there is no whole-recording buffer
    /// and therefore no length limit on the recording. Bit-identical to the
    /// corresponding slice of the whole-file form: one submit, one GEMM, one
    /// readback, same operands per output row.
    pub fn embed_mel_range(
        &mut self,
        mel: &[f32],
        num_frames: usize,
        lo: usize,
        hi: usize,
    ) -> Result<Vec<f32>> {
        let (sub, mels) = (self.sub, self.cfg.num_mel_bins);
        let h = self.cfg.hidden_size;
        let groups = hi - lo;
        if groups > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!(
                "embed range {lo}..{hi} is {groups} groups, over MAX_SEQ {MAX_SEQ}"
            )));
        }
        let mut stacked = vec![0.0f32; groups * sub * mels];
        for f in 0..groups * sub {
            let g = lo * sub + f;
            let src = g * mels;
            // frames past the end of the recording stay zero; that padding is
            // load-bearing because it lands in the last group
            if g < num_frames && src + mels <= num_frames * mels {
                stacked[f * mels..f * mels + mels].copy_from_slice(&mel[src..src + mels]);
            }
        }
        self.k.begin();
        let mut enc = self.k.encoder();
        self.uploader
            .copy(self.k.gpu(), &mut enc, &self.scratch.stacked, &stacked)?;
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.k.gemm(
                &mut pass,
                &self.scratch.stacked,
                &self.embed.w,
                None,
                &self.scratch.x,
                groups as u32,
                h as u32,
                (sub * mels) as u32,
            );
        }
        self.k.submit(enc);
        self.k.gpu().readback_f32(&self.scratch.x, groups * h)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Wall time of one tower at `seq`, with only the op classes in `mask`
    /// dispatched `iters` times. Omitting a class leaves its buffers at whatever the
    /// previous run left there, so the remaining classes still see the same memory
    /// traffic and the same shapes — the delta between two masks is the cost of the
    /// class that was added, not of everything downstream changing.
    pub fn bench_stages(&mut self, seq: u32, valid: u32, mask: u32, iters: u32) -> f64 {
        let h = self.cfg.hidden_size as u32;
        let inter = self.cfg.intermediate_size as u32;
        let nh = self.cfg.num_attention_heads as u32;
        let hd = (self.cfg.hidden_size / self.cfg.num_attention_heads) as u32;
        self.k.write_uniform(&self.unis.qkv, seq, 3 * h, h, 0);
        self.k.write_uniform(&self.unis.o, seq, h, h, 1);
        self.k.write_uniform(&self.unis.fc1, seq, inter, h, 1);
        self.k.write_uniform(&self.unis.fc2, seq, h, inter, 1);
        self.k.write_uniform(&self.unis.ln, h, 1e-5f32.to_bits(), 0, 0);
        self.k.write_uniform(&self.unis.gelu, seq * inter, 0, 0, 0);
        self.k.write_uniform(&self.unis.add, seq * h, 0, 0, 0);
        self.k.write_uniform6(&self.unis.rope, seq, nh, hd, 0, 3 * h);
        self.k.write_uniform6(&self.unis.rope_k, seq, nh, hd, h, 3 * h);

        let gx = h.div_ceil(GEMM_BN);
        let gx_qkv = (3 * h).div_ceil(GEMM_BN);
        let gy = seq.div_ceil(gemm_bm());
        let gx_fc1 = inter.div_ceil(GEMM_BN);
        let ln_p = self.k.pipes().ln.clone();
        let gemm_p = self.k.pipes().gemm.clone();
        let qk_p = self.k.pipes().gemm_strided.clone();
        let pv_p = self.k.pipes().gemm_n64.clone();
        let sm_p = self.k.pipes().attn_sm.clone();
        let rope_p = self.k.pipes().rope.clone();
        let add_p = self.k.pipes().add.clone();
        let unary_p = self.k.pipes().unary.clone();
        let (qk, pv) = attn_gemms(seq, nh, hd, h);
        self.k.write_uniform_g16(&self.unis.attn_qk, &qk);
        self.k.write_uniform_g16(&self.unis.attn_pv, &pv);
        self.k.write_uniform(&self.unis.attn_sm, seq * nh, seq, valid, (1.0 / (hd as f32).sqrt()).to_bits());
        // One submit per iteration, not `iters` towers in one command buffer: this
        // keeps each submit shaped like production (a tower is always submitted on
        // its own) and keeps the command buffer well under any driver limit. The
        // per-tower cost is identical either way -- measured 92.4/93.1/92.4/91.3/92.1
        // ms at iters 1/2/5/20/50 in one pass, and 92.6/91.3/90.9/91.2 at iters
        // 1/2/5/20 with a submit each -- so the total does scale linearly with iters
        // and `gpu_check`'s divide-by-iters is sound. An earlier round read the
        // un-divided total as a per-tower figure and concluded the driver was
        // dropping 49 of 50 towers; that was a units error, not a driver behaviour.
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let mut enc = self.k.encoder();
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for i in 0..self.layer_bg.len() {
                    let b = &self.layer_bg[i];
                    if mask & S_LN != 0 {
                        pass.set_pipeline(&ln_p);
                        pass.set_bind_group(0, &b.ln1, &[]);
                        pass.dispatch_workgroups(seq, 1, 1);
                    }
                    if mask & S_QKV != 0 {
                        pass.set_pipeline(&gemm_p);
                        pass.set_bind_group(0, &b.qkv, &[]);
                        pass.dispatch_workgroups(gx_qkv, gy, 1);
                    }
                    if mask & S_ROPE != 0 {
                        pass.set_pipeline(&rope_p);
                        pass.set_bind_group(0, &self.shared_bg.rope_q, &[]);
                        pass.dispatch_workgroups((seq * nh).div_ceil(64), 1, 1);
                        pass.set_bind_group(0, &self.shared_bg.rope_k, &[]);
                        pass.dispatch_workgroups((seq * nh).div_ceil(64), 1, 1);
                    }
                    if mask & S_QK != 0 {
                        pass.set_pipeline(&qk_p);
                        pass.set_bind_group(0, &self.shared_bg.attn_qk, &[]);
                        pass.dispatch_workgroups(qk.gx, qk.gy, qk.gz);
                    }
                    if mask & S_SM != 0 {
                        pass.set_pipeline(&sm_p);
                        pass.set_bind_group(0, &self.shared_bg.attn_sm, &[]);
                        pass.dispatch_workgroups((seq * nh).div_ceil(8), 1, 1);
                    }
                    if mask & S_PV != 0 {
                        pass.set_pipeline(&pv_p);
                        pass.set_bind_group(0, &self.shared_bg.attn_pv, &[]);
                        pass.dispatch_workgroups(pv.gx, pv.gy, pv.gz);
                    }
                    if mask & S_O != 0 {
                        pass.set_pipeline(&gemm_p);
                        pass.set_bind_group(0, &b.o, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    if mask & S_ADD != 0 {
                        pass.set_pipeline(&add_p);
                        pass.set_bind_group(0, &self.shared_bg.add, &[]);
                        pass.dispatch_workgroups((seq * h).div_ceil(256), 1, 1);
                    }
                    if mask & S_FC1 != 0 {
                        pass.set_pipeline(&ln_p);
                        pass.set_bind_group(0, &b.ln2, &[]);
                        pass.dispatch_workgroups(seq, 1, 1);
                        pass.set_pipeline(&gemm_p);
                        pass.set_bind_group(0, &b.fc1, &[]);
                        pass.dispatch_workgroups(gx_fc1, gy, 1);
                    }
                    if mask & S_GELU != 0 {
                        pass.set_pipeline(&unary_p);
                        pass.set_bind_group(0, &self.shared_bg.gelu, &[]);
                        pass.dispatch_workgroups((seq * inter).div_ceil(256), 1, 1);
                    }
                    if mask & S_FC2 != 0 {
                        pass.set_pipeline(&gemm_p);
                        pass.set_bind_group(0, &b.fc2, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            self.k.submit(enc);
        }
        let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Encoder over already-stacked embeddings `[groups, hidden]`.
    pub fn forward_embeds(&mut self, embeds: &[f32], valid: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let seq = embeds.len() / h;
        if seq == 0 || seq > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!("seq {seq} outside 1..={MAX_SEQ}")));
        }
        self.encode_hidden(embeds, valid)
    }

    /// Encoder + head in one compute pass; only logits come back to the host.
    pub fn forward_window(&mut self, embeds: &[f32], valid: usize) -> Result<Vec<f32>> {
        let seq = embeds.len() / self.cfg.hidden_size;
        self.encode_on_gpu(embeds, valid)?;
        self.head_on_gpu(seq, true)
    }

    fn encode_hidden(&mut self, embeds: &[f32], valid: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let seq = embeds.len() / h;
        self.encode_on_gpu(embeds, valid)?;
        self.k.gpu().readback_f32(&self.scratch.n, seq * h)
    }

    fn encode_on_gpu(&mut self, embeds: &[f32], valid: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let seq = embeds.len() / h;
        if seq == 0 || seq > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!("seq {seq} outside 1..={MAX_SEQ}")));
        }
        let valid = valid.min(seq) as u32;
        let seq_u = seq as u32;
        let h_u = h as u32;
        let nh = self.cfg.num_attention_heads as u32;
        let hd = (h / self.cfg.num_attention_heads) as u32;
        let inter = self.cfg.intermediate_size as u32;

        let t_up = std::time::Instant::now();
        let mut enc = self.k.encoder();
        self.uploader.copy(self.k.gpu(), &mut enc, &self.scratch.res, embeds)?;
        let up_ms = t_up.elapsed().as_secs_f64() * 1e3;
        self.k.write_uniform(&self.unis.qkv, seq_u, 3 * h_u, h_u, 0);
        self.k.write_uniform(&self.unis.o, seq_u, h_u, h_u, 1);
        self.k.write_uniform(&self.unis.fc1, seq_u, inter, h_u, 1);
        self.k.write_uniform(&self.unis.fc2, seq_u, h_u, inter, 1);
        self.k.write_uniform(&self.unis.ln, h_u, 1e-5f32.to_bits(), 0, 0);
        self.k.write_uniform(&self.unis.gelu, seq_u * inter, 0, 0, 0);
        self.k.write_uniform(&self.unis.add, seq_u * h_u, 0, 0, 0);
        self.k.write_uniform6(&self.unis.rope, seq_u, nh, hd, 0, 3 * h_u);
        self.k.write_uniform6(&self.unis.rope_k, seq_u, nh, hd, h_u, 3 * h_u);

        let gx = h_u.div_ceil(GEMM_BN);
        let gx_qkv = (3 * h_u).div_ceil(GEMM_BN);
        // Must track the *pipeline's* row tile, not the compile-time constant: the
        // shared `gemm_p` pipeline is the 64-row instantiation under GEMM_BM=64,
        // and dispatching a 128-row grid against it silently computes only the
        // first half of every tile. GEMM_CHECK does not reach this path, so the
        // frame-level check is what catches it.
        let gy = seq_u.div_ceil(gemm_bm());
        let gx_fc1 = inter.div_ceil(GEMM_BN);
        let gx_fc2 = h_u.div_ceil(GEMM_BN);
        // `o` and `fc2` both have `n == hidden == 512`, which is the one shape
        // whose tile count (`4 * gy`) is *not* already a multiple of the 12
        // resident slots at most `gy` the streaming modes reach. `gemm_wave_bm`
        // returns the row tile *and* the pipeline together, so the grid can never
        // disagree with the instantiation that runs it -- the failure mode of the
        // sixth round's `GEMM_BM` was exactly a 128-row grid against a 64-row
        // pipeline, which computes the first half of every tile and still returns
        // plausible logits. Pipeline and `gy` are bound in one tuple for that
        // reason, not for style.
        let bm_o = gemm_wave_bm(seq_u, h_u);
        let bm_fc2 = gemm_wave_bm(seq_u, h_u);
        let (gemm_o_p, gy_o) = if bm_o == GEMM_BM {
            (self.k.pipes().gemm.clone(), seq_u.div_ceil(GEMM_BM))
        } else {
            (self.k.pipes().gemm_m64.clone(), seq_u.div_ceil(bm_o))
        };
        let (gemm_fc2_p, gy_fc2) = if bm_fc2 == GEMM_BM {
            (self.k.pipes().gemm.clone(), seq_u.div_ceil(GEMM_BM))
        } else {
            (self.k.pipes().gemm_m64.clone(), seq_u.div_ceil(bm_fc2))
        };
        let _ = (&bm_o, &bm_fc2);

        let t_rec = std::time::Instant::now();
        let uni_ms = t_rec.duration_since(t_up).as_secs_f64() * 1e3 - up_ms;
        let (scratch_cos, scratch_sin) = (self.scratch.cos.clone(), self.scratch.sin.clone());
        let scratch_qkv = self.scratch.qkv.clone();
        #[allow(unused_mut)]
        let ln_p = self.k.pipes().ln.clone();
        let gemm_p = self.k.pipes().gemm.clone();
        let qk_p = self.k.pipes().gemm_strided.clone();
        let pv_p = self.k.pipes().gemm_n64.clone();
        let sm_p = self.k.pipes().attn_sm.clone();
        let add_p = self.k.pipes().add.clone();
        let unary_p = self.k.pipes().unary.clone();
        let (qk, pv) = attn_gemms(seq_u, nh, hd, h_u);
        self.k.write_uniform_g16(&self.unis.attn_qk, &qk);
        self.k.write_uniform_g16(&self.unis.attn_pv, &pv);
        self.k
            .write_uniform(&self.unis.attn_sm, seq_u * nh, seq_u, valid, (1.0 / (hd as f32).sqrt()).to_bits());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tower"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ln_p);
            pass.set_bind_group(0, &self.shared_bg.input_ln, &[]);
            pass.dispatch_workgroups(seq_u, 1, 1);
            for i in 0..self.layer_bg.len() {
                let b = &self.layer_bg[i];
                pass.set_pipeline(&ln_p);
                pass.set_bind_group(0, &b.ln1, &[]);
                pass.dispatch_workgroups(seq_u, 1, 1);
                pass.set_pipeline(&gemm_p);
                pass.set_bind_group(0, &b.qkv, &[]);
                pass.dispatch_workgroups(gx_qkv, gy, 1);
                self.k.rope(&mut pass, &scratch_qkv, &scratch_cos, &scratch_sin, seq_u, nh, hd, 0, 3 * h_u);
                self.k.rope(
                    &mut pass,
                    &scratch_qkv,
                    &scratch_cos,
                    &scratch_sin,
                    seq_u,
                    nh,
                    hd,
                    h_u,
                    3 * h_u,
                );
                pass.set_pipeline(&qk_p);
                pass.set_bind_group(0, &self.shared_bg.attn_qk, &[]);
                pass.dispatch_workgroups(qk.gx, qk.gy, qk.gz);
                pass.set_pipeline(&sm_p);
                pass.set_bind_group(0, &self.shared_bg.attn_sm, &[]);
                pass.dispatch_workgroups((seq_u * nh).div_ceil(8), 1, 1);
                pass.set_pipeline(&pv_p);
                pass.set_bind_group(0, &self.shared_bg.attn_pv, &[]);
                pass.dispatch_workgroups(pv.gx, pv.gy, pv.gz);
                pass.set_pipeline(&gemm_o_p);
                if bm_o == GEMM_BM {
                    pass.set_bind_group(0, &b.o, &[]);
                } else {
                    pass.set_bind_group(0, &b.o_m64, &[]);
                }
                pass.dispatch_workgroups(gx, gy_o, 1);
                pass.set_pipeline(&add_p);
                pass.set_bind_group(0, &self.shared_bg.add, &[]);
                pass.dispatch_workgroups((seq_u * h_u).div_ceil(256), 1, 1);

                pass.set_pipeline(&ln_p);
                pass.set_bind_group(0, &b.ln2, &[]);
                pass.dispatch_workgroups(seq_u, 1, 1);
                pass.set_pipeline(&gemm_p);
                pass.set_bind_group(0, &b.fc1, &[]);
                pass.dispatch_workgroups(gx_fc1, gy, 1);
                pass.set_pipeline(&unary_p);
                pass.set_bind_group(0, &self.shared_bg.gelu, &[]);
                pass.dispatch_workgroups((seq_u * inter).div_ceil(256), 1, 1);
                pass.set_pipeline(&gemm_fc2_p);
                if bm_fc2 == GEMM_BM {
                    pass.set_bind_group(0, &b.fc2, &[]);
                } else {
                    pass.set_bind_group(0, &b.fc2_m64, &[]);
                }
                pass.dispatch_workgroups(gx_fc2, gy_fc2, 1);
                pass.set_pipeline(&add_p);
                pass.set_bind_group(0, &self.shared_bg.add, &[]);
                pass.dispatch_workgroups((seq_u * h_u).div_ceil(256), 1, 1);
            }
            pass.set_pipeline(&ln_p);
            pass.set_bind_group(0, &self.shared_bg.final_ln, &[]);
            pass.dispatch_workgroups(seq_u, 1, 1);
        }
        let rec_ms = t_rec.elapsed().as_secs_f64() * 1e3;
        let t_sub = std::time::Instant::now();
        self.k.submit(enc);
        let sub_ms = t_sub.elapsed().as_secs_f64() * 1e3;
        if std::env::var("GPU_PROFILE").is_ok() {
            let t_gpu = std::time::Instant::now();
            let _ = self.k.gpu().device.poll(wgpu::PollType::wait_indefinitely());
            let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "profile seq={seq} valid={valid}: upload {up_ms:.1} ms, uniforms {uni_ms:.1} ms, \
                 record {rec_ms:.1} ms, submit {sub_ms:.1} ms, poll {gpu_ms:.1} ms"
            );
        }
        Ok(())
    }

    /// Full head: encoder frames -> `(frames * up, num_speakers)` logits.
    pub fn head_forward(&mut self, encoder_out: &[f32], frames: usize) -> Result<Vec<f32>> {
        let x = self.scratch.n.clone();
        self.k.gpu().upload_f32(&x, encoder_out);
        self.head_on_gpu(frames, false)
    }

    fn head_on_gpu(&mut self, frames: usize, _from_device: bool) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let hh = self.head_cfg.hidden_size;
        let up = self.sub;
        let ns = self.head_cfg.num_speakers;
        if frames == 0 || frames > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!("head frames {frames} outside 1..={MAX_SEQ}")));
        }
        let fu = frames as u32;
        let up_n = fu * up as u32;
        self.k.write_uniform(&self.unis.proj, fu, hh as u32, h as u32, 1);
        self.k.write_uniform(&self.unis.conv, fu, hh as u32, (hh * up) as u32, 0);
        self.k.write_uniform(&self.unis.relu, up_n * hh as u32, 1, 0, 0);
        self.k.write_uniform(&self.unis.dense, up_n, hh as u32, hh as u32, 1);
        self.k.write_uniform(&self.unis.outp, up_n, ns as u32, hh as u32, 1);

        let gemm_p = self.k.pipes().gemm.clone();
        self.k.write_uniform(&self.unis.conv_gemm, fu, (hh * up) as u32, (hh * 3) as u32, 1);
        let unary_p = self.k.pipes().unary.clone();
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("head"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &self.shared_bg.proj, &[]);
            pass.dispatch_workgroups((hh as u32).div_ceil(GEMM_BN), fu.div_ceil(gemm_bm()), 1);
            self.k.col3(&mut pass, &self.scratch.proj, &self.scratch.conv_col, fu, hh as u32);
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &self.shared_bg.conv_gemm, &[]);
            pass.dispatch_workgroups(
                (hh * up) as u32 / GEMM_BN,
                fu.div_ceil(gemm_bm()),
                1,
            );
            pass.set_pipeline(&unary_p);
            pass.set_bind_group(0, &self.shared_bg.relu_up, &[]);
            pass.dispatch_workgroups((up_n * hh as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &self.shared_bg.dense, &[]);
            pass.dispatch_workgroups((hh as u32).div_ceil(GEMM_BN), up_n.div_ceil(gemm_bm()), 1);
            pass.set_pipeline(&unary_p);
            pass.set_bind_group(0, &self.shared_bg.relu_d, &[]);
            pass.dispatch_workgroups((up_n * hh as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(&gemm_p);
            pass.set_bind_group(0, &self.shared_bg.outp, &[]);
            pass.dispatch_workgroups((ns as u32).div_ceil(GEMM_BN), up_n.div_ceil(gemm_bm()), 1);
        }
        let t = std::time::Instant::now();
        self.k.submit(enc);
        let t_submit = t.elapsed().as_secs_f64() * 1e3;
        let out = self.k.gpu().readback_f32(&self.scratch.logits, frames * up * ns);
        if std::env::var("HEAD_PROFILE").is_ok() {
            eprintln!(
                "    head frames={frames} up_n={up_n}: submit {t_submit:.1} ms, \
                 readback {:.1} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        out
    }

    /// Embedder: stacked log-mel `[groups, sub * n_mels]` -> `[groups, hidden]`.
    pub fn embed(&mut self, stacked: &[f32], groups: usize) -> Result<Vec<f32>> {
        let inn = self.embed.inn;
        let h = self.cfg.hidden_size;
        if groups > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!("embed groups {groups} > {MAX_SEQ}")));
        }
        let stacked_b = self.scratch.stacked.clone();
        let x = self.scratch.x.clone();
        let w = self.embed.w.clone();
        self.k.gpu().upload_f32(&stacked_b, stacked);
        self.k.begin();
        let mut enc = self.k.encoder();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed"),
                timestamp_writes: None,
            });
            self.k.gemm(&mut pass, &stacked_b, &w, None, &x, groups as u32, h as u32, inn as u32);
        }
        self.k.submit(enc);
        self.k.gpu().readback_f32(&x, groups * h)
    }
}
