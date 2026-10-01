//! CUDA + cuBLAS encoder/head. Same numerics contract as the wgpu path, but
//! linears go through cuBLAS (the same library the Python reference uses).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::config::{AudioConfig, HeadConfig, ModelConfig};
use crate::cuda_kernels;
use crate::error::{DiarizationError, Result};
use crate::weights::Weights;

const MAX_SEQ: usize = 1024;
const P: &str = "model.audio_tower";
const LN_THREADS: u32 = 256;
/// Streaming window is at most cache(264)+fifo(264)+chunk(13) = 541.
const STREAM_BUCKET: usize = 32;
const STREAM_MAX: usize = 544;
const CUBLAS_WORKSPACE: usize = 32 * 1024 * 1024;

fn gpu_err(e: impl std::fmt::Display) -> DiarizationError {
    DiarizationError::Gpu(e.to_string())
}

fn nvidia_lib_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../.venv/lib/python3.12/site-packages/nvidia");
    ["cublas/lib", "cuda_runtime/lib", "cuda_nvrtc/lib", "nvjitlink/lib"]
        .iter()
        .map(|s| root.join(s))
        .filter_map(|p| p.canonicalize().ok())
        .collect()
}

fn setup_nvidia_libs() {
    let dirs = nvidia_lib_dirs();
    let extra = dirs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":");
    if !extra.is_empty() {
        let old = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new = if old.is_empty() { extra } else { format!("{extra}:{old}") };
        std::env::set_var("LD_LIBRARY_PATH", new);
    }
    // Keep the handles alive so subsequent `dlopen("libnvrtc.so.12")` can bind.
    static LIBS: std::sync::OnceLock<Vec<libloading::Library>> = std::sync::OnceLock::new();
    LIBS.get_or_init(|| {
        let mut out = Vec::new();
        let names = [
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudart.so.12",
            "libnvrtc.so.12",
            "libnvJitLink.so.12",
        ];
        for dir in &dirs {
            for name in names {
                let p = dir.join(name);
                if let Ok(lib) = unsafe {
                    libloading::os::unix::Library::open(
                        Some(&p),
                        libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                    )
                } {
                    out.push(lib.into());
                }
            }
        }
        out
    });
}

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

struct LinearW {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    out: usize,
    inn: usize,
}

struct LnW {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct BlockW {
    ln1: LnW,
    qkv: LinearW,
    o: LinearW,
    ln2: LnW,
    fc1: LinearW,
    fc2: LinearW,
}

struct Fns {
    ln: CudaFunction,
    gelu: CudaFunction,
    relu: CudaFunction,
    add: CudaFunction,
    bias: CudaFunction,
    add_bias: CudaFunction,
    bias_gelu: CudaFunction,
    rope: CudaFunction,
    softmax: CudaFunction,
    conv: CudaFunction,
}

pub struct CudaEngine {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    fns: Fns,
    cfg: AudioConfig,
    head_cfg: HeadConfig,
    sub: usize,
    embed: LinearW,
    input_ln: LnW,
    layers: Vec<BlockW>,
    final_ln: LnW,
    proj: LinearW,
    conv_w: CudaSlice<f32>,
    conv_b: CudaSlice<f32>,
    dense: LinearW,
    out: LinearW,
    x: CudaSlice<f32>,
    n: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    ff: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    proj_act: CudaSlice<f32>,
    up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    inp: CudaSlice<f32>,
    mel_buf: CudaSlice<f32>,
    emb_buf: CudaSlice<f32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    d_valid: CudaSlice<i32>,
    /// Held so cuBLAS graph capture does not allocate.
    _workspace: CudaSlice<u8>,
    graphs: HashMap<usize, CudaGraph>,
    /// Frames at the front of `inp` that already hold the next streaming prefix.
    reuse_frames: usize,
}

impl CudaEngine {
    pub fn load(model: &ModelConfig, w: &Weights) -> Result<Self> {
        setup_nvidia_libs();
        let ctx = CudaContext::new(0).map_err(gpu_err)?;
        // Non-default stream: the null stream cannot be graph-captured.
        let stream = ctx.new_stream().map_err(gpu_err)?;
        let ptx = compile_ptx_with_opts(
            cuda_kernels::SRC,
            CompileOptions {
                arch: Some("compute_61"),
                name: Some("n3d".into()),
                ..Default::default()
            },
        )
        .map_err(gpu_err)?;
        let module = ctx.load_module(ptx).map_err(gpu_err)?;
        let fns = Fns {
            ln: module.load_function("layer_norm").map_err(gpu_err)?,
            gelu: module.load_function("gelu").map_err(gpu_err)?,
            relu: module.load_function("relu").map_err(gpu_err)?,
            add: module.load_function("add_inplace").map_err(gpu_err)?,
            bias: module.load_function("bias_add").map_err(gpu_err)?,
            add_bias: module.load_function("add_bias").map_err(gpu_err)?,
            bias_gelu: module.load_function("bias_gelu").map_err(gpu_err)?,
            rope: module.load_function("rope").map_err(gpu_err)?,
            softmax: module.load_function("softmax_rows").map_err(gpu_err)?,
            conv: module.load_function("conv1d_k3").map_err(gpu_err)?,
        };
        let blas = CudaBlas::new(stream.clone()).map_err(gpu_err)?;
        let mut workspace = stream.alloc_zeros::<u8>(CUBLAS_WORKSPACE).map_err(gpu_err)?;
        {
            let (ptr, _guard) = workspace.device_ptr_mut(&stream);
            unsafe {
                cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *blas.handle(),
                    ptr as *mut _,
                    CUBLAS_WORKSPACE,
                )
                .result()
                .map_err(gpu_err)?;
            }
        }

        let cfg = model.audio_config.clone();
        let head_cfg = model.head_config.clone();
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hh = head_cfg.hidden_size;
        let up = cfg.subsampling_factor;
        let ns = head_cfg.num_speakers;
        let kv = cfg.num_key_value_heads;
        let hd = h / cfg.num_attention_heads;

        let upload = |s: &Arc<CudaStream>, data: &[f32]| -> Result<CudaSlice<f32>> {
            s.memcpy_stod(data).map_err(gpu_err)
        };
        let lin = |s: &Arc<CudaStream>, name: &str, out: usize, inn: usize, bias: Option<&str>| -> Result<LinearW> {
            Ok(LinearW {
                w: upload(s, w.tensor(name, &[out, inn])?)?,
                b: match bias {
                    Some(bn) => Some(upload(s, w.tensor(bn, &[out])?)?),
                    None => None,
                },
                out,
                inn,
            })
        };
        let ln = |s: &Arc<CudaStream>, prefix: &str| -> Result<LnW> {
            Ok(LnW {
                w: upload(s, w.tensor(&format!("{prefix}.weight"), &[h])?)?,
                b: upload(s, w.tensor(&format!("{prefix}.bias"), &[h])?)?,
            })
        };

        let embed = lin(
            &stream,
            &format!("{P}.embedder.projection.weight"),
            h,
            cfg.subsampling_factor * cfg.num_mel_bins,
            None,
        )?;
        let input_ln = ln(&stream, &format!("{P}.input_layer_norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let b = format!("{P}.layers.{i}");
            let qw = w.tensor(&format!("{b}.self_attn.q_proj.weight"), &[h, h])?;
            let kw = w.tensor(&format!("{b}.self_attn.k_proj.weight"), &[kv * hd, h])?;
            let vw = w.tensor(&format!("{b}.self_attn.v_proj.weight"), &[h, h])?;
            let mut packed = vec![0.0f32; 3 * h * h];
            packed[..h * h].copy_from_slice(qw);
            packed[h * h..2 * h * h].copy_from_slice(kw);
            packed[2 * h * h..].copy_from_slice(vw);
            layers.push(BlockW {
                ln1: ln(&stream, &format!("{b}.layer_norm1"))?,
                qkv: LinearW {
                    w: upload(&stream, &packed)?,
                    b: None,
                    out: 3 * h,
                    inn: h,
                },
                o: lin(
                    &stream,
                    &format!("{b}.self_attn.o_proj.weight"),
                    h,
                    h,
                    Some(&format!("{b}.self_attn.o_proj.bias")),
                )?,
                ln2: ln(&stream, &format!("{b}.layer_norm2"))?,
                fc1: lin(
                    &stream,
                    &format!("{b}.mlp.fc1.weight"),
                    inter,
                    h,
                    Some(&format!("{b}.mlp.fc1.bias")),
                )?,
                fc2: lin(
                    &stream,
                    &format!("{b}.mlp.fc2.weight"),
                    h,
                    inter,
                    Some(&format!("{b}.mlp.fc2.bias")),
                )?,
            });
        }
        let final_ln = ln(&stream, &format!("{P}.layer_norm"))?;
        let proj = lin(&stream, "model.proj.weight", hh, h, Some("model.proj.bias"))?;
        let conv_w = upload(&stream, w.tensor("model.upsampler.conv.weight", &[hh * up, hh, 3])?)?;
        let conv_b = upload(&stream, w.tensor("model.upsampler.conv.bias", &[hh * up])?)?;
        let dense = lin(&stream, "classifier.dense.weight", hh, hh, Some("classifier.dense.bias"))?;
        let out = lin(
            &stream,
            "classifier.out_proj.weight",
            ns,
            hh,
            Some("classifier.out_proj.bias"),
        )?;

        let alloc = |s: &Arc<CudaStream>, n: usize| -> Result<CudaSlice<f32>> {
            s.alloc_zeros::<f32>(n.max(1)).map_err(gpu_err)
        };
        let nh = cfg.num_attention_heads;
        let n_mels = cfg.num_mel_bins;
        let (cos, sin) = rope_tables(MAX_SEQ, hd, cfg.rope_parameters.rope_theta);

        Ok(Self {
            stream: stream.clone(),
            blas,
            fns,
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
            x: alloc(&stream, MAX_SEQ * h)?,
            n: alloc(&stream, MAX_SEQ * h)?,
            qkv: alloc(&stream, MAX_SEQ * 3 * h)?,
            ctx: alloc(&stream, MAX_SEQ * h)?,
            ff: alloc(&stream, MAX_SEQ * inter)?,
            scores: alloc(&stream, nh * MAX_SEQ * MAX_SEQ)?,
            proj_act: alloc(&stream, MAX_SEQ * up * hh)?,
            up: alloc(&stream, MAX_SEQ * up * hh)?,
            logits: alloc(&stream, MAX_SEQ * up * ns)?,
            inp: alloc(&stream, MAX_SEQ * h)?,
            mel_buf: alloc(&stream, 4096 * up * n_mels)?,
            emb_buf: alloc(&stream, 4096 * h)?,
            cos: upload(&stream, &cos)?,
            sin: upload(&stream, &sin)?,
            d_valid: stream.alloc_zeros::<i32>(1).map_err(gpu_err)?,
            _workspace: workspace,
            graphs: HashMap::new(),
            reuse_frames: 0,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "NVIDIA CUDA + cuBLAS (sm_61, fp32, {} graphs)",
            self.graphs.len()
        )
    }

    fn ln(
        stream: &Arc<CudaStream>,
        f: &CudaFunction,
        x: &CudaSlice<f32>,
        w: &LnW,
        y: &mut CudaSlice<f32>,
        rows: i32,
        cols: i32,
    ) -> Result<()> {
        let mut b = stream.launch_builder(f);
        b.arg(x);
        b.arg(&w.w);
        b.arg(&w.b);
        b.arg(y);
        b.arg(&rows);
        b.arg(&cols);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (LN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(gpu_err)?;
        Ok(())
    }

    fn unary(stream: &Arc<CudaStream>, f: &CudaFunction, x: &mut CudaSlice<f32>, n: i32) -> Result<()> {
        let mut b = stream.launch_builder(f);
        b.arg(x);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n.max(1) as u32)) }.map_err(gpu_err)?;
        Ok(())
    }

    fn add(
        stream: &Arc<CudaStream>,
        f: &CudaFunction,
        c: &mut CudaSlice<f32>,
        bsrc: &CudaSlice<f32>,
        n: i32,
    ) -> Result<()> {
        let mut b = stream.launch_builder(f);
        b.arg(c);
        b.arg(bsrc);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n.max(1) as u32)) }.map_err(gpu_err)?;
        Ok(())
    }

    /// Row-major `Y[M,N] = X[M,K] @ W[N,K]^T (+ bias[N])`.
    fn linear(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        bias_fn: &CudaFunction,
        x: &CudaSlice<f32>,
        lin: &LinearW,
        y: &mut CudaSlice<f32>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n: m,
            k,
            alpha: 1.0,
            lda: k,
            ldb: k,
            beta: 0.0,
            ldc: n,
        };
        unsafe { blas.gemm(cfg, &lin.w, x, y) }.map_err(gpu_err)?;
        if let Some(bias) = &lin.b {
            let mut b = stream.launch_builder(bias_fn);
            b.arg(y);
            b.arg(bias);
            b.arg(&m);
            b.arg(&n);
            unsafe { b.launch(LaunchConfig::for_num_elems((m * n).max(1) as u32)) }.map_err(gpu_err)?;
        }
        Ok(())
    }

    fn rope(
        stream: &Arc<CudaStream>,
        f: &CudaFunction,
        x: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        seq: i32,
        heads: i32,
        hd: i32,
    ) -> Result<()> {
        let n = seq * heads;
        let mut b = stream.launch_builder(f);
        b.arg(x);
        b.arg(cos);
        b.arg(sin);
        b.arg(&seq);
        b.arg(&heads);
        b.arg(&hd);
        unsafe { b.launch(LaunchConfig::for_num_elems(n.max(1) as u32)) }.map_err(gpu_err)?;
        Ok(())
    }

    fn attn(&mut self, seq: i32, heads: i32, hd: i32, valid: i32) -> Result<()> {
        let hidden = heads * hd;
        let sseq = seq as i64;
        let shd = hd as i64;
        // S[h,q,k] = Q[q, h, :] @ K[k, h, :]^T
        let qk = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: seq,
            n: seq,
            k: hd,
            alpha: 1.0 / (hd as f32).sqrt(),
            lda: hidden,
            ldb: hidden,
            beta: 0.0,
            ldc: seq,
        };
        let qk_b = StridedBatchedConfig {
            gemm: qk,
            batch_size: heads,
            stride_a: shd,
            stride_b: shd,
            stride_c: sseq * sseq,
        };
        unsafe { self.blas.gemm_strided_batched(qk_b, &self.k, &self.q, &mut self.scores) }
            .map_err(gpu_err)?;

        let mut b = self.stream.launch_builder(&self.fns.softmax);
        b.arg(&mut self.scores);
        b.arg(&seq);
        b.arg(&heads);
        b.arg(&valid);
        let cfg = LaunchConfig {
            grid_dim: (seq as u32, heads as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(gpu_err)?;

        // ctx = S @ V
        let pv = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: hd,
            n: seq,
            k: seq,
            alpha: 1.0,
            lda: hidden,
            ldb: seq,
            beta: 0.0,
            ldc: hidden,
        };
        let pv_b = StridedBatchedConfig {
            gemm: pv,
            batch_size: heads,
            stride_a: shd,
            stride_b: sseq * sseq,
            stride_c: shd,
        };
        unsafe { self.blas.gemm_strided_batched(pv_b, &self.v, &self.scores, &mut self.ctx) }
            .map_err(gpu_err)?;
        Ok(())
    }

    pub fn forward_embeds(&mut self, embeds: &[f32], valid: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let seq = embeds.len() / h;
        self.encode(embeds, valid)?;
        self.stream.memcpy_dtov(&self.n).map_err(gpu_err).map(|mut v| {
            v.truncate(seq * h);
            v
        })
    }

    pub fn forward_window(&mut self, embeds: &[f32], valid: usize) -> Result<Vec<f32>> {
        let seq = embeds.len() / self.cfg.hidden_size;
        if std::env::var("CUDA_PROFILE").is_ok() {
            eprintln!("forward_window seq={seq} valid={valid} graphs={}", self.graphs.len());
        }
        if valid == seq && self.graphs.contains_key(&seq) {
            self.stream.memcpy_htod(embeds, &mut self.inp).map_err(gpu_err)?;
            self.graphs[&seq].launch().map_err(gpu_err)?;
            let n = seq * self.sub * self.head_cfg.num_speakers;
            let view = self.logits.slice(0..n);
            return self.stream.memcpy_dtov(&view).map_err(gpu_err);
        }
        self.encode(embeds, valid)?;
        self.head_from_n(seq)
    }

    pub fn head_forward(&mut self, encoder_out: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.stream.memcpy_htod(encoder_out, &mut self.n).map_err(gpu_err)?;
        self.head_from_n(frames)
    }

    fn encode(&mut self, embeds: &[f32], valid: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let seq = embeds.len() / h;
        if seq == 0 || seq > MAX_SEQ {
            return Err(DiarizationError::Gpu(format!("seq {seq} outside 1..={MAX_SEQ}")));
        }
        self.stream.memcpy_htod(embeds, &mut self.inp).map_err(gpu_err)?;
        self.encode_compute(seq, valid)
    }

    fn encode_compute(&mut self, seq: usize, valid: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let valid = valid.min(seq) as i32;
        let seq_i = seq as i32;
        let h_i = h as i32;
        let nh = self.cfg.num_attention_heads as i32;
        let hd = (h / self.cfg.num_attention_heads) as i32;
        let inter = self.cfg.intermediate_size as i32;
        let t0 = std::time::Instant::now();
        Self::ln(&self.stream, &self.fns.ln, &self.inp, &self.input_ln, &mut self.x, seq_i, h_i)?;

        for i in 0..self.layers.len() {
            Self::ln(&self.stream, &self.fns.ln, &self.x, &self.layers[i].ln1, &mut self.n, seq_i, h_i)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.n, &self.layers[i].q, &mut self.q, seq_i, h_i, h_i)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.n, &self.layers[i].k, &mut self.k, seq_i, h_i, h_i)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.n, &self.layers[i].v, &mut self.v, seq_i, h_i, h_i)?;
            Self::rope(&self.stream, &self.fns.rope, &mut self.q, &self.cos, &self.sin, seq_i, nh, hd)?;
            Self::rope(&self.stream, &self.fns.rope, &mut self.k, &self.cos, &self.sin, seq_i, nh, hd)?;
            self.attn(seq_i, nh, hd, valid)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.ctx, &self.layers[i].o, &mut self.n, seq_i, h_i, h_i)?;
            Self::add(&self.stream, &self.fns.add, &mut self.x, &self.n, seq_i * h_i)?;

            Self::ln(&self.stream, &self.fns.ln, &self.x, &self.layers[i].ln2, &mut self.n, seq_i, h_i)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.n, &self.layers[i].fc1, &mut self.ff, seq_i, inter, h_i)?;
            Self::unary(&self.stream, &self.fns.gelu, &mut self.ff, seq_i * inter)?;
            Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.ff, &self.layers[i].fc2, &mut self.n, seq_i, h_i, inter)?;
            Self::add(&self.stream, &self.fns.add, &mut self.x, &self.n, seq_i * h_i)?;
        }
        Self::ln(&self.stream, &self.fns.ln, &self.x, &self.final_ln, &mut self.n, seq_i, h_i)?;
        if std::env::var("CUDA_PROFILE").is_ok() {
            self.stream.synchronize().map_err(gpu_err)?;
            eprintln!("cuda encode seq={seq} {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Touch the GEMM shapes used in offline/streaming so cuBLAS picks algorithms
    /// before the timed run.
    pub fn warmup(&mut self) -> Result<()> {
        let h = self.cfg.hidden_size;
        for seq in [64usize, 308, 317, 340, 380, 512, 684] {
            let dummy = vec![0.0f32; seq * h];
            self.encode(&dummy, seq)?;
        }
        let _ = self.head_from_n(64)?;
        self.stream.synchronize().map_err(gpu_err)?;
        if std::env::var("CUDA_GRAPH").is_err() {
            return Ok(());
        }
        for seq in [308usize, 312, 317] {
            let dummy = vec![0.0f32; seq * h];
            self.stream.memcpy_htod(&dummy, &mut self.inp).map_err(gpu_err)?;
            if let Err(e) = self
                .stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                eprintln!("cuda graph begin_capture: {e}");
                break;
            }
            let captured = self.encode_compute(seq, seq).and_then(|_| self.head_from_n(seq).map(|_| ()));
            match (captured, self.stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)) {
                (Ok(()), Ok(Some(g))) => {
                    self.graphs.insert(seq, g);
                }
                (Err(e), _) => {
                    let _ = self.stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD);
                    eprintln!("cuda graph capture seq={seq}: {e}");
                }
                (_, Err(e)) => eprintln!("cuda graph instantiate seq={seq}: {e}"),
                (Ok(()), Ok(None)) => eprintln!("cuda graph seq={seq}: empty capture"),
            }
        }
        eprintln!("cuda graphs: {} {:?}", self.graphs.len(), self.graphs.keys().collect::<Vec<_>>());
        Ok(())
    }

    pub fn embed(&mut self, features: &[f32], num_frames: usize) -> Result<Vec<f32>> {
        let sub = self.cfg.subsampling_factor;
        let mels = self.cfg.num_mel_bins;
        let h = self.cfg.hidden_size;
        let groups = num_frames.div_ceil(sub);
        let mut buf = vec![0.0f32; groups * sub * mels];
        let n_copy = (num_frames * mels).min(features.len());
        buf[..n_copy].copy_from_slice(&features[..n_copy]);
        self.stream.memcpy_htod(&buf, &mut self.mel_buf).map_err(gpu_err)?;
        Self::linear(
            &self.blas,
            &self.stream,
            &self.fns.bias,
            &self.mel_buf,
            &self.embed,
            &mut self.emb_buf,
            groups as i32,
            h as i32,
            (sub * mels) as i32,
        )?;
        let view = self.emb_buf.slice(0..groups * h);
        self.stream.memcpy_dtov(&view).map_err(gpu_err)
    }

    fn head_from_n(&mut self, frames: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size as i32;
        let hh = self.head_cfg.hidden_size as i32;
        let up = self.sub as i32;
        let ns = self.head_cfg.num_speakers as i32;
        let fu = frames as i32;
        let up_n = fu * up;
        Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.n, &self.proj, &mut self.proj_act, fu, hh, h)?;
        let mut b = self.stream.launch_builder(&self.fns.conv);
        b.arg(&self.proj_act);
        b.arg(&self.conv_w);
        b.arg(&self.conv_b);
        b.arg(&mut self.up);
        b.arg(&fu);
        b.arg(&hh);
        let out_c = hh * up;
        b.arg(&out_c);
        let cfg = LaunchConfig {
            grid_dim: ((fu as u32).div_ceil(8), (out_c as u32).div_ceil(8), 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(gpu_err)?;
        Self::unary(&self.stream, &self.fns.relu, &mut self.up, up_n * hh)?;
        Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.up, &self.dense, &mut self.proj_act, up_n, hh, hh)?;
        Self::unary(&self.stream, &self.fns.relu, &mut self.proj_act, up_n * hh)?;
        Self::linear(&self.blas, &self.stream, &self.fns.bias, &self.proj_act, &self.out, &mut self.logits, up_n, ns, hh)?;
        let mut out = self.stream.memcpy_dtov(&self.logits).map_err(gpu_err)?;
        out.truncate(frames * self.sub * self.head_cfg.num_speakers);
        Ok(out)
    }
}
