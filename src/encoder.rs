//! The audio tower: 31 pre-LN transformer blocks over 80 ms encoder frames.
//!
//! A direct port of `Nemotron3DiarizationAudioModel` in
//! `transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py`.
//! The block is Llama-shaped, and two details are easy to get wrong:
//!
//! * **q/k/v have no bias, `o_proj` does.** `nn.Linear(..., bias=False)` on the fused
//!   projections is deliberate; adding a bias shifts every attention score.
//! * **there is no GQA.** `num_key_value_heads == num_attention_heads == 8`, so the
//!   `repeat_kv` in the reference is a no-op and head count can be assumed equal.
//!
//! RoPE is applied on the full head dim (64) with `theta = 10000`; `cos`/`sin` are
//! built by concatenating the frequency vector with itself, and `rotate_half` splits
//! each head in two — the Llama convention, not GPT-NeoX's interleaved one.

use rayon::prelude::*;

use crate::config::AudioConfig;
use crate::error::Result;
use crate::tensor::{gelu, layer_norm, linear, softmax_last};
use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

pub struct Tower {
    pub cfg: AudioConfig,
    embed_proj: Vec<f32>, // [hidden, sub * n_mels]
    input_ln_w: Vec<f32>,
    input_ln_b: Vec<f32>,
    layers: Vec<Block>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

struct Block {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    o_bias: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2: Vec<f32>,
    fc2_bias: Vec<f32>,
}

/// One block's output, captured for the layer-by-layer diff.
pub struct LayerSnapshot {
    pub layer: usize,
    pub states: Vec<f32>,
}

const P: &str = "model.audio_tower";

impl Tower {
    pub fn load(cfg: &AudioConfig, w: &Weights) -> Result<Self> {
        let h = cfg.hidden_size;
        let get = |n: &str, shape: &[usize]| -> Result<Vec<f32>> {
            Ok(w.tensor(n, shape)?.to_vec())
        };

        let embed_proj = get(
            &format!("{P}.embedder.projection.weight"),
            &[h, cfg.subsampling_factor * cfg.num_mel_bins],
        )?;
        let input_ln_w = get(&format!("{P}.input_layer_norm.weight"), &[h])?;
        let input_ln_b = get(&format!("{P}.input_layer_norm.bias"), &[h])?;

        let kv = cfg.num_key_value_heads;
        let hd = h / cfg.num_attention_heads;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let b = format!("{P}.layers.{i}");
            layers.push(Block {
                ln1_w: get(&format!("{b}.layer_norm1.weight"), &[h])?,
                ln1_b: get(&format!("{b}.layer_norm1.bias"), &[h])?,
                q: get(&format!("{b}.self_attn.q_proj.weight"), &[h, h])?,
                k: get(&format!("{b}.self_attn.k_proj.weight"), &[kv * hd, h])?,
                v: get(&format!("{b}.self_attn.v_proj.weight"), &[h, h])?,
                o: get(&format!("{b}.self_attn.o_proj.weight"), &[h, h])?,
                o_bias: get(&format!("{b}.self_attn.o_proj.bias"), &[h])?,
                ln2_w: get(&format!("{b}.layer_norm2.weight"), &[h])?,
                ln2_b: get(&format!("{b}.layer_norm2.bias"), &[h])?,
                fc1: get(&format!("{b}.mlp.fc1.weight"), &[cfg.intermediate_size, h])?,
                fc1_bias: get(&format!("{b}.mlp.fc1.bias"), &[cfg.intermediate_size])?,
                fc2: get(&format!("{b}.mlp.fc2.weight"), &[h, cfg.intermediate_size])?,
                fc2_bias: get(&format!("{b}.mlp.fc2.bias"), &[h])?,
            });
        }

        let final_ln_w = get(&format!("{P}.layer_norm.weight"), &[h])?;
        let final_ln_b = get(&format!("{P}.layer_norm.bias"), &[h])?;

        Ok(Self {
            cfg: cfg.clone(),
            embed_proj,
            input_ln_w,
            input_ln_b,
            layers,
            final_ln_w,
            final_ln_b,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Stack `subsampling_factor` frames and project to `hidden_size`.
    ///
    /// The last group is zero-padded, matching `nn.functional.pad` in the reference.
    /// The 1024-vector is **time-major**: `[f0(128), f1(128), …, f7(128)]`.
    pub fn embed(&self, input_features: &[f32], num_frames: usize) -> Vec<f32> {
        self.embed_range(input_features, num_frames, 0, num_frames.div_ceil(self.cfg.subsampling_factor))
    }

    /// Encoder groups `lo..hi` only. Group `g` reads mel frames `g*sub .. g*sub+sub`,
    /// so a range is a contiguous slice of `input_features` and the projection is
    /// the same `linear` call on a shorter row count — no per-group index math, and
    /// **bit-identical to the corresponding slice of [`embed`]** because each output
    /// row is a dot product over the same operands in the same order.
    pub fn embed_range(
        &self,
        input_features: &[f32],
        num_frames: usize,
        lo: usize,
        hi: usize,
    ) -> Vec<f32> {
        let (sub, mels, h) = (self.cfg.subsampling_factor, self.cfg.num_mel_bins, self.cfg.hidden_size);
        let groups = hi - lo;
        // frames past `num_frames` stay zero, exactly as the whole-recording form
        // does — that padding is load-bearing (it lands in the last group).
        let end_mel = (hi * sub).min(num_frames);
        let mut buf = vec![0.0f32; groups * sub * mels];
        if lo * sub < end_mel {
            let src = &input_features[lo * sub * mels..end_mel * mels];
            buf[..src.len()].copy_from_slice(src);
        }
        linear(&buf, groups, sub * mels, &self.embed_proj, h, None)
    }

    /// The encoder's input `layer_norm`, exposed so the reference's `input_ln` dump
    /// can be diffed without duplicating the weight lookup here.
    pub fn apply_input_ln(&self, embeds: &[f32], groups: usize) -> Vec<f32> {
        layer_norm(
            embeds,
            groups,
            self.cfg.hidden_size,
            &self.input_ln_w,
            &self.input_ln_b,
        )
    }

    /// RoPE `cos`/`sin` tables, `(seq, head_dim)` each.
    ///
    /// `inv_freq[i] = 1 / theta^(2i / head_dim)` for `i` in `0..head_dim/2`, then the
    /// frequency vector is duplicated — `cat((freqs, freqs))` — so `cos` has `head_dim`
    /// columns. `partial_rotary_factor` is 1.0 here, i.e. the whole head is rotated.
    pub fn rope_tables(&self, seq: usize, offset: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let half = hd / 2;
        // computed in f64, exactly as the reference's `torch.float` inv_freq does
        let theta = self.cfg.rope_parameters.rope_theta;
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0f64 / theta.powf(2.0 * i as f64 / hd as f64)) as f32)
            .collect();
        let mut cos = vec![0.0f32; seq * hd];
        let mut sin = vec![0.0f32; seq * hd];
        for p in 0..seq {
            let pos = (offset + p) as f32;
            for i in 0..half {
                let a = pos * inv[i];
                cos[p * hd + i] = a.cos();
                sin[p * hd + i] = a.sin();
                // the duplicated half
                cos[p * hd + half + i] = cos[p * hd + i];
                sin[p * hd + half + i] = sin[p * hd + i];
            }
        }
        (cos, sin)
    }

    /// Run the tower over one chunk.
    ///
    /// `valid_frames` is the number of *encoder* frames that carry real audio; frames
    /// beyond it are masked out of attention exactly as the reference's bidirectional
    /// padding mask does. `snapshots` receives the output of the requested layers.
    pub fn forward(
        &self,
        input_features: &[f32],
        num_frames: usize,
        valid_frames: Option<usize>,
        position_offset: usize,
        snapshot_layers: &[usize],
    ) -> (Vec<f32>, Vec<LayerSnapshot>) {
        let embeds = self.embed(input_features, num_frames);
        self.forward_embeds(&embeds, valid_frames, position_offset, snapshot_layers)
    }

    /// [`forward`](Self::forward) for a caller that already holds the embedder output.
    ///
    /// Offline mode needs this: the reference runs `embedder` **once** over the whole
    /// recording and only then slices the result into chunks, so re-embedding per
    /// chunk would not be the same computation.
    pub fn forward_embeds(
        &self,
        inputs_embeds: &[f32],
        valid_frames: Option<usize>,
        position_offset: usize,
        snapshot_layers: &[usize],
    ) -> (Vec<f32>, Vec<LayerSnapshot>) {
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads;
        let hd = h / nh;
        let groups = inputs_embeds.len() / h;
        let valid = valid_frames.unwrap_or(groups).min(groups);

        let mut x = layer_norm(inputs_embeds, groups, h, &self.input_ln_w, &self.input_ln_b);
        let (cos, sin) = self.rope_tables(groups, position_offset);
        let mut snapshots = Vec::new();

        for (li, blk) in self.layers.iter().enumerate() {
            x = self.block(blk, &x, groups, nh, hd, valid, &cos, &sin);
            if snapshot_layers.contains(&li) {
                snapshots.push(LayerSnapshot { layer: li, states: x.clone() });
            }
        }
        let out = layer_norm(&x, groups, h, &self.final_ln_w, &self.final_ln_b);
        (out, snapshots)
    }

    fn block(
        &self,
        blk: &Block,
        x: &[f32],
        seq: usize,
        nh: usize,
        hd: usize,
        valid: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;

        // --- attention sub-block, pre-norm ---
        let n1 = layer_norm(x, seq, h, &blk.ln1_w, &blk.ln1_b);
        let q = linear(&n1, seq, h, &blk.q, h, None);
        let k = linear(&n1, seq, h, &blk.k, h, None);
        let v = linear(&n1, seq, h, &blk.v, h, None);

        let q = apply_rope(&q, seq, nh, hd, cos, sin);
        let k = apply_rope(&k, seq, nh, hd, cos, sin);

        // `ctx` is [seq, nh * hd] which is [seq, h] because nh * hd == hidden_size
        let ctx = attention(&q, &k, &v, seq, nh, hd, valid);
        let attn_out = linear(&ctx, seq, h, &blk.o, h, Some(&blk.o_bias));

        let mut x1 = x.to_vec();
        for i in 0..seq * h {
            x1[i] += attn_out[i];
        }

        // --- MLP sub-block, pre-norm ---
        let n2 = layer_norm(&x1, seq, h, &blk.ln2_w, &blk.ln2_b);
        let inter = self.cfg.intermediate_size;
        let ff = linear(&n2, seq, h, &blk.fc1, inter, Some(&blk.fc1_bias));
        let ff = gelu(&ff, seq * inter);
        let ff = linear(&ff, seq, inter, &blk.fc2, h, Some(&blk.fc2_bias));
        for i in 0..seq * h {
            x1[i] += ff[i];
        }
        x1
    }
}

/// `q' = q*cos + rotate_half(q)*sin`, per head, on the full head dim.
fn apply_rope(x: &[f32], seq: usize, nh: usize, hd: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    y.par_chunks_mut(hd).enumerate().for_each(|(row, head)| {
        let p = row / nh;
        let c = &cos[p * hd..(p + 1) * hd];
        let s = &sin[p * hd..(p + 1) * hd];
        let half = hd / 2;
        for i in 0..half {
            let x1 = head[i];
            let x2 = head[half + i];
            head[i] = x1 * c[i] - x2 * s[i];
            head[half + i] = x2 * c[half + i] + x1 * s[half + i];
        }
    });
    y
}

/// Softmax attention over `valid` keys, scaled by `head_dim^-0.5`.
///
/// The reference adds a large negative constant to masked *keys*, which is
/// mathematically the same as dropping them from the softmax but cheaper.
fn attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, nh: usize, hd: usize, valid: usize) -> Vec<f32> {
    let scaling = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; seq * nh * hd];
    out.par_chunks_mut(nh * hd).enumerate().for_each(|(qi, orow)| {
        for head in 0..nh {
            let qb = (qi * nh + head) * hd;
            let mut scores = vec![0.0f32; valid];
            for j in 0..valid {
                let kb = (j * nh + head) * hd;
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += q[qb + d] * k[kb + d];
                }
                scores[j] = acc * scaling;
            }
            softmax_last(&mut scores, 1, valid);
            let ob = head * hd;
            for d in 0..hd {
                let mut acc = 0.0f32;
                for j in 0..valid {
                    acc += scores[j] * v[(j * nh + head) * hd + d];
                }
                orow[ob + d] = acc;
            }
        }
    });
    out
}

/// Load the tower straight from a checkpoint directory.
pub fn load_tower(cfg: &AudioConfig, dir: &std::path::Path) -> Result<Tower> {
    Tower::load(cfg, &crate::weights::Weights::load(dir)?)
}
