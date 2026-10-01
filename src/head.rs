//! The classification head: encoder frames back to 10 ms speaker logits.
//!
//! Port of `Nemotron3DiarizationModel.forward` + `Nemotron3DiarizationClassificationHead`:
//!
//! ```text
//! encoder (T/8, 512)          <- final layer_norm output
//!   -> proj        512 -> 192  (with bias)
//!   -> upsampler   Conv1d 192 -> 192*8, kernel 3, padding 1, then a sub-pixel reshape x8
//!   -> classifier  out_proj(relu(dense(relu(x))))
//! ```
//!
//! Two things that are easy to get wrong:
//!
//! * **the sub-pixel conv is temporal, so it needs its neighbours.** The reference
//!   runs the head over `cached ++ chunk ++ look-ahead` and *then* slices out the
//!   scored window, because the kernel-3 conv at the first scored frame takes the last
//!   cached frame as its left tap. Running the head on the scored frames alone gives
//!   different values at both boundaries — a small diff that is very hard to see.
//! * **`silence_embeds` is not added to the encoder output.** It only fills the
//!   reserved silence slots when the speaker cache is compressed. Reading the weight
//!   list and assuming it is an encoder bias is a natural mistake.

use rayon::prelude::*;

use crate::config::{HeadConfig, ModelConfig};
use crate::error::Result;
use crate::tensor::{linear, relu};
use crate::weights::Weights;

/// Every intermediate of [`Head::trace`], for diffing one stage at a time.
#[derive(Debug, Clone)]
pub struct HeadTrace {
    pub projected: Vec<f32>,
    pub upsampled: Vec<f32>,
    pub r1: Vec<f32>,
    pub dense: Vec<f32>,
    pub r2: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct Head {
    pub cfg: HeadConfig,
    /// encoder width feeding `proj` (`audio_config.hidden_size`).
    in_dim: usize,
    up: usize,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    /// `[hidden * up, hidden, 3]` — PyTorch `Conv1d(out, in, k)` layout.
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dense_w: Vec<f32>,
    dense_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

const P: &str = "model";

impl Head {
    pub fn load(cfg: &HeadConfig, model: &ModelConfig, w: &Weights) -> Result<Self> {
        let up = model.audio_config.subsampling_factor;
        let h = model.audio_config.hidden_size;
        let hh = cfg.hidden_size;
        let get = |n: &str, shape: &[usize]| -> Result<Vec<f32>> {
            Ok(w.tensor(n, shape)?.to_vec())
        };
        Ok(Self {
            cfg: cfg.clone(),
            in_dim: h,
            up,
            proj_w: get(&format!("{P}.proj.weight"), &[hh, h])?,
            proj_b: get(&format!("{P}.proj.bias"), &[hh])?,
            conv_w: get(&format!("{P}.upsampler.conv.weight"), &[hh * up, hh, 3])?,
            conv_b: get(&format!("{P}.upsampler.conv.bias"), &[hh * up])?,
            dense_w: get("classifier.dense.weight", &[hh, hh])?,
            dense_b: get("classifier.dense.bias", &[hh])?,
            out_w: get("classifier.out_proj.weight", &[cfg.num_speakers, hh])?,
            out_b: get("classifier.out_proj.bias", &[cfg.num_speakers])?,
        })
    }

    pub fn num_speakers(&self) -> usize {
        self.cfg.num_speakers
    }

    /// `Conv1d(hidden -> hidden*up, k=3, padding=1)` followed by the sub-pixel reshape.
    ///
    /// `input` is `(frames, hidden)`, the channel-last form of PyTorch's
    /// `(batch, channels, time)`. `padding=1` grows the sequence by one zero on each
    /// side and `stride=1` keeps the length, so this returns `frames * up` rows of
    /// `hidden` — the row `(t * up + j)` is sub-pixel `j` of input frame `t`.
    fn upsample(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let (hh, up) = (self.cfg.hidden_size, self.up);
        let out_c = hh * up;

        // zero-padded, time-major: padded[t] is what PyTorch sees at index t
        let mut pad = vec![0.0f32; (frames + 2) * hh];
        for t in 0..frames {
            pad[(t + 1) * hh..(t + 2) * hh].copy_from_slice(&input[t * hh..(t + 1) * hh]);
        }

        let mut out = vec![0.0f32; frames * out_c];
        out.par_chunks_mut(out_c).enumerate().for_each(|(t, orow)| {
            let (p0, p1, p2) = (t * hh, (t + 1) * hh, (t + 2) * hh);
            for oc in 0..out_c {
                // w is [out_c, hidden, 3] -> row `oc` is `hidden` triples
                let w = &self.conv_w[oc * hh * 3..(oc + 1) * hh * 3];
                let mut acc = self.conv_b[oc];
                for ic in 0..hh {
                    acc += w[ic * 3] * pad[p0 + ic]
                        + w[ic * 3 + 1] * pad[p1 + ic]
                        + w[ic * 3 + 2] * pad[p2 + ic];
                }
                orow[oc] = acc;
            }
        });

        // sub-pixel: row t of the conv splits into `up` interleaved rows of `hidden`
        let mut shaped = vec![0.0f32; frames * up * hh];
        for t in 0..frames {
            for j in 0..up {
                let src = &out[t * out_c + j * hh..t * out_c + (j + 1) * hh];
                shaped[(t * up + j) * hh..(t * up + j + 1) * hh].copy_from_slice(src);
            }
        }
        shaped
    }

    /// The three stages of [`forward`](Self::forward), kept separate for diffing:
    /// `(projected, upsampled, logits)`.
    pub fn trace(&self, encoder_out: &[f32], frames: usize) -> HeadTrace {
        let hh = self.cfg.hidden_size;
        let projected = linear(encoder_out, frames, self.in_dim, &self.proj_w, hh, Some(&self.proj_b));
        let up = self.upsample(&projected, frames);
        let up_frames = frames * self.up;
        let r1 = relu(&up, up_frames * hh);
        let dense = linear(&r1, up_frames, hh, &self.dense_w, hh, Some(&self.dense_b));
        let r2 = relu(&dense, up_frames * hh);
        let logits = linear(&r2, up_frames, hh, &self.out_w, self.cfg.num_speakers, Some(&self.out_b));
        HeadTrace { projected, upsampled: up, r1, dense, r2, logits }
    }

    /// Full head: encoder frames -> `(frames * up, num_speakers)` logits.
    ///
    /// `encoder_out` must already include whatever context the caller needs on both
    /// sides — pass the whole `cached ++ chunk ++ look-ahead` window, then slice.
    pub fn forward(&self, encoder_out: &[f32], frames: usize) -> Vec<f32> {
        let hh = self.cfg.hidden_size;
        let projected = linear(encoder_out, frames, self.in_dim, &self.proj_w, hh, Some(&self.proj_b));
        let up = self.upsample(&projected, frames);
        let up_frames = frames * self.up;
        // out_proj(relu(dense(relu(x)))) -- a ReLU on *both* sides of the dense layer
        let r1 = relu(&up, up_frames * hh);
        let d = linear(&r1, up_frames, hh, &self.dense_w, hh, Some(&self.dense_b));
        let r2 = relu(&d, up_frames * hh);
        linear(&r2, up_frames, hh, &self.out_w, self.cfg.num_speakers, Some(&self.out_b))
    }
}
