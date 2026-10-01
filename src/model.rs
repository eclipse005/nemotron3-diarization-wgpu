//! End-to-end inference: front-end → audio tower → head → speaker cache.
//!
//! This is the port of `Nemotron3DiarizationForAudioFrameClassification.forward` plus
//! the chunking that `Nemotron3DiarizationProcessor` does around it. Two loops cover
//! every mode:
//!
//! * [`Model::run_offline`] — one stateless call over a whole recording. The embedder
//!   runs **once** over the whole file, the result is sliced into 340-encoder-frame
//!   chunks, and each chunk is `[AOSC ++ FIFO ++ chunk ++ 40 right-context]`.
//! * [`Model::run_streaming`] — one chunk per call, 9/6/3 scored frames depending on
//!   the mode, with 4/2/1 look-ahead frames held back.
//!
//! # Four conventions that are silently wrong if you guess them
//!
//! 1. **RoPE positions restart at 0 for every chunk.** They are `arange(seq_len)` of
//!    the *concatenated* window, not an offset into the recording.
//! 2. **`torch.stft(center=True)` emits one frame more than are valid.** The extra
//!    frame reads only the zero padding, and the processor then zeroes it through the
//!    attention mask, so it must be reproduced as an exact zero mel row — not
//!    recomputed, and not dropped (it changes the last encoder group).
//! 3. **The head sees the whole window, then the logits are sliced.** The kernel-3
//!    sub-pixel conv needs the frame before the first scored one, so running the head
//!    on the scored frames alone shifts the first and last scored rows.
//! 4. **The last streaming chunk is centred.** The reference processor keys `center=`
//!    off `is_first_audio_chunk`, which defaults to `True`; a session's final chunk is
//!    not marked otherwise, so it re-centres. That is why a streaming session ends up
//!    with the same frame count as an offline pass.

use std::cell::RefCell;
use std::path::Path;

use crate::config::{ModelConfig, ModeParams, ProcessorConfig, StreamingMode};
use crate::encoder::Tower;
use crate::error::Result;
use crate::gpu_engine::GpuEngine;
use crate::head::Head;
use crate::mel::{frame_count, log_mel, mel_filters, Padding};
use crate::streaming::{CacheConfig, SpeakerCache};
use crate::weights::Weights;

/// Everything one inference run produced.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// `(num_frames, num_speakers)` logits, pre-sigmoid, at the 10 ms frame rate.
    pub logits: Vec<f32>,
    pub num_frames: usize,
    /// Encoder forwards performed — one per chunk.
    pub num_chunks: usize,
    /// True once the speaker cache has been compressed at least once.
    pub cache_compressed: bool,
    pub cache_frames: usize,
    pub fifo_frames: usize,
}

/// One chunk's encoder window, recorded by [`Model::run_offline_capturing`].
#[derive(Debug, Clone)]
pub struct ChunkTrace {
    /// Frames carried over from the speaker cache and the FIFO.
    pub cached_frames: usize,
    /// `cached ++ chunk ++ look-ahead` — the sequence the tower actually saw.
    pub window_frames: usize,
    /// Frames the attention mask lets through.
    pub valid_frames: usize,
    /// Frames whose logits are emitted (the chunk itself, look-ahead excluded).
    pub scored_frames: usize,
    /// The tower's output on the window, `(window_frames, hidden)`.
    pub encoder_out: Vec<f32>,
    /// `_pool_probs` on this step's logits, before the cache update consumes it.
    pub step_probs: Vec<f32>,
    /// The cache as it stands *after* this step: selected frames, their stored
    /// probabilities, and whatever is still in the FIFO.
    pub cache_embeds: Vec<f32>,
    pub cache_probs: Vec<f32>,
    pub fifo_embeds: Vec<f32>,
}

/// One mel chunk handed to the model: its features, how many are real, and how many
/// trailing encoder frames are look-ahead (0 in offline mode, 0 in the last chunk).
pub struct Chunk {
    pub mel: Vec<f32>,
    pub num_mel_frames: usize,
    pub lookahead: usize,
}

/// What the per-chunk loop reads its encoder input from.
///
/// Offline and streaming differ only in *where* the embeddings come from: offline
/// embeds the whole file once and hands out sub-ranges, streaming embeds each chunk as
/// it arrives. Everything after that is identical.
enum Inputs<'a> {
    /// The offline path keeps the **mel** spectrogram, not the embeddings, and
    /// embeds one window at a time inside the loop. That is what removes the
    /// length limit: the whole-recording embedder output used to be materialised
    /// in one GPU buffer, so a long file overflowed it (and the CPU `Vec` with
    /// it). Every window is at most `chunk_length + chunk_right_context` groups,
    /// which is what the encoder scratch is sized for anyway.
    Offline {
        mel: &'a [f32],
        /// Frames handed to the embedder, `valid + 1` (see `run_offline`).
        num_mel_frames: usize,
        /// Real mel frames; the last encoder group's mask comes from this.
        valid: usize,
        /// `(start, end)` encoder-frame ranges, right-context already added.
        spans: Vec<(usize, usize)>,
        /// Scored frames of each window, which is `end - start` minus the right context.
        chunk_frames: Vec<usize>,
        /// Frames the *whole run* is truncated to, applied after concatenation.
        total_mel_frames: usize,
        capture: Option<&'a mut Vec<ChunkTrace>>,
    },
    Streaming {
        /// `(embeddings, scored encoder frames, mel frames)` per chunk.
        chunks: Vec<(Vec<f32>, usize, usize)>,
    },
    /// Diagnostic A/B twin of [`Inputs::Offline`]: the whole recording embedded up
    /// front. Needs a whole-recording buffer, so it hits the length cap; see
    /// `OFFLINE_WHOLEFILE`.
    OfflineWhole {
        embeds: &'a [f32],
        mask: &'a [bool],
        spans: Vec<(usize, usize)>,
        chunk_frames: Vec<usize>,
        total_mel_frames: usize,
        capture: Option<&'a mut Vec<ChunkTrace>>,
    },
}

pub struct Model {
    pub cfg: ModelConfig,
    pub proc: ProcessorConfig,
    pub tower: Tower,
    pub head: Head,
    /// `silence_embeds` — a learned vector that fills the reserved slots of a
    /// compressed speaker cache. It is **not** added to the encoder output.
    pub silence: Vec<f32>,
    filters: Vec<f32>,
    /// When set, the encoder and the head run on wgpu. The speaker cache stays
    /// on the CPU. Backends are Vulkan, Metal, DX12 and GL.
    accel: Option<RefCell<GpuEngine>>,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let (cfg, proc) = crate::config::load_configs(dir)?;
        Self::from_parts(cfg, proc, &Weights::load(dir)?)
    }

    pub fn from_parts(cfg: ModelConfig, proc: ProcessorConfig, w: &Weights) -> Result<Self> {
        let tower = Tower::load(&cfg.audio_config, w)?;
        let head = Head::load(&cfg.head_config, &cfg, w)?;
        let silence = w
            .tensor("silence_embeds", &[cfg.audio_config.hidden_size])?
            .to_vec();
        let filters = mel_filters(&proc.feature_extractor);
        Ok(Self { cfg, proc, tower, head, silence, filters, accel: None })
    }

    /// Load the CPU model and attach the wgpu encoder/head. The speaker cache
    /// stays on CPU. Device selection is whatever wgpu can drive on this box.
    pub fn load_gpu(dir: &Path) -> Result<Self> {
        let (cfg, proc) = crate::config::load_configs(dir)?;
        let w = Weights::load(dir)?;
        let mut m = Self::from_parts(cfg, proc, &w)?;
        m.accel = Some(RefCell::new(GpuEngine::load(&m.cfg, &w)?));
        Ok(m)
    }

    /// Whole-recording embedder, for callers that want every group at once. The GPU
    /// form chunks internally, so this has no length limit either.
    fn embed_features(&self, mel: &[f32], num_frames: usize) -> Result<Vec<f32>> {
        match &self.accel {
            // The whole-recording embedder is one 1.3 GFLOP GEMM. It is still a net
            // win over the CPU's 55 ms even though it needs a host round-trip: the
            // GPU work in between also ramps the clocks, which is worth more than
            // the transfer costs (the first tower drops from 216 ms to 111 ms).
            Some(g) => g.borrow_mut().embed_mel(mel, num_frames),
            None => Ok(self.tower.embed(mel, num_frames)),
        }
    }

    /// Encoder groups `lo..hi` of the whole recording.
    ///
    /// This is the form every caller wants. The whole-file form has to hold the
    /// entire recording's encoder output in one buffer, which caps the recording
    /// length; a window is at most `chunk_length + chunk_right_context` groups, so
    /// nothing has to. Both paths are the same GEMM over the same operands, so the
    /// overlapping right-context frames are computed twice rather than reused —
    /// 12% more embedder work on a job that is ~2.5% of the run, and the price of
    /// not needing a whole-recording buffer.
    fn embed_features_range(
        &self,
        mel: &[f32],
        num_frames: usize,
        lo: usize,
        hi: usize,
    ) -> Result<Vec<f32>> {
        match &self.accel {
            // The whole-recording embedder is one 1.3 GFLOP GEMM. It is still a net
            // win over the CPU's 55 ms even though it needs a host round-trip: the
            // GPU work in between also ramps the clocks, which is worth more than
            // the transfer costs (the first tower drops from 216 ms to 111 ms).
            Some(g) => g.borrow_mut().embed_mel_range(mel, num_frames, lo, hi),
            None => Ok(self.tower.embed_range(mel, num_frames, lo, hi)),
        }
    }

    pub fn gpu_describe(&self) -> Option<String> {
        self.accel.as_ref().map(|g| g.borrow().describe())
    }

    pub fn num_speakers(&self) -> usize {
        self.head.num_speakers()
    }

    pub fn hidden_size(&self) -> usize {
        self.tower.hidden_size()
    }

    pub fn params(&self, mode: StreamingMode) -> ModeParams {
        ModeParams::resolve(mode, &self.cfg, &self.proc)
    }

    /// Log-mel features of one chunk, plus the frame count the processor returns after
    /// trimming away the frame `torch.stft` over-emits.
    fn mel_of(&self, samples: &[f32], padding: Padding) -> (Vec<f32>, usize) {
        let fe = &self.proc.feature_extractor;
        let valid = frame_count(samples.len(), fe, padding);
        (log_mel(samples, fe, padding, &self.filters), valid)
    }

    /// Whole-file offline pass.
    pub fn run_offline(&self, samples: &[f32]) -> Result<RunOutput> {
        self.run_offline_capturing(samples, None)
    }

    /// [`run_offline`](Self::run_offline) that also records, per chunk, the encoder
    /// window (`cached ++ chunk ++ look-ahead`) and the tower's output on it.
    ///
    /// This is the hook `head_check` uses to tell an encoder regression apart from a
    /// head regression: the recorded window lengths are what the reference's own
    /// hidden-state dump is cut into.
    pub fn run_offline_capturing(
        &self,
        samples: &[f32],
        mut capture: Option<&mut Vec<ChunkTrace>>,
    ) -> Result<RunOutput> {
        let fe = &self.proc.feature_extractor;
        let sub = self.proc.subsampling_factor;
        let t_mel = std::time::Instant::now();
        let (mut mel, valid) = self.mel_of(samples, Padding::Centered);

        // `torch.stft(center=True)` yields `floor(L / hop) + 1` frames: the extra one
        // sits entirely on the centre padding, so the processor's
        // `input_features *= attention_mask` turns it into an exact zero row. It is
        // kept, because it lands in the last encoder group and changes its embedding.
        mel.resize((valid + 1) * fe.feature_size, 0.0);
        if std::env::var("CUDA_PROFILE").is_ok() {
            eprintln!("host mel {:.1} ms", t_mel.elapsed().as_secs_f64() * 1e3);
        }

        // `groups` is the encoder-frame count, and it is computable **without
        // embedding anything** — that is the whole point. The old code called
        // `embed_features` on the entire recording here to find it, which is what
        // put a length limit on offline mode.
        let groups = (valid + 1).div_ceil(sub);
        // `embed_mask = attention_mask[:, ::subsampling]`: frame `g` is valid when its
        // first mel frame `g * sub` is real audio
        let (chunk_len, right_ctx) = (self.cfg.chunk_length, self.cfg.chunk_right_context);
        let mut spans = Vec::new();
        let mut chunk_frames = Vec::new();
        let mut start = 0usize;
        while start < groups {
            let end = (start + chunk_len).min(groups);
            spans.push((start, (end + right_ctx).min(groups)));
            chunk_frames.push(end - start);
            start += chunk_len;
        }
        let cache = SpeakerCache::new(CacheConfig::offline(&self.cfg), self.hidden_size());
        if std::env::var_os("OFFLINE_WHOLEFILE").is_some() {
            // Diagnostic only: the pre-2026-10-01 shape, kept so the per-window
            // embedder can be A/B'd against it on a file the old one could still
            // handle. It is the path that needed `EMBED_GROUPS` and therefore the
            // ~10.9 min cap. Default off, and it is *not* a supported configuration.
            let embeds = self.embed_features(&mel, valid + 1)?;
            let mask: Vec<bool> = (0..groups).map(|g| g * sub < valid).collect();
            return self.chunk_loop(
                Inputs::OfflineWhole {
                    embeds: &embeds,
                    mask: &mask,
                    spans,
                    chunk_frames,
                    total_mel_frames: valid + 1,
                    capture: capture.as_deref_mut(),
                },
                cache,
            );
        }
        self.chunk_loop(
            Inputs::Offline {
                mel: &mel,
                num_mel_frames: valid + 1,
                valid,
                spans,
                chunk_frames,
                total_mel_frames: valid + 1,
                capture: capture.as_deref_mut(),
            },
            cache,
        )
    }

    /// A full streaming session over a whole file, in the given mode.
    pub fn run_streaming(&self, samples: &[f32], mode: StreamingMode) -> Result<RunOutput> {
        let prof = std::env::var("STEP_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        let sub = self.proc.subsampling_factor;
        let hidden = self.hidden_size();
        let mut chunks = Vec::new();
        for c in self.streaming_chunks(samples, mode)? {
            // every streaming chunk is fully valid, so the mask is all-true; the
            // processor trims to the valid frames and leaves nothing masked out
            let embeds = self.embed_features(&c.mel, c.num_mel_frames)?;
            let groups = embeds.len() / hidden;
            debug_assert_eq!(groups * sub, c.num_mel_frames.div_ceil(sub) * sub);
            chunks.push((embeds, groups - c.lookahead, c.num_mel_frames));
        }
        if prof {
            eprintln!(
                "  streaming front end: {} chunks, mel+embed {:.1} ms",
                chunks.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let cache = SpeakerCache::new(CacheConfig::from_model(&self.cfg), self.hidden_size());
        self.chunk_loop(Inputs::Streaming { chunks }, cache)
    }

    /// The chunk schedule a `Nemotron3DiarizationProcessor` session feeds the model.
    ///
    /// The first chunk is centred and holds `first_chunk_samples`; later chunks start
    /// `n_fft / 2` samples before their first mel frame and hold `samples_per_chunk`.
    /// The final chunk is whatever is left — and, per the reference, **centred**.
    pub fn streaming_chunks(&self, samples: &[f32], mode: StreamingMode) -> Result<Vec<Chunk>> {
        let mp = self.params(mode);
        let lookahead = mp.lookahead_encoder_frames;
        let mut chunks = Vec::new();

        let (mel, valid) =
            self.mel_of(&samples[..mp.first_chunk_samples.min(samples.len())], Padding::Centered);
        chunks.push(Chunk { mel, num_mel_frames: valid, lookahead });

        let mut mel_idx = mp.mel_frames_per_step;
        let mut start = mp.audio_chunk_start(mel_idx);
        while start + mp.samples_per_chunk <= samples.len() {
            let (mel, valid) =
                self.mel_of(&samples[start..start + mp.samples_per_chunk], Padding::Uncentered);
            chunks.push(Chunk { mel, num_mel_frames: valid, lookahead });
            mel_idx += mp.mel_frames_per_step;
            start = mp.audio_chunk_start(mel_idx);
        }
        // the last chunk has no next step, so nothing is held back as look-ahead
        let (mel, valid) = self.mel_of(&samples[start.min(samples.len())..], Padding::Centered);
        chunks.push(Chunk { mel, num_mel_frames: valid, lookahead: 0 });
        Ok(chunks)
    }

    /// One step of the reference's chunk loop: `[cache ++ fifo ++ chunk ++ look-ahead]`
    /// through the tower and the head, the cache update, and the scored logit window.
    ///
    /// Returns the number of logit values appended.
    fn step(
        &self,
        cache: &mut SpeakerCache,
        embeds: &[f32],
        mask: &[bool],
        num_chunk_frames: usize,
        cap: usize,
        logits: &mut Vec<f32>,
        mut capture: Option<&mut Vec<ChunkTrace>>,
    ) -> usize {
        let (sub, ns, hidden) = (self.proc.subsampling_factor, self.num_speakers(), self.hidden_size());
        let t_step = std::time::Instant::now();
        let prefix = cache.prefix();
        let cached_len = cache.prefix_frames();
        let mut window = prefix;
        window.extend_from_slice(embeds);
        let mut step_mask = vec![true; cached_len];
        step_mask.extend_from_slice(mask);
        // the reference's bidirectional padding mask is a true prefix followed by
        // false, which a count reproduces exactly
        let valid = step_mask.iter().filter(|&&b| b).count();

        // positions restart at zero on every chunk
        let frames = cached_len + embeds.len() / hidden;
        let want_hidden = capture.is_some();
        let t_call = std::time::Instant::now();
        let (enc_out, step_logits) = if let Some(accel) = &self.accel {
            let mut g = accel.borrow_mut();
            if want_hidden {
                let enc_out = g.forward_embeds(&window, valid).expect("gpu encoder");
                let step_logits = g.head_forward(&enc_out, frames).expect("gpu head");
                (enc_out, step_logits)
            } else {
                let step_logits = g.forward_window(&window, valid).expect("gpu window");
                (Vec::new(), step_logits)
            }
        } else {
            let (enc_out, _) = self.tower.forward_embeds(&window, Some(valid), 0, &[]);
            let step_logits = self.head.forward(&enc_out, frames);
            (enc_out, step_logits)
        };
        let t_gpu = std::time::Instant::now();
        // Window assembly is `t_call - t_step`; the GPU forward is
        // `t_gpu - t_call`. Printing `t_gpu - t_step` as "pre" double-counts the
        // forward and makes host work look like half the streaming budget when it
        // is well under 1%.
        let prof = if std::env::var("STEP_PROFILE").is_ok() {
            Some((
                t_call.duration_since(t_step).as_secs_f64() * 1e3,
                t_gpu.duration_since(t_call).as_secs_f64() * 1e3,
            ))
        } else {
            None
        };
        let mut trace = capture.as_deref_mut().map(|_| ChunkTrace {
            cached_frames: cached_len,
            window_frames: window.len() / hidden,
            valid_frames: valid,
            scored_frames: num_chunk_frames,
            encoder_out: enc_out.clone(),
            step_probs: Vec::new(),
            cache_embeds: Vec::new(),
            cache_probs: Vec::new(),
            fifo_embeds: Vec::new(),
        });
        if trace.is_some() {
            trace.as_mut().unwrap().step_probs = cache.pool_probs(
                &step_logits,
                (cached_len + embeds.len() / hidden) * sub,
                &step_mask,
            );
        }
        let t_cache = std::time::Instant::now();
        cache.update(&window, &step_logits, &self.silence, num_chunk_frames, &step_mask);
        if let Some(t) = trace.as_mut() {
            t.cache_embeds = cache.embeds().to_vec();
            t.cache_probs = cache.probs().to_vec();
            t.fifo_embeds = cache.fifo().to_vec();
        }
        if let (Some(t), Some(out)) = (trace, capture) {
            out.push(t);
        }

        let t_slice = std::time::Instant::now();
        let s = cached_len * sub;
        let e = (cached_len + num_chunk_frames) * sub;
        // the forward keeps at most `num_frames`: for streaming that is *this call's*
        // mel frame count, and for offline the run is truncated once at the end, so
        // its windows are uncapped here
        let keep = (e - s).min(cap) * ns;
        logits.extend_from_slice(&step_logits[s * ns..s * ns + keep]);
        if let Some((asm, call)) = prof {
            eprintln!(
                "  step frames={} cached={}: asm {:.2} ms, call {:.1} ms, \
                 cache.update {:.2} ms, slice {:.2} ms",
                frames,
                cached_len,
                asm,
                call,
                t_slice.duration_since(t_cache).as_secs_f64() * 1e3,
                t_slice.elapsed().as_secs_f64() * 1e3,
            );
        }
        keep
    }

    /// Drive [`step`](Self::step) over every window, in order.
    fn chunk_loop(&self, inputs: Inputs<'_>, mut cache: SpeakerCache) -> Result<RunOutput> {
        let (ns, hidden) = (self.num_speakers(), self.hidden_size());
        let mut logits: Vec<f32> = Vec::new();
        let mut num_chunks = 0;
        let mut total_cap = usize::MAX;

        match inputs {
            Inputs::Offline { mel, num_mel_frames, valid, spans, chunk_frames, total_mel_frames, mut capture } => {
                total_cap = total_mel_frames * ns;
                let sub = self.proc.subsampling_factor;
                for ((lo, hi), &n_chunk) in spans.iter().zip(&chunk_frames) {
                    let (lo, hi) = (*lo, *hi);
                    let embeds = self.embed_features_range(mel, num_mel_frames, lo, hi)?;
                    let mask: Vec<bool> = (lo..hi).map(|g| g * sub < valid).collect();
                    self.step(
                        &mut cache,
                        &embeds,
                        &mask,
                        n_chunk,
                        usize::MAX,
                        &mut logits,
                        capture.as_deref_mut(),
                    );
                    num_chunks += 1;
                }
            }
            Inputs::Streaming { chunks } => {
                for (embeds, n_chunk, n_mel) in &chunks {
                    let mask = vec![true; embeds.len() / hidden];
                    self.step(&mut cache, embeds, &mask, *n_chunk, *n_mel, &mut logits, None);
                    num_chunks += 1;
                }
            }
            Inputs::OfflineWhole { embeds, mask, spans, chunk_frames, total_mel_frames, mut capture } => {
                total_cap = total_mel_frames * ns;
                for ((lo, hi), &n_chunk) in spans.iter().zip(&chunk_frames) {
                    let (lo, hi) = (*lo, *hi);
                    self.step(
                        &mut cache,
                        &embeds[lo * hidden..hi * hidden],
                        &mask[lo..hi],
                        n_chunk,
                        usize::MAX,
                        &mut logits,
                        capture.as_deref_mut(),
                    );
                    num_chunks += 1;
                }
            }
        }
        // offline: `torch.cat(logits, dim=1)[:, :num_frames]`
        logits.truncate(total_cap);
        Ok(RunOutput {
            num_frames: logits.len() / self.num_speakers(),
            logits,
            num_chunks,
            cache_compressed: cache.is_compressed(),
            cache_frames: cache.cache_frames(),
            fifo_frames: cache.fifo_frames(),
        })
    }
}

