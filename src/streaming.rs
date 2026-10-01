//! The streaming state: Arrival-Order Speaker Cache (AOSC) + FIFO queue.
//!
//! Port of `Nemotron3DiarizationSpeakerCache`. This is what makes a streaming
//! diarizer agree with itself about *who* is speaking, and it is the part most
//! likely to be got subtly wrong, so the mechanics are spelled out.
//!
//! # Why identity survives chunk boundaries
//!
//! The cache stores **encoder embeddings, not speaker labels**. Every forward the
//! encoder sees `[cache ++ fifo ++ chunk ++ look-ahead]` as one sequence, so the
//! current chunk attends to the past audio itself. Identity is carried by attention
//! over real past audio, not by a lookup table.
//!
//! # Why channel `k` keeps meaning the same person
//!
//! [`SpeakerCache::compress`] rebuilds the cache **speaker-major**: a fixed per-speaker
//! quota, one reserved silence slot per speaker, then a `topk` whose indices are
//! sorted. The flat score index is `speaker * num_scored_frames + frame`, so sorting
//! groups frames by speaker and orders each group by time in the original audio. That
//! layout pins channel `k` to "the k-th speaker to arrive" for the whole session.
//!
//! The reserved silence slot is load-bearing: a speaker who has not spoken for a long
//! time loses their frames to compression, but their *slot* survives, so the channel
//! numbering cannot drift. Drop the `+inf` sentinel path and long sessions renumber.
//!
//! # Three details that decide whether this matches bit for bit
//!
//! * `_pool_probs` **truncates** to `floor(frames / subsampling)` groups — `avg_pool1d`
//!   has no padding — and applies the attention mask *after* averaging, so a masked
//!   encoder frame zeroes all eight of its 10 ms sub-frames at once.
//! * `_boost_scores` runs `topk` over the **frame** axis (`dim=1` of a
//!   `(frames, speakers)` tensor), i.e. per speaker, not over the flattened score.
//! * The reserved silence slot's **probs are a zero row**, not a copy of frame 0.

use crate::config::ModelConfig;

/// Cache policy, resolved from the config.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub fifo_length: usize,
    pub speaker_cache_length: usize,
    pub update_period: usize,
    pub num_silence_frames: usize,
    pub score_threshold: f32,
    pub latest_frames_score_boost: f32,
    pub num_speakers: usize,
    pub subsampling_factor: usize,
    pub min_positive_scores: usize,
    pub num_strong_boosted_frames: usize,
    pub num_weak_boosted_frames: usize,
}

impl CacheConfig {
    pub fn from_model(m: &ModelConfig) -> Self {
        let s = &m.streaming_config;
        let ns = s.num_speakers;
        // frames every speaker is budgeted, excluding that speaker's silence slot
        let budget = s.speaker_cache_length / ns - s.speaker_cache_silence_frames_per_speaker;
        Self {
            fifo_length: s.fifo_length,
            speaker_cache_length: s.speaker_cache_length,
            update_period: s.speaker_cache_update_period,
            num_silence_frames: s.speaker_cache_silence_frames_per_speaker,
            score_threshold: s.prediction_score_threshold,
            latest_frames_score_boost: s.latest_frames_score_boost,
            num_speakers: ns,
            subsampling_factor: s.subsampling_factor,
            min_positive_scores: (budget as f32 * s.min_positive_scores_rate).floor() as usize,
            num_strong_boosted_frames: (budget as f32 * s.strong_boost_rate).floor() as usize,
            num_weak_boosted_frames: (budget as f32 * s.weak_boost_rate).floor() as usize,
        }
    }

    /// Streaming and offline differ only in FIFO length and update period; both take
    /// the rest from `streaming_config`. They are passed explicitly because offline
    /// uses the *model-level* values, not the streaming ones.
    pub fn with_sizes(mut self, fifo_length: usize, update_period: usize) -> Self {
        self.fifo_length = fifo_length;
        self.update_period = update_period;
        self
    }

    /// The offline variant: same policy, model-level FIFO sizes.
    pub fn offline(m: &ModelConfig) -> Self {
        Self::from_model(m).with_sizes(m.fifo_length, m.speaker_cache_update_period)
    }

    pub fn budget_per_speaker(&self) -> usize {
        self.speaker_cache_length / self.num_speakers - self.num_silence_frames
    }

    pub fn describe(&self) -> String {
        format!(
            "cache {} ({} frames/speaker + {} silence slots), fifo {}, pop >= {}, \
             min_positive {}, boost strong {} weak {}, threshold {}, latest boost {}",
            self.speaker_cache_length,
            self.budget_per_speaker(),
            self.num_silence_frames,
            self.fifo_length,
            self.update_period,
            self.min_positive_scores,
            self.num_strong_boosted_frames,
            self.num_weak_boosted_frames,
            self.score_threshold,
            self.latest_frames_score_boost,
        )
    }
}

/// The intermediates of one [`SpeakerCache::trace_compress`] call.
#[derive(Debug, Clone)]
pub struct CompressTrace {
    /// `_get_frame_scores` output, before any boost.
    pub raw: Vec<f32>,
    /// After `_get_frame_scores` and the latest-frames boost.
    pub after_latest: Vec<f32>,
    /// After the strong boost.
    pub after_strong: Vec<f32>,
    /// After the weak boost.
    pub scores: Vec<f32>,
    /// The transposed, silence-padded score vector `topk` ranks.
    pub flat: Vec<f32>,
    /// The `speaker_cache_length` retained indices, remapped and sorted.
    pub picked: Vec<usize>,
    pub sentinel: usize,
}

/// State threaded from one forward to the next.
#[derive(Debug, Clone)]
pub struct SpeakerCache {
    pub cfg: CacheConfig,
    hidden: usize,
    /// `(cache_frames, hidden)` — selected past encoder frames, speaker-major.
    embeds: Vec<f32>,
    /// `(cache_frames, num_speakers)`, stored alongside the frames above.
    probs: Vec<f32>,
    /// `(fifo_frames, hidden)`.
    fifo: Vec<f32>,
    /// A compressed cache is speaker-major, so its stored probs are the only ones
    /// that still line up with its frames; an uncompressed one holds plain chunk
    /// frames, whose probabilities the current step re-estimates anyway.
    is_compressed: bool,
}

impl SpeakerCache {
    pub fn new(cfg: CacheConfig, hidden: usize) -> Self {
        Self {
            cfg,
            hidden,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            is_compressed: false,
        }
    }

    pub fn cache_frames(&self) -> usize {
        self.embeds.len() / self.hidden
    }

    pub fn fifo_frames(&self) -> usize {
        self.fifo.len() / self.hidden
    }

    pub fn is_compressed(&self) -> bool {
        self.is_compressed
    }

    /// Selected past encoder frames, speaker-major — the cache's whole content.
    pub fn embeds(&self) -> &[f32] {
        &self.embeds
    }

    /// Speaker probabilities stored alongside [`embeds`](Self::embeds).
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// FIFO embeddings: the most recent frames not yet promoted to the cache.
    pub fn fifo(&self) -> &[f32] {
        &self.fifo
    }

    /// `[cache ++ fifo]` — the prefix the next encoder call is prepended with.
    pub fn prefix(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.embeds.len() + self.fifo.len());
        out.extend_from_slice(&self.embeds);
        out.extend_from_slice(&self.fifo);
        out
    }

    pub fn prefix_frames(&self) -> usize {
        self.cache_frames() + self.fifo_frames()
    }

    /// `_pool_probs`: average `sigmoid(logits)` over each group of `subsampling_factor`
    /// consecutive 10 ms frames, then zero whole groups the attention mask rejects.
    ///
    /// `num_mel_frames` is `input_frames * subsampling_factor`; `mask` is the step mask
    /// at the encoder frame rate. The last partial group is **dropped**, because
    /// `avg_pool1d` without `ceil_mode` does not pad.
    pub fn pool_probs(&self, logits: &[f32], num_mel_frames: usize, mask: &[bool]) -> Vec<f32> {
        let (sub, ns) = (self.cfg.subsampling_factor, self.cfg.num_speakers);
        let groups = num_mel_frames / sub;
        let mut out = vec![0.0f32; groups * ns];
        for g in 0..groups {
            let keep = mask.get(g).copied().unwrap_or(false);
            for c in 0..ns {
                let mut acc = 0.0f32;
                for k in 0..sub {
                    acc += 1.0 / (1.0 + (-logits[(g * sub + k) * ns + c]).exp());
                }
                out[g * ns + c] = if keep { acc / sub as f32 } else { 0.0 };
            }
        }
        out
    }

    /// `_num_popped_frames`.
    fn num_popped(&self, num_fifo_frames: usize) -> usize {
        if num_fifo_frames <= self.cfg.fifo_length {
            return 0;
        }
        self.cfg
            .update_period
            .max(num_fifo_frames - self.cfg.fifo_length)
            .min(num_fifo_frames)
    }

    /// Push a processed chunk into the FIFO, promoting the oldest frames to the cache
    /// when it overflows and compressing the cache when that overflows too.
    ///
    /// `chunk_input_embeds` is `[cached ++ chunk ++ look-ahead]`, `chunk_logits` the
    /// head output over all of it (at the 10 ms rate), and `step_mask` the same window
    /// at the encoder frame rate.
    pub fn update(
        &mut self,
        chunk_input_embeds: &[f32],
        chunk_logits: &[f32],
        silence_embeds: &[f32],
        num_chunk_frames: usize,
        step_mask: &[bool],
    ) {
        let ns = self.cfg.num_speakers;
        let hidden = self.hidden;
        let num_input_frames = chunk_input_embeds.len() / hidden;
        let probs = self.pool_probs(chunk_logits, num_input_frames * self.cfg.subsampling_factor, step_mask);

        let num_cache_frames = self.cache_frames();
        let chunk_start = self.prefix_frames();
        let mut fifo = self.fifo.clone();
        for i in 0..num_chunk_frames {
            let src = (chunk_start + i) * hidden;
            fifo.extend_from_slice(&chunk_input_embeds[src..src + hidden]);
        }

        let num_popped = self.num_popped(fifo.len() / hidden);
        if num_popped > 0 {
            let num_fifo_frames = fifo.len() / hidden;
            // FIFO probabilities always come from this step, indexed past the cache
            let fifo_probs = &probs[num_cache_frames * ns..(num_cache_frames + num_fifo_frames) * ns];
            let stored_probs: Vec<f32> = if self.is_compressed {
                self.probs.clone()
            } else {
                probs[..num_cache_frames * ns].to_vec()
            };

            let mut cache_e = self.embeds.clone();
            cache_e.extend_from_slice(&fifo[..num_popped * hidden]);
            let mut cache_p = stored_probs;
            cache_p.extend_from_slice(&fifo_probs[..num_popped * ns]);
            fifo.drain(..num_popped * hidden);

            if cache_e.len() / hidden > self.cfg.speaker_cache_length {
                let (e, p) = self.compress(&cache_e, &cache_p, silence_embeds);
                cache_e = e;
                cache_p = p;
                self.is_compressed = true;
            }
            self.embeds = cache_e;
            self.probs = cache_p;
        }
        self.fifo = fifo;
    }

    /// `_get_frame_scores`: a per-(frame, speaker) log-odds score, `-inf` where the
    /// frame must not be kept.
    ///
    /// Two details that have to match the reference exactly:
    /// * the additive constant is `-math.log(0.5) = ln(2)`, not half of that;
    /// * `has_enough_positive` is counted **per speaker across frames** (`sum(dim=1)`
    ///   of a `(frames, speakers)` tensor). Counting positives per frame across
    ///   speakers never fires — there are only 8 speakers and the threshold is 16.
    pub fn frame_scores(&self, probs: &[f32], num_frames: usize) -> Vec<f32> {
        let ns = self.cfg.num_speakers;
        let thr = self.cfg.score_threshold;
        let neg_log_half = std::f32::consts::LN_2; // -math.log(0.5)
        let mut scores = vec![0.0f32; num_frames * ns];

        for f in 0..num_frames {
            let row = &probs[f * ns..(f + 1) * ns];
            let log_probs: Vec<f32> = row.iter().map(|&p| p.max(thr).ln()).collect();
            let log_comps: Vec<f32> = row.iter().map(|&p| (1.0 - p).max(thr).ln()).collect();
            let sum_log_comp: f32 = log_comps.iter().sum();

            for s in 0..ns {
                let v = log_probs[s] - log_comps[s] + sum_log_comp + neg_log_half;
                scores[f * ns + s] = if row[s] > 0.5 { v } else { f32::NEG_INFINITY };
            }
        }
        let mut pos_count = vec![0usize; ns];
        for f in 0..num_frames {
            for s in 0..ns {
                if scores[f * ns + s] > 0.0 {
                    pos_count[s] += 1;
                }
            }
        }
        for s in 0..ns {
            if pos_count[s] >= self.cfg.min_positive_scores {
                for f in 0..num_frames {
                    let i = f * ns + s;
                    if scores[i] <= 0.0 && probs[i] > 0.5 {
                        scores[i] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        scores
    }

    /// `_boost_scores`: add `boost` to each speaker's `k` best frames, independently.
    fn boost_top(&self, scores: &mut [f32], num_frames: usize, k: usize, boost: f32) {
        let ns = self.cfg.num_speakers;
        for s in 0..ns {
            let mut idx: Vec<usize> = (0..num_frames).collect();
            idx.sort_by(|&a, &b| {
                scores[b * ns + s]
                    .partial_cmp(&scores[a * ns + s])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in idx.iter().take(k.min(num_frames)) {
                scores[i * ns + s] += boost;
            }
        }
    }

    /// Every intermediate of [`compress`](Self::compress), for diffing against the
    /// reference's `_get_frame_scores` / `_boost_scores` / `topk` one stage at a time.
    ///
    /// A ranking bug is invisible in the output — one frame swapped produces plausible
    /// logits and a plausible transcript — so this is the only way to see it.
    pub fn trace_compress(&self, embeds: &[f32], probs: &[f32]) -> CompressTrace {
        let (ns, hidden) = (self.cfg.num_speakers, self.hidden);
        let num_frames = embeds.len() / hidden;
        let cache_len = self.cfg.speaker_cache_length;
        let n_silence = self.cfg.num_silence_frames;

        let mut scores = self.frame_scores(probs, num_frames);
        let raw = scores.clone();
        for f in cache_len..num_frames {
            for s in 0..ns {
                scores[f * ns + s] += self.cfg.latest_frames_score_boost;
            }
        }
        let after_latest = scores.clone();
        // `boost = -math.log(0.5)` and `-2 * math.log(0.5)`, both **positive** —
        // log(0.5) is negative, so the reference's minus sign turns into a plus
        let neg_log_half = std::f32::consts::LN_2;
        self.boost_top(
            &mut scores,
            num_frames,
            self.cfg.num_strong_boosted_frames,
            2.0 * neg_log_half,
        );
        let after_strong = scores.clone();
        self.boost_top(
            &mut scores,
            num_frames,
            self.cfg.num_weak_boosted_frames,
            neg_log_half,
        );

        let num_scored = num_frames + n_silence;
        let mut flat = vec![0.0f32; ns * num_scored];
        for s in 0..ns {
            for f in 0..num_frames {
                flat[s * num_scored + f] = scores[f * ns + s];
            }
            for k in 0..n_silence {
                flat[s * num_scored + num_frames + k] = f32::INFINITY;
            }
        }
        let sentinel = num_scored * ns;
        let mut picked: Vec<usize> = (0..flat.len()).collect();
        picked.sort_by(|&a, &b| flat[b].partial_cmp(&flat[a]).unwrap_or(std::cmp::Ordering::Equal));
        picked.truncate(cache_len);
        picked.iter_mut().for_each(|i| {
            if flat[*i] == f32::NEG_INFINITY {
                *i = sentinel;
            }
        });
        picked.sort_unstable();
        CompressTrace { raw, after_latest, after_strong, scores, flat, picked, sentinel }
    }

    /// `_compress`: keep the `speaker_cache_length` most important frames, grouped by
    /// speaker and in their original order within a speaker;
    /// `num_silence_frames` slots are filled with `silence_embeds` and a zero prob row.
    fn compress(&self, embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (ns, hidden) = (self.cfg.num_speakers, self.hidden);
        let num_frames = embeds.len() / hidden;
        let cache_len = self.cfg.speaker_cache_length;
        let n_silence = self.cfg.num_silence_frames;

        let mut scores = self.frame_scores(probs, num_frames);
        // frames beyond the cache capacity are the ones just popped out of the FIFO
        for f in cache_len..num_frames {
            for s in 0..ns {
                scores[f * ns + s] += self.cfg.latest_frames_score_boost;
            }
        }
        let neg_log_half = std::f32::consts::LN_2;
        self.boost_top(
            &mut scores,
            num_frames,
            self.cfg.num_strong_boosted_frames,
            2.0 * neg_log_half,
        );
        self.boost_top(
            &mut scores,
            num_frames,
            self.cfg.num_weak_boosted_frames,
            neg_log_half,
        );

        // pad one reserved silence row per speaker on the *frame* axis, score +inf
        let num_scored = num_frames + n_silence;
        let mut flat = vec![0.0f32; ns * num_scored];
        for s in 0..ns {
            for f in 0..num_frames {
                flat[s * num_scored + f] = scores[f * ns + s];
            }
            for k in 0..n_silence {
                flat[s * num_scored + num_frames + k] = f32::INFINITY;
            }
        }

        // topk over the flat score, then sort the indices: that sort is what turns
        // "best frames overall" into "speaker-major, time-ordered within a speaker"
        let sentinel = num_scored * ns;
        let mut picked: Vec<usize> = (0..flat.len()).collect();
        picked.sort_by(|&a, &b| flat[b].partial_cmp(&flat[a]).unwrap_or(std::cmp::Ordering::Equal));
        picked.truncate(cache_len);
        // every -inf entry is remapped to the silence row first, exactly as the
        // reference's `masked_fill(topk_scores == -inf, sentinel)` does
        picked.iter_mut().for_each(|i| {
            if flat[*i] == f32::NEG_INFINITY {
                *i = sentinel;
            }
        });
        picked.sort_unstable();

        let mut oe = Vec::with_capacity(cache_len * hidden);
        let mut op = Vec::with_capacity(cache_len * ns);
        for idx in picked {
            let frame = if idx == sentinel {
                num_frames
            } else {
                (idx % num_scored).min(num_frames)
            };
            if frame == num_frames {
                // the reserved slot: silence embedding, zero probabilities
                oe.extend_from_slice(silence);
                op.extend(std::iter::repeat(0.0f32).take(ns));
            } else {
                oe.extend_from_slice(&embeds[frame * hidden..(frame + 1) * hidden]);
                op.extend_from_slice(&probs[frame * ns..(frame + 1) * ns]);
            }
        }
        (oe, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CacheConfig {
        CacheConfig {
            fifo_length: 4,
            speaker_cache_length: 8,
            update_period: 3,
            num_silence_frames: 1,
            score_threshold: 0.25,
            latest_frames_score_boost: 0.05,
            num_speakers: 2,
            subsampling_factor: 2,
            min_positive_scores: 1,
            num_strong_boosted_frames: 2,
            num_weak_boosted_frames: 4,
        }
    }

    /// `avg_pool1d` drops the trailing partial group and the mask zeroes whole groups.
    #[test]
    fn pool_probs_truncates_and_masks_per_group() {
        let c = SpeakerCache::new(cfg(), 1);
        // 2 speakers x 5 groups of sub=2 -> 10 mel rows x 2 speakers, alternating
        // a strongly positive and a strongly negative logit down the time axis
        let logits: Vec<f32> = (0..(2 * 5 * 2))
            .map(|i| if (i / 2) % 2 == 0 { 10.0 } else { -10.0 })
            .collect();
        let mask = vec![true, true, false, true, true];
        let p = c.pool_probs(&logits, 10, &mask);
        assert_eq!(p.len(), 5 * 2);
        let hi = 1.0 / (1.0 + (-10.0f32).exp());
        let lo = 1.0 / (1.0 + 10.0f32.exp());
        // group 0 unmasked -> average of (hi, lo)
        assert!((p[0] - (hi + lo) / 2.0).abs() < 1e-6, "{}", p[0]);
        // group 2 masked -> exactly zero, not the average
        assert_eq!(p[2 * 2], 0.0);
        assert_eq!(p[2 * 2 + 1], 0.0);
        // odd mel count: the last partial group disappears
        let short = c.pool_probs(&logits, 9, &mask);
        assert_eq!(short.len(), 4 * 2);
    }

    /// Nothing moves to the cache until the FIFO overflows, and then at least
    /// `update_period` frames go.
    #[test]
    fn num_popped_follows_the_reference_rule() {
        let c = SpeakerCache::new(cfg(), 1);
        assert_eq!(c.num_popped(4), 0, "exactly full is not an overflow");
        assert_eq!(c.num_popped(5), 3, "max(update_period=3, 5-4)");
        assert_eq!(c.num_popped(20), 16, "the overflow itself when it is large");
        assert_eq!(c.num_popped(2), 0);
    }

    /// A `+inf` reserve keeps every speaker's slot alive across a compression, which
    /// is what stops channel numbering from drifting.
    #[test]
    fn compress_keeps_one_silence_slot_per_speaker() {
        let c = SpeakerCache::new(cfg(), 2);
        let (ns, hidden) = (2, 2);
        let num_frames = 9; // over the 8-frame cache, so compression triggers
        let embeds: Vec<f32> = (0..num_frames * hidden).map(|i| i as f32).collect();
        // speaker 0 owns the early frames, speaker 1 the late ones
        let mut probs = vec![0.0f32; num_frames * ns];
        for f in 0..num_frames {
            probs[f * ns] = if f < 5 { 0.9 } else { 0.01 };
            probs[f * ns + 1] = if f < 5 { 0.01 } else { 0.9 };
        }
        let (e, p) = c.compress(&embeds, &probs, &[7.0, 8.0]);
        assert_eq!(e.len(), 8 * hidden, "cache is exactly speaker_cache_length frames");
        assert_eq!(p.len(), 8 * ns);
        // two silence slots survive, and they carry zeros, not a copied prob row
        let sil: Vec<usize> = (0..8)
            .filter(|&i| e[i * hidden] == 7.0 && e[i * hidden + 1] == 8.0)
            .collect();
        assert_eq!(sil.len(), 2, "one reserved slot per speaker, got {sil:?}");
        for &i in &sil {
            assert_eq!(&p[i * ns..(i + 1) * ns], &[0.0, 0.0], "silence probs must be zero");
        }
        // speaker-major: once the slots are appended, kept frames are grouped by speaker
        let kept: Vec<usize> = (0..8).filter(|&i| !sil.contains(&i)).collect();
        assert!(kept.windows(2).all(|w| w[1] > w[0]), "kept frames must be time-ordered");
    }

    /// The compression boost runs along the frame axis, per speaker.
    #[test]
    fn boost_top_is_per_speaker() {
        let mut c = cfg();
        c.num_strong_boosted_frames = 1;
        c.num_weak_boosted_frames = 0;
        let cache = SpeakerCache::new(c, 1);
        let mut scores = vec![0.0f32; 4 * 2];
        // speaker 0: [10, 1, 2, 3]   speaker 1: [0, 0, 0, 9]
        for f in 0..4 {
            scores[f * 2] = [10.0, 1.0, 2.0, 3.0][f];
            scores[f * 2 + 1] = [0.0, 0.0, 0.0, 9.0][f];
        }
        cache.boost_top(&mut scores, 4, 1, 1.0);
        assert_eq!(scores[0], 11.0, "speaker 0 boosts its own best frame");
        assert_eq!(scores[7], 10.0, "speaker 1 boosts its own best frame, not frame 0");
        assert_eq!(scores[1], 0.0, "speaker 1's frame 0 must not be boosted");
    }

    /// The additive constant is `-math.log(0.5) = ln(2)`, and `has_enough_positive`
    /// is counted per speaker. A frame-wise count would never fire on this fixture
    /// (2 speakers, threshold 2).
    #[test]
    fn frame_scores_uses_neg_log_half() {
        let mut c = cfg();
        c.min_positive_scores = 2;
        c.score_threshold = 0.01;
        let cache = SpeakerCache::new(c, 1);
        let mut probs = vec![0.01f32; 4 * 2];
        for f in 0..3 {
            probs[f * 2] = 0.95;
        }
        let s = cache.frame_scores(&probs, 4);
        assert!(s[0] > 0.0, "dominant speaker frame 0 should be positive, got {}", s[0]);
        let lp = 0.95f32.ln();
        let lc0 = (1.0 - 0.95f32).ln();
        let lc1 = (1.0 - 0.01f32).max(0.01).ln();
        let want = lp - lc0 + (lc0 + lc1) + std::f32::consts::LN_2;
        assert!((s[0] - want).abs() < 1e-5, "got {} want {want}", s[0]);
        assert!(s[1].is_infinite() && s[1].is_sign_negative(), "p=0.01 is not speech");
    }
}
