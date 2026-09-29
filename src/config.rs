//! Model and processor configuration, parsed from the checkpoint's JSON files.
//!
//! The field names mirror `config.json` and `processor_config.json` exactly — they
//! come straight from the reference implementation and are deliberately not renamed,
//! so a value can be traced back to its source without a lookup table.
//!
//! The only things *not* in the JSON are the three streaming modes, which live in
//! `processor_config.json` as `[chunk_length, right_context]` pairs; those are
//! resolved into concrete sample counts by [`StreamingMode`].

use serde::Deserialize;

use crate::error::{DiarizationError, Result};

/// A streaming configuration: how many 80 ms encoder frames are scored per step,
/// and how many are held back as look-ahead.
///
/// Latency is `(chunk_len + right_ctx) * 80 ms`; the audio advances
/// `chunk_len * subsampling_factor * 10 ms` per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    LowLatency,
    VeryLowLatency,
    UltraLowLatency,
    /// One stateless forward over the whole file, chunked at 340 encoder frames.
    Offline,
}

impl StreamingMode {
    /// `[chunk_length, right_context]` in encoder frames, or `None` for offline.
    fn streaming_pair(&self, p: &ProcessorConfig) -> Option<(usize, usize)> {
        let raw = match self {
            Self::LowLatency => &p.streaming_modes.low_latency,
            Self::VeryLowLatency => &p.streaming_modes.very_low_latency,
            Self::UltraLowLatency => &p.streaming_modes.ultra_low_latency,
            Self::Offline => return None,
        };
        Some((raw[0], raw[1]))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_mel_bins: usize,
    pub subsampling_factor: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Deserialize)]
pub struct HeadConfig {
    pub audio_hidden_size: usize,
    pub hidden_size: usize,
    pub num_speakers: usize,
    pub subsampling_factor: usize,
}

#[derive(Debug, Deserialize)]
pub struct StreamingConfig {
    pub fifo_length: usize,
    pub speaker_cache_length: usize,
    pub speaker_cache_silence_frames_per_speaker: usize,
    pub speaker_cache_update_period: usize,
    pub prediction_score_threshold: f32,
    pub min_positive_scores_rate: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub latest_frames_score_boost: f32,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub audio_config: AudioConfig,
    pub head_config: HeadConfig,
    pub streaming_config: StreamingConfig,
    /// Offline chunk size in encoder frames.
    pub chunk_length: usize,
    /// Offline right-context in encoder frames.
    pub chunk_right_context: usize,
    /// Offline FIFO length, distinct from `streaming_config.fifo_length`.
    pub fifo_length: usize,
    /// Offline cache update period, distinct from the streaming one.
    pub speaker_cache_update_period: usize,
}

#[derive(Debug, Deserialize)]
pub struct FeatureExtractorConfig {
    pub feature_size: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub sampling_rate: u32,
    pub preemphasis: f32,
}

#[derive(Debug, Deserialize)]
pub struct StreamingModes {
    pub low_latency: [usize; 2],
    pub very_low_latency: [usize; 2],
    pub ultra_low_latency: [usize; 2],
}

#[derive(Debug, Deserialize)]
pub struct ProcessorConfig {
    pub feature_extractor: FeatureExtractorConfig,
    pub streaming_modes: StreamingModes,
    pub subsampling_factor: usize,
}

/// Resolved, ready-to-use parameters for one inference mode.
#[derive(Debug, Clone)]
pub struct ModeParams {
    pub mode: StreamingMode,
    /// `feature_extractor.hop_length`, carried here so the chunk maths is self-contained.
    pub hop_length: usize,
    /// `feature_extractor.n_fft`.
    pub n_fft: usize,
    /// Mel frames the processor must be fed per chunk.
    pub mel_frames_per_chunk: usize,
    /// Mel frames actually scored per step.
    pub mel_frames_per_step: usize,
    /// Look-ahead, in encoder frames.
    pub lookahead_encoder_frames: usize,
    /// Samples in the first chunk of a session (centred windows).
    pub first_chunk_samples: usize,
    /// Samples in every later chunk (uncentred windows).
    pub samples_per_chunk: usize,
    /// Total input-buffer latency, milliseconds.
    pub latency_ms: usize,
}

impl ModeParams {
    /// Build the chunk geometry for `mode` exactly as
    /// `Nemotron3DiarizationProcessor` does.
    pub fn resolve(mode: StreamingMode, model: &ModelConfig, proc: &ProcessorConfig) -> Self {
        let fe = &proc.feature_extractor;
        let sub = proc.subsampling_factor;
        let (hop, n_fft, win) = (fe.hop_length, fe.n_fft, fe.win_length);

        let Some((chunk_len, right_ctx)) = mode.streaming_pair(proc) else {
            return Self {
                mode,
                hop_length: hop,
                n_fft,
                mel_frames_per_chunk: 0,
                mel_frames_per_step: 0,
                lookahead_encoder_frames: 0,
                first_chunk_samples: 0,
                samples_per_chunk: 0,
                latency_ms: (model.chunk_length + model.chunk_right_context) * 80,
            };
        };

        let mel_per_chunk = (chunk_len + right_ctx) * sub;
        let mel_per_step = chunk_len * sub;
        Self {
            mode,
            hop_length: hop,
            n_fft,
            mel_frames_per_chunk: mel_per_chunk,
            mel_frames_per_step: mel_per_step,
            lookahead_encoder_frames: right_ctx,
            // centred: `floor(L / hop)` valid frames, so L = (frames - 1) * hop + win
            first_chunk_samples: (mel_per_chunk - 1) * hop + win,
            // uncentred: `floor((L - n_fft) / hop) + 1 == mel_per_chunk`
            samples_per_chunk: mel_per_chunk * hop + n_fft,
            latency_ms: (chunk_len + right_ctx) * 80,
        }
    }

    /// First audio sample of the chunk that *starts* at mel frame `mel_frame_idx`.
    ///
    /// Uncentred windows begin `n_fft / 2` samples before the frame they belong to.
    pub fn audio_chunk_start(&self, mel_frame_idx: usize) -> usize {
        mel_frame_idx * self.step_samples() - self.n_fft / 2
    }

    /// Samples of audio consumed per step: `mel_frames_per_step * hop_length`.
    pub fn step_samples(&self) -> usize {
        self.mel_frames_per_step * self.hop_length
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|e| DiarizationError::io(path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| DiarizationError::json(path, e))
}

/// Load `config.json` and `processor_config.json` from a checkpoint directory.
pub fn load_configs(dir: &std::path::Path) -> Result<(ModelConfig, ProcessorConfig)> {
    let model: ModelConfig = read_json(&dir.join("config.json"))?;
    let proc: ProcessorConfig = read_json(&dir.join("processor_config.json"))?;
    Ok((model, proc))
}
