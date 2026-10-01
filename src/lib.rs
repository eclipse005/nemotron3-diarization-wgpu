//! NVIDIA Nemotron 3 Diarization on wgpu — a bit-accurate Rust port.
//!
//! This crate reproduces the official HuggingFace `transformers` implementation
//! (`nvidia/Nemotron-3-Diarization`) so it can run on any wgpu backend without
//! PyTorch or CUDA. The reference it is checked against lives one directory up:
//! the Python baseline in `../baseline/`.
//!
//! The port is **validated stage by stage against that baseline**, not written
//! top-down and hoped for. Each stage has a target file it must reproduce:
//!
//! | stage | target | checker |
//! |---|---|---|
//! | front-end (preemphasis, STFT, slaney mel, log) | `baseline/frontend/*.npz` | `front_end_check` |
//! | encoder blocks | `baseline/hidden/*__layer*.npy` | `encoder_check` |
//! | head + streaming state | `baseline/frames/*.npy` | `frame_check` |
//! | diarization output | `baseline/segments/*.json` | `diarize --check` |
//!
//! # Layout
//!
//! * [`mel`] — the audio front-end. Ported first because every later stage consumes
//!   its output, and because it is where three easy-to-miss conventions live.
//! * [`config`] — `config.json` / `processor_config.json` and the streaming chunk
//!   geometry derived from them.
//! * [`weights`] — `model.safetensors` loading.
//! * [`error`] — the crate's error type.
//!
//! The remaining modules are engine internals. They are public because the probe
//! binaries in `src/bin/` are separate crates that drive them directly, matching
//! the layout used by the sibling `qwen3-asr-wgpu` project.

mod config;
mod encoder;
mod error;
mod gpu;
mod gpu_engine;
mod head;
mod mel;
mod model;
mod npy;
mod segments;
mod streaming;
mod tensor;
mod weights;

pub use config::{
    load_configs, AudioConfig, FeatureExtractorConfig, HeadConfig, ModelConfig, ModeParams,
    ProcessorConfig, StreamingConfig, StreamingMode,
};
pub use error::{DiarizationError, Result};
pub use mel::{
    frame_count, log_mel, mel_filters, slaney_mel_filterbank, Padding, LOG_ZERO_GUARD_VALUE,
};
pub use encoder::Tower;
pub use gpu_engine::GpuEngine;
pub use gpu_engine::{
    S_ADD, S_ALL, S_ATTN, S_FC1, S_FC2, S_GELU, S_LN, S_O, S_PV, S_QK, S_QKV,
    S_ROPE, S_SM,
};
pub use head::{Head, HeadTrace};
pub use model::{Chunk, ChunkTrace, Model, RunOutput};
pub use npy::{read_f32, read_f32_bytes, Array};
pub use segments::{activity_stats, extract_speaker_dict, frame_duration, Segment};
pub use streaming::{CacheConfig, CompressTrace, SpeakerCache};
pub use weights::Weights;
