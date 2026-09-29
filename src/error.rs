//! Errors.

use std::path::PathBuf;

/// The error type for this crate.
#[derive(Debug, thiserror::Error)]
pub enum DiarizationError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("tensor `{name}` has dtype {dtype}, this port only reads F32")]
    TensorDtype { name: String, dtype: String },

    #[error("tensor `{name}` is missing from the checkpoint")]
    MissingTensor { name: String },

    #[error("tensor `{name}` has shape {got:?}, expected {expected:?}")]
    TensorShape {
        name: String,
        got: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("unsupported audio: {0}")]
    UnsupportedAudio(String),

    #[error("gpu: {0}")]
    Gpu(String),
}

impl DiarizationError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, DiarizationError>;
