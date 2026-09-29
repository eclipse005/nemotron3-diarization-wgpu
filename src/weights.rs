//! `model.safetensors` loading.
//!
//! The checkpoint is a flat map of 417 `F32` tensors under three top-level prefixes:
//! `model` (audio tower + head), `classifier` and `silence_embeds`. Tensors are read
//! eagerly into `Vec<f32>` in row-major order, which is what every consumer in this
//! crate wants — the shapes are small and fixed, and a lazily-mapped view would only
//! complicate the kernels.
//!
//! Transposes are deliberately **not** done here. Weights keep the layout the
//! checkpoint has, and the GEMM path handles the orientation, so a diff against the
//! reference never depends on this file having reshuffled anything.

use std::collections::BTreeMap;
use std::path::Path;

use safetensors::SafeTensors;

use crate::error::{DiarizationError, Result};

/// All tensors from a checkpoint, keyed by their safetensors name.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    /// Read `model.safetensors` from a checkpoint directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path).map_err(|e| DiarizationError::io(&path, e))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes)?;
        let mut tensors = BTreeMap::new();
        for name in st.names() {
            let t = st.tensor(name)?;
            let shape = t.shape().to_vec();
            if t.dtype() != safetensors::Dtype::F32 {
                return Err(DiarizationError::TensorDtype {
                    name: name.to_string(),
                    dtype: format!("{:?}", t.dtype()),
                });
            }
            let raw = t.data();
            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(name.to_string(), (shape, data));
        }
        Ok(Self { tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|(_, d)| d.as_slice())
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|(s, _)| s.as_slice())
    }

    /// Fetch a tensor and assert its shape — the most common porting mistake is a
    /// silently transposed weight, so this fails loudly instead.
    pub fn tensor(&self, name: &str, expected: &[usize]) -> Result<&[f32]> {
        let (shape, data) = self
            .tensors
            .get(name)
            .ok_or_else(|| DiarizationError::MissingTensor {
                name: name.to_string(),
            })?;
        if shape.as_slice() != expected {
            return Err(DiarizationError::TensorShape {
                name: name.to_string(),
                got: shape.clone(),
                expected: expected.to_vec(),
            });
        }
        Ok(data)
    }

    /// Total parameter count, for reporting.
    pub fn num_parameters(&self) -> usize {
        self.tensors
            .values()
            .map(|(shape, _)| shape.iter().product::<usize>())
            .sum()
    }
}
