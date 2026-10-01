//! Minimal `.npy` reader for the Python baseline dumps.
//!
//! The baseline is written by `numpy.save`, so reading it back is the only way to
//! diff this port against the reference. This supports what the baseline actually
//! contains — little-endian `<f4` and `<i8`, 1-D or 2-D, C or Fortran order — and
//! errors loudly on anything else rather than guessing a layout.

use std::path::Path;

use crate::error::{DiarizationError, Result};

/// A decoded array: its shape plus row-major data.
#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}

/// Parse the header of a `.npy` buffer: `(dtype, data offset, fortran_order)`.
fn header(bytes: &[u8]) -> Result<(String, usize, bool, Vec<usize>)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(DiarizationError::Npy("not a .npy file".into()));
    }
    let major = bytes[6];
    let (header_len, off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        )
    };
    let text = std::str::from_utf8(&bytes[off..off + header_len])
        .map_err(|e| DiarizationError::Npy(e.to_string()))?;

    let field = |key: &str| -> Option<String> {
        let rest = text.split(&format!("'{key}':")).nth(1)?.trim_start();
        let rest = rest.strip_prefix('\'').unwrap_or(rest).trim();
        // stop at the closing quote or the end of the dict, then drop the separating
        // comma — but *not* at commas inside the shape tuple
        Some(
            rest.split(|c| c == '\'' || c == '}')
                .next()
                .unwrap_or("")
                .trim()
                .trim_end_matches(',')
                .trim()
                .to_string(),
        )
    };
    let descr = field("descr")
        .ok_or_else(|| DiarizationError::Npy("no descr in .npy header".into()))?;
    let fortran = field("fortran_order").as_deref() == Some("True");
    let shape_txt = field("shape")
        .ok_or_else(|| DiarizationError::Npy("no shape in .npy header".into()))?;
    let shape: Vec<usize> = shape_txt
        .trim_matches(|c| c == '(' || c == ')' || c == ',')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("integer dimension"))
        .collect();
    Ok((descr, off + header_len, fortran, shape))
}

/// Read a little-endian `<f4` `.npy` as f64 data, normalising Fortran order to
/// row-major.
pub fn read_f32(path: &Path) -> Result<Array> {
    let bytes = std::fs::read(path).map_err(|e| DiarizationError::io(path, e))?;
    read_f32_bytes(&bytes).map_err(|e| match e {
        DiarizationError::Npy(m) => DiarizationError::Npy(format!("{}: {m}", path.display())),
        other => other,
    })
}

pub fn read_f32_bytes(bytes: &[u8]) -> Result<Array> {
    let (descr, off, fortran, shape) = header(bytes)?;
    let vals: Vec<f64> = match descr.as_str() {
        "<f4" => bytes[off..]
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        // `numpy.save` keeps integer arrays integer, and the reference's `topk`
        // dumps are exactly that
        "<i8" => bytes[off..]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        other => return Err(DiarizationError::Npy(format!("expected <f4 or <i8, got {other}"))),
    };
    let data = if fortran { to_row_major(vals, &shape) } else { vals };
    Ok(Array { shape, data })
}

fn to_row_major(vals: Vec<f64>, shape: &[usize]) -> Vec<f64> {
    if shape.len() != 2 {
        return vals;
    }
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = vec![0.0f64; vals.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = vals[c * rows + r];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(descr: &str, shape: &str, fortran: &str, body: &[u8]) -> Vec<u8> {
        let mut v = b"\x93NUMPY\x01\x00".to_vec();
        let h = format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}");
        let pad = 64 - (10 + h.len() + 1) % 64;
        let h = format!("{h}{:pad$} ", "", pad = pad);
        v.extend_from_slice(&(h.len() as u16).to_le_bytes());
        v.extend_from_slice(h.as_bytes());
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn reads_a_c_order_2d_array() {
        let mut body = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let a = read_f32_bytes(&build("<f4", "(2, 3)", "False", &body)).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// The reference's mel dumps are a transposed view, so numpy records them as
    /// Fortran order; getting this wrong transposes the whole comparison.
    #[test]
    fn reads_a_fortran_order_2d_array() {
        let mut body = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        // stored as two columns of three: (0,0) (1,0) (2,0) (0,1) (1,1) (2,1)
        let a = read_f32_bytes(&build("<f4", "(2, 3)", "True", &body)).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn rejects_the_wrong_dtype() {
        let a = build("<f8", "(1,)", "False", &[0u8; 8]);
        assert!(read_f32_bytes(&a).is_err());
    }

    #[test]
    fn reads_integer_index_arrays() {
        let mut body = Vec::new();
        for v in [7i64, 3, 99] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let a = read_f32_bytes(&build("<i8", "(3,)", "False", &body)).unwrap();
        assert_eq!(a.shape, vec![3]);
        assert_eq!(a.data, vec![7.0, 3.0, 99.0]);
    }
}
