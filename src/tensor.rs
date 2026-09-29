//! Small dense tensor helpers for the CPU reference path.
//!
//! Everything is row-major `Vec<f32>` with an explicit `(rows, cols)` shape. The wgpu
//! kernels will replace these, so nothing here is allowed to be clever: each function
//! is the most literal transcription of the PyTorch op it stands for, which is what
//! makes a numerical diff against the baseline meaningful.

use rayon::prelude::*;

/// `y = x @ w^T + bias`, with `w` stored as `[out, in]` (PyTorch's `nn.Linear` layout).
///
/// Hand-rolled rather than delegated to a BLAS wrapper: this is the CPU *reference*
/// path, and owning the summation order keeps the diff against PyTorch attributable
/// to the model rather than to whichever kernel a library happened to pick.
pub fn linear(x: &[f32], rows: usize, cols: usize, w: &[f32], out: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    y.par_chunks_mut(out).enumerate().for_each(|(r, yrow)| {
        let xrow = &x[r * cols..(r + 1) * cols];
        for o in 0..out {
            let wrow = &w[o * cols..(o + 1) * cols];
            let mut acc = 0.0f32;
            for i in 0..cols {
                acc += xrow[i] * wrow[i];
            }
            yrow[o] = acc + bias.map_or(0.0, |b| b[o]);
        }
    });
    y
}

/// `y = a @ b` for row-major `[m, k] @ [k, n] -> [m, n]`.
pub fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(r, crow)| {
        let arow = &a[r * k..(r + 1) * k];
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += arow[i] * b[i * n + j];
            }
            crow[j] = acc;
        }
    });
    c
}

/// `torch.nn.LayerNorm` over the last dim, eps 1e-5 (PyTorch's default).
pub fn layer_norm(x: &[f32], rows: usize, cols: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * cols];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let src = &x[r * cols..(r + 1) * cols];
        let mean = src.iter().sum::<f32>() / cols as f32;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..cols {
            row[i] = (src[i] - mean) * inv * w[i] + b[i];
        }
    });
    y
}

/// GELU, exact erf form — `ACT2FN["gelu"]` is `GELUActivation` with
/// `approximate="none"`, i.e. `0.5 * x * (1 + erf(x / sqrt(2)))`.
pub fn gelu(x: &[f32], len: usize) -> Vec<f32> {
    use libm::erff;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    x[..len]
        .par_iter()
        .map(|&v| 0.5 * v * (1.0 + erff(v * INV_SQRT2)))
        .collect()
}

/// ReLU, for the classification head's `act_fn`.
pub fn relu(x: &[f32], len: usize) -> Vec<f32> {
    x[..len].par_iter().map(|&v| v.max(0.0)).collect()
}

/// Numerically stable softmax over the last dim, computed in f32 like the reference
/// (`softmax(..., dtype=torch.float32)` in `eager_attention_forward`).
pub fn softmax_last(x: &mut [f32], rows: usize, cols: usize) {
    x.par_chunks_mut(cols).for_each(|row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_hand_computation() {
        // [[1,2,3],[4,5,6]] (2x3) @ [[1,0],[0,1],[0,0]] (3x2) -> [[1,2],[4,5]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let c = matmul(&a, 2, 3, &b, 2);
        assert_eq!(c, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn linear_matches_hand_computation() {
        // 2x3 input, 2x3 weight, no bias
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let y = linear(&x, 2, 3, &w, 2, None);
        assert_eq!(&y[0..2], &[1.0, 2.0]);
        assert_eq!(&y[2..4], &[4.0, 5.0]);
    }

    #[test]
    fn layer_norm_normalises() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0];
        let y = layer_norm(&x, 1, 4, &w, &b);
        let mean = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        let var = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var should be ~1, got {var}");
    }

    #[test]
    fn gelu_matches_known_values() {
        // exact erf form: 0.5*x*(1 + erf(x/sqrt(2)))
        //   gelu(0)  = 0
        //   gelu(1)  = 0.5*(1 + erf(0.70711)) = 0.8413447
        //   gelu(-1) = 0.5*(1 - erf(0.70711)) = -0.1586553
        let g = gelu(&[0.0, 1.0, -1.0], 3);
        assert!(g[0].abs() < 1e-6);
        assert!((g[1] - 0.841_344_7).abs() < 1e-6, "gelu(1) = {}", g[1]);
        assert!((g[2] + 0.158_655_3).abs() < 1e-6, "gelu(-1) = {}", g[2]);
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut x = vec![1000.0, 1000.0, 1000.0, -1e30];
        softmax_last(&mut x, 1, 4);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum {s}");
        assert!(x[3] < 1e-6, "masked entry should vanish, got {}", x[3]);
    }
}
