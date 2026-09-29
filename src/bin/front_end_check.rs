//! Stage-1 validation: does the Rust front-end reproduce the reference's mel features?
//!
//! Reads the Python baseline's `baseline/frontend/<file>__<mode>.npz` dumps — the
//! first 104 mel frames of the first chunk — and diffs them against what this crate
//! computes from `baseline/waveforms/<file>.npy`.
//!
//! Because the baseline stores the exact 16 kHz mono waveform the reference saw,
//! this check is closed: no decoder, no resampler, only our own maths.
//!
//! Usage:  cargo run --release --bin front_end_check [../baseline]

use std::io::Read;
use std::path::{Path, PathBuf};

use nemotron3_diarization_wgpu::{
    frame_count, load_configs, log_mel, mel_filters, Padding, LOG_ZERO_GUARD_VALUE,
};

/// Max allowed |ours - reference| per mel value. Above this the maths is wrong;
/// below it, it is just FFT summation order.
const TOLERANCE: f64 = 1e-2;

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../baseline".into()));
    let ckpt = PathBuf::from(std::env::args().nth(2).unwrap_or_else(
        || "../models/Nemotron-3-Diarization".into(),
    ));

    let (model, proc) = load_configs(&ckpt)?;
    let fe = &proc.feature_extractor;
    let filters = mel_filters(fe);

    println!("front-end config: {} mels, n_fft {}, win {}, hop {}, preemph {}",
        fe.feature_size, fe.n_fft, fe.win_length, fe.hop_length, fe.preemphasis);
    println!("mel filterbank: {} x {} (slaney, fmin 0, fmax {})",
        fe.feature_size, fe.n_fft / 2 + 1, fe.sampling_rate / 2);
    println!(
        "log guard 2^-24 = {e:.6e}  ->  floor ln(...) = {floor:.4}",
        e = LOG_ZERO_GUARD_VALUE,
        floor = LOG_ZERO_GUARD_VALUE.ln()
    );
    println!("offline: chunk_length {} + right_context {} encoder frames = {} ms\n",
        model.chunk_length, model.chunk_right_context,
        (model.chunk_length + model.chunk_right_context) * 80);

    let mut worst: f64 = 0.0;
    let mut checked = 0usize;

    let mut files: Vec<PathBuf> = std::fs::read_dir(root.join("frontend"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "npz"))
        .collect();
    files.sort();

    for npz in &files {
        let stem = npz.file_stem().unwrap().to_string_lossy().to_string();
        let (file, mode) = stem.rsplit_once("__").unwrap_or((stem.as_str(), "offline"));

        let wave_path = root.join("waveforms").join(format!("{file}.npy"));
        let Ok(samples) = read_npy_f32(&wave_path) else {
            println!("{stem:<56} SKIP (no waveform)");
            continue;
        };
        let Ok(ref_feats) = npz_member_f32(npz, "input_features.npy") else {
            println!("{stem:<56} SKIP (no input_features)");
            continue;
        };

        // Every stored dump is the *first* chunk, and every first chunk uses centred
        // windows — offline included, because the whole file is centred too.
        let centered = Padding::Centered;
        let n_samples = if mode == "offline" {
            samples.len()
        } else {
            npz_member_i64(npz, "first_chunk_samples.npy").unwrap_or(0) as usize
        };
        let chunk = &samples[..n_samples.min(samples.len())];
        let got = log_mel(chunk, fe, centered, &filters);

        // the npz is padded to 104 frames; compare the valid prefix
        let valid = frame_count(chunk.len(), fe, centered);
        let n = valid.min(ref_feats.len() / fe.feature_size).min(got.len() / fe.feature_size);
        if n == 0 {
            println!("{stem:<56} SKIP (no valid frames)");
            continue;
        }

        let mut max_abs = 0.0f64;
        let mut sum = 0.0f64;
        let mut per_frame = vec![0.0f64; n];
        for t in 0..n {
            for m in 0..fe.feature_size {
                let i = t * fe.feature_size + m;
                let d = (got[i] as f64 - ref_feats[i] as f64).abs();
                per_frame[t] = per_frame[t].max(d);
                max_abs = max_abs.max(d);
                sum += d * d;
            }
        }
        let rms = (sum / (n * fe.feature_size) as f64).sqrt();
        worst = worst.max(max_abs);
        checked += 1;
        let flag = if max_abs < TOLERANCE { "OK  " } else { "FAIL" };
        let first_bad = per_frame.iter().position(|&d| d > 1e-3);
        let detail = match first_bad {
            None => String::new(),
            Some(t) => format!(
                "  first frame >1e-3: {t:>4} (of {n})   per-frame tail: {:?}",
                per_frame[n.saturating_sub(4)..]
                    .iter()
                    .map(|d| format!("{d:.1e}"))
                    .collect::<Vec<_>>()
            ),
        };
        println!("{flag} {stem:<52} frames {n:>4}  maxdiff {max_abs:.3e}  rms {rms:.3e}{detail}");
    }

    // 1e-2 is the fp32 tolerance: the only remaining difference is the FFT itself,
    // where rustfft and torch accumulate in a different order. Feature values live in
    // [-16.64, ~2], so 1e-2 is ~1e-3 relative.
    println!("\n{checked} front-end dumps checked, worst maxdiff = {worst:.3e}");
    if worst < TOLERANCE {
        println!("front-end matches the reference within tolerance.");
        Ok(())
    } else {
        eprintln!("front-end does NOT match the reference");
        std::process::exit(1);
    }
}

/// Read a flat f32 `.npy` (little endian), honouring the layout flag.
fn read_npy_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let (dtype, offset, fortran) = npy_header(&bytes)?;
    if dtype != "<f4" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected <f4, got {dtype}"),
        ));
    }
    let data = &bytes[offset..];
    let vals: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape = npy_shape(&bytes)?;
    Ok(untangle(vals, fortran, shape))
}

/// Pull one member out of a `.npz` (a zip) and decode it as f32.
fn npz_member_f32(npz: &Path, member: &str) -> std::io::Result<Vec<f32>> {
    let bytes = zip_member(npz, member)?;
    let (dtype, offset, fortran) = npy_header(&bytes)?;
    if dtype != "<f4" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected <f4, got {dtype}"),
        ));
    }
    let vals: Vec<f32> = bytes[offset..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape = npy_shape(&bytes)?;
    Ok(untangle(vals, fortran, shape))
}

/// Reorder a column-major array into row-major.
///
/// The reference computes `mel_spec` as `(batch, n_mels, frames)` and then
/// `permute(0, 2, 1)`, so the tensor handed to `.numpy()` is non-contiguous and
/// keeps column-major strides. `np.savez` then records `fortran_order: True` for
/// the offline dumps, while the streaming ones — sliced to a contiguous prefix —
/// are C order. Both have to be read correctly.
fn untangle(vals: Vec<f32>, fortran: bool, shape: (usize, usize)) -> Vec<f32> {
    if !fortran {
        return vals;
    }
    let (rows, cols) = shape;
    let mut out = vec![0.0f32; vals.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = vals[c * rows + r];
        }
    }
    out
}

fn npz_member_i64(npz: &Path, member: &str) -> Option<i64> {
    let bytes = zip_member(npz, member).ok()?;
    let (_dtype, offset, _f) = npy_header(&bytes).ok()?;
    let d = bytes.get(offset..offset + 8)?;
    Some(i64::from_le_bytes(d.try_into().ok()?))
}

fn zip_member(npz: &Path, member: &str) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(npz)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut f = archive
        .by_name(member)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, member.to_string()))?;
    let mut out = Vec::new();
    f.read_to_end(&mut out)?;
    Ok(out)
}

/// Returns `(dtype, data offset, fortran_order)` for a little-endian `.npy`.
fn npy_header(bytes: &[u8]) -> std::io::Result<(String, usize, bool)> {
    if &bytes[..6] != b"\x93NUMPY" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a .npy file",
        ));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "no descr in .npy header")
        })?
        .to_string();
    let fortran = header.contains("'fortran_order': True");
    Ok((descr, 10 + header_len, fortran))
}

/// The 2-D `shape` recorded in a `.npy` header.
fn npy_shape(bytes: &[u8]) -> std::io::Result<(usize, usize)> {
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .and_then(|s| {
            let dims: Vec<usize> = s
                .trim_start()
                .trim_start_matches('(')
                .split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(|x| x.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            (!dims.is_empty()).then_some(dims)
        })
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "no shape in .npy header")
        })?;
    match shape.as_slice() {
        [r, c] => Ok((*r, *c)),
        [n] => Ok((*n, 1)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected a 2-D array, got shape {shape:?}"),
        )),
    }
}
