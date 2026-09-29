//! Stage-2 validation: does the Rust encoder reproduce the reference layer by layer?
//!
//! Diffs against `../baseline/hidden/diarization_example__offline_*.npy`, which are
//! forward hooks on the real `model.model.audio_tower.layers[i]`.
//!
//! Those dumps are the concatenation of the reference's *internal* calls: for a 97.6 s
//! file the offline path calls the encoder with `[380, 684, 684, 505]` frames. This
//! checker runs the same first call (340 scored + 40 right-context encoder frames) and
//! compares only that first slice, which is enough to localise any divergence.
//!
//! Usage: cargo run --release --bin encoder_check

use std::path::{Path, PathBuf};

use nemotron3_diarization_wgpu::{
    load_configs, log_mel, mel_filters, Padding, Tower,
};

/// Same tolerance story as the front-end: fp32 arithmetic in a different order.
const TOLERANCE: f64 = 1e-2;

fn read_npy_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    if &b[..6] != b"\x93NUMPY" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not .npy"));
    }
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    Ok(b[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn stats(got: &[f32], want: &[f32], n: usize) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut sum = 0.0f64;
    for i in 0..n {
        let d = (got[i] as f64 - want[i] as f64).abs();
        max_abs = max_abs.max(d);
        sum += d * d;
    }
    (max_abs, (sum / n as f64).sqrt())
}

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from("../baseline");
    let ckpt = PathBuf::from("../models/Nemotron-3-Diarization");

    let (model, proc) = load_configs(&ckpt)?;
    let fe = &proc.feature_extractor;
    let filters = mel_filters(fe);
    let tower = Tower::load(&model.audio_config, &nemotron3_diarization_wgpu::Weights::load(&ckpt)?)?;

    // the exact 16 kHz mono waveform the reference saw
    let wave: Vec<f32> = read_npy_f32(&root.join("waveforms").join("diarization_example.npy"))?;
    println!(
        "tower: {} layers, d_model {}, heads {}, head_dim {}, ffn {}",
        model.audio_config.num_hidden_layers,
        model.audio_config.hidden_size,
        model.audio_config.num_attention_heads,
        model.audio_config.hidden_size / model.audio_config.num_attention_heads,
        model.audio_config.intermediate_size
    );
    println!("input: {} samples = {:.2}s\n", wave.len(), wave.len() as f64 / 16000.0);

    // The first internal call of the offline path: 340 encoder frames + 40 right-context.
    //
    // The mel must be computed over the *whole file* and then sliced, not over a
    // 3040-frame slice: the last frame of a slice sits on the zero padding instead of
    // on real audio, which perturbs its 512-bin window and, through the embedder, the
    // final encoder group. The reference computes features once for the whole file.
    let chunk_enc = model.chunk_length + model.chunk_right_context; // 380
    let chunk_mel = chunk_enc * proc.subsampling_factor;           // 3040
    let full = log_mel(&wave, fe, Padding::Centered, &filters);
    println!(
        "whole-file features: {} mel frames; first offline call takes {} -> {} encoder frames \
         ({} scored + {} look-ahead)",
        full.len() / fe.feature_size, chunk_mel, chunk_enc, model.chunk_length, model.chunk_right_context
    );
    let mel = full[..chunk_mel * fe.feature_size].to_vec();

    // dump our mel so it can be diffed against a freshly computed Python one
    std::fs::write("/tmp/rust_mel.f32", bytemuck::cast_slice(&mel)).ok();
    let emb_dbg = tower.embed(&mel, chunk_mel);
    std::fs::write("/tmp/rust_embed.f32", bytemuck::cast_slice(&emb_dbg)).ok();

    // snapshot every layer we have a reference for, plus the boundaries
    let have: Vec<String> = std::fs::read_dir(root.join("hidden"))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let layer_of = |name: &str| -> Option<usize> {
        name.strip_prefix("diarization_example__offline__layer")
            .and_then(|r| r.strip_suffix(".npy"))
            .and_then(|r| r.parse().ok())
    };
    let mut snap_layers: Vec<usize> = have.iter().filter_map(|n| layer_of(n)).collect();
    snap_layers.sort_unstable();
    println!("reference layers available: {snap_layers:?}\n");

    // Every frame the processor marked valid participates in attention, *including* the
    // 40 look-ahead frames: those are real audio that later steps re-score, not padding.
    // `step_mask` is just `embed_mask[start : start + chunk_width]`, and a fully-valid
    // file makes all 380 of them visible. Masking them would shrink every query's key
    // set from 380 to 340 and blow the whole block up.
    let (final_out, snapshots) = tower.forward(&mel, chunk_mel, None, 0, &snap_layers);
    let n_frames = chunk_enc;
    let cols = tower.hidden_size();

    println!("{:<22}{:>10}{:>10}  {}", "tensor", "max|d|", "rms", "verdict");
    println!("{}", "-".repeat(58));

    let mut worst = 0.0f64;
    let mut report = |name: &str, got: &[f32], want: &[f32], n: usize| {
        let (mx, rms) = stats(got, want, n);
        worst = worst.max(mx);
        let v = if mx < TOLERANCE { "OK" } else { "FAIL" };
        println!("{name:<22}{mx:>10.3e}{rms:>10.3e}  {v}");
    };

    for (file, label) in [
        ("embedder", "embedder"),
        ("input_ln", "input_layer_norm"),
    ] {
        if let Ok(want) = read_npy_f32(&root.join("hidden").join(format!("diarization_example__offline__{file}.npy"))) {
            let n = (want.len() / cols).min(n_frames) * cols;
            // embedder output has no public accessor; recompute via the tower's own path
            let emb = tower.embed(&mel, chunk_mel);
            let got = if file == "input_ln" {
                tower.apply_input_ln(&emb, n_frames)
            } else {
                emb
            };
            report(label, &got, &want, n);
        }
    }

    for s in &snapshots {
        let want = read_npy_f32(&root.join("hidden").join(format!(
            "diarization_example__offline__layer{:02}.npy",
            s.layer
        )))?;
        let n = (want.len() / cols).min(n_frames) * cols;
        report(&format!("layer{:02}", s.layer), &s.states, &want, n);
    }

    if let Ok(want) = read_npy_f32(&root.join("hidden").join("diarization_example__offline__final_ln.npy")) {
        let n = (want.len() / cols).min(n_frames) * cols;
        report("final_layer_norm", &final_out, &want, n);
    }

    println!("\nworst max|d| = {worst:.3e}  (tolerance {TOLERANCE:.0e})");
    if worst < TOLERANCE {
        println!("encoder matches the reference layer by layer.");
        Ok(())
    } else {
        anyhow::bail!("encoder diverges from the reference");
    }
}
