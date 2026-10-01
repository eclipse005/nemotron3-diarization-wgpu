//! Stage-3 validation: does `_compress` keep the same frames as the reference?
//!
//! Compression is a **ranking**: 264 frames are chosen out of 300 by score, and the
//! choice silently reshapes every later chunk. A wrong pick still produces plausible
//! logits and a plausible transcript, so it has to be diffed directly. This compares
//! each stage against the dumps written by `export_cache.py`:
//!
//! | stage | reference file |
//! |---|---|
//! | frame scores + latest-frames boost | `c_scores.npy` |
//! | after strong boost | `c_scores_boosted.npy` |
//! | after weak boost | `c_scores_final.npy` |
//! | flattened speaker-major scores | `c_flat.npy` |
//! | retained indices, sorted | `c_topk_sorted.npy` |
//!
//! Usage:  cargo run --release --bin compress_check [prefix] [model-dir]
use std::path::PathBuf;

use nemotron3_diarization_wgpu::{read_f32, CacheConfig, Model, SpeakerCache};

fn main() -> anyhow::Result<()> {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "/tmp/c".into());
    let ckpt = PathBuf::from(
        std::env::args().nth(2).unwrap_or_else(|| "../models/Nemotron-3-Diarization".into()),
    );
    let rd = |n: &str| -> anyhow::Result<Vec<f64>> {
        Ok(read_f32(&PathBuf::from(format!("{prefix}_{n}.npy")))?.data)
    };
    let probs: Vec<f32> = rd("probs_in")?.into_iter().map(|v| v as f32).collect();
    let model = Model::load(&ckpt)?;
    let hidden = model.hidden_size();
    let ns = model.num_speakers();
    let n_frames = probs.len() / ns;
    let embeds = vec![0.0f32; n_frames * hidden];

    let cache = SpeakerCache::new(CacheConfig::offline(&model.cfg), hidden);
    println!(
        "compress over {n_frames} frames -> {} kept, {} silence slots per speaker",
        cache.cfg.speaker_cache_length, cache.cfg.num_silence_frames
    );
    let t = cache.trace_compress(&embeds, &probs);

    let mut worst = 0.0f64;
    let mut cmp = |name: &str, ours: &[f32], want: &[f64]| {
        if ours.len() != want.len() {
            println!("  {name:<14} LENGTH {} vs {}", ours.len(), want.len());
            worst = worst.max(f64::INFINITY);
            return;
        }
        let mut bad = 0usize;
        let mut maxd = 0.0f64;
        for (a, b) in ours.iter().zip(want) {
            let d = (*a as f64 - *b).abs();
            if d > 1e-3 {
                bad += 1;
            }
            maxd = maxd.max(d);
        }
        println!(
            "  {name:<14} {:>6} values, maxdiff {maxd:.3e}, {bad} differ by >1e-3",
            ours.len()
        );
        worst = worst.max(maxd);
    };
    cmp("scores", &t.raw, &rd("scores")?);
    cmp("latest-boost", &t.after_latest, &rd("scores_boosted")?);
    cmp("strong-boost", &t.after_strong, &rd("scores_strong")?);
    cmp("weak-boost", &t.scores, &rd("scores_final")?);
    cmp("flat", &t.flat, &rd("flat")?);

    let want_idx: Vec<usize> = rd("topk_sorted")?.iter().map(|&v| v as usize).collect();
    if t.picked.len() == want_idx.len() {
        let first = t.picked.iter().zip(&want_idx).position(|(a, b)| a != b);
        println!(
            "  {:<14} {:>6} indices, first difference at {:?}",
            "picked", t.picked.len(), first
        );
        if let Some(i) = first {
            let lo = i.saturating_sub(3);
            println!("      ours {:?}", &t.picked[lo..(i + 6).min(t.picked.len())]);
            println!("      ref  {:?}", &want_idx[lo..(i + 6).min(want_idx.len())]);
        }
        if first.is_some() {
            worst = worst.max(1.0);
        }
    } else {
        println!("  picked         LENGTH {} vs {}", t.picked.len(), want_idx.len());
        worst = worst.max(f64::INFINITY);
    }

    if worst < 1e-3 {
        println!("\ncompress keeps exactly the frames the reference keeps.");
        Ok(())
    } else {
        eprintln!("\ncompress diverges (worst {worst:.3e})");
        std::process::exit(1);
    }
}
