//! Stage-3 validation: does the classification head reproduce the reference?
//!
//! The head is checked in two halves, because a bad end-to-end number does not say
//! *which* half is wrong:
//!
//! 1. **the encoder, driven through the real offline loop** — the recorded window
//!    lengths are printed next to the reference's, and each window's tower output is
//!    diffed against the matching slice of
//!    `baseline/hidden/<file>__offline__final_ln.npy`. The reference dump is the four
//!    chunk windows concatenated, so slice `k` of it *is* window `k`.
//! 2. **the head alone** — the reference's own `final_ln` rows for the first window go
//!    through `proj -> Conv1d sub-pixel upsample -> classifier`, and the result is
//!    diffed against the first `340 * 8` rows of `baseline/frames/<file>__offline.npy`.
//!
//! Feeding the *reference's* hidden states into our head is deliberate: it removes the
//! encoder from the equation, so a head bug cannot hide behind encoder noise.
//!
//! Usage:  cargo run --release --bin head_check [../baseline] [file-stem]

use std::path::PathBuf;

use nemotron3_diarization_wgpu::{read_f32, ChunkTrace, Model};

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../baseline".into()));
    let stem = std::env::args().nth(2).unwrap_or_else(|| "diarization_example".into());
    let ckpt = PathBuf::from(
        std::env::args().nth(3).unwrap_or_else(|| "../models/Nemotron-3-Diarization".into()),
    );

    let samples: Vec<f32> = read_f32(&root.join("waveforms").join(format!("{stem}.npy")))?
        .data
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let model = Model::load(&ckpt)?;
    let hidden = model.hidden_size();

    let final_ln = read_f32(&root.join("hidden").join(format!("{stem}__offline__final_ln.npy")))?;
    let frames_ref = read_f32(&root.join("frames").join(format!("{stem}__offline.npy")))?;
    let nf = final_ln.rows();
    println!("{stem}: reference has {nf} encoder frames across its windows, {} mel frames of logits",
        frames_ref.rows());

    // --- 1. the encoder, through the real chunk loop ----------------------------
    let mut trace: Vec<ChunkTrace> = Vec::new();
    model.run_offline_capturing(&samples, Some(&mut trace))?;

    let mut cursor = 0usize;
    let mut enc_worst = 0.0f64;
    println!("\nchunk  cache  window  valid  scored");
    for (k, t) in trace.iter().enumerate() {
        println!(
            "{k:>5}  {:>5}  {:>6}  {:>5}  {:>6}",
            t.cached_frames, t.window_frames, t.valid_frames, t.scored_frames
        );
        let (lo, hi) = (cursor * hidden, (cursor + t.window_frames) * hidden);
        cursor += t.window_frames;
        if hi > final_ln.data.len() {
            println!("   ^ beyond the reference dump ({} rows)", final_ln.rows());
            continue;
        }
        let d = (0..hi - lo)
            .map(|i| (t.encoder_out[i] as f64 - final_ln.data[lo + i]).abs())
            .fold(0.0f64, f64::max);
        enc_worst = enc_worst.max(d);
        println!("        maxdiff vs reference window: {d:.3e}");
    }
    println!("\nencoder through the chunk loop: worst maxdiff {enc_worst:.3e} over {cursor} frames");

    // --- 1b. the speaker cache, against PyTorch's own SpeakerCache --------------
    // `compress` is the only place in the model where a *ranking* decides the output,
    // so a one-frame disagreement in the scores reshuffles the whole cache and every
    // later chunk. Comparing the cache after each step localises it exactly.
    let cmp = |name: &str, ours: &[f32], file: &str| -> f64 {
        let want = read_f32(std::path::Path::new(file)).map(|a| a.data).unwrap_or_default();
        if want.len() != ours.len() {
            println!("  {name:<12} LENGTH {} vs reference {}", ours.len(), want.len());
            return f64::INFINITY;
        }
        let d = ours
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b).abs())
            .fold(0.0f64, f64::max);
        println!("  {name:<12} {:>7} values, maxdiff {d:.3e}", ours.len());
        d
    };
    let mut cache_worst = 0.0f64;
    if std::path::Path::new("/tmp/s0_probs.npy").exists() {
        println!("\nspeaker cache (dumps from export_cache.py):");
        for (i, t) in trace.iter().enumerate() {
            println!("  step {i} (after update: {} cache + {} fifo frames)",
                t.cache_embeds.len() / hidden, t.fifo_embeds.len() / hidden);
            if i == 0 {
                cache_worst = cache_worst.max(cmp("  step_probs", &t.step_probs, "/tmp/s0_probs.npy"));
            }
            cache_worst = cache_worst.max(cmp("  cache_embeds", &t.cache_embeds, &format!("/tmp/s{i}_cache_e.npy")));
            cache_worst = cache_worst.max(cmp("  cache_probs", &t.cache_probs, &format!("/tmp/s{i}_cache_p.npy")));
            cache_worst = cache_worst.max(cmp("  fifo_embeds", &t.fifo_embeds, &format!("/tmp/s{i}_fifo_e.npy")));
        }
    }

    // --- 2. the head alone ------------------------------------------------------
    let take = trace.first().map_or(0, |t| t.window_frames);
    let ref_hidden: Vec<f32> = (0..take * hidden)
        .map(|i| final_ln.data[i] as f32)
        .collect();
    let got = model.head.forward(&ref_hidden, take);
    let want_frames = (model.cfg.chunk_length * model.proc.subsampling_factor).min(frames_ref.rows());
    let rc = frames_ref.cols();

    let (mut max_abs, mut sum_sq) = (0.0f64, 0.0f64);
    for t in 0..want_frames {
        for s in 0..rc {
            let d = (got[t * rc + s] as f64 - frames_ref.data[t * rc + s]).abs();
            sum_sq += d * d;
            max_abs = max_abs.max(d);
        }
    }
    let rms = (sum_sq / (want_frames * rc) as f64).sqrt();
    println!("\nhead on the reference's first {take} hidden frames -> {want_frames} logit rows");
    println!("  maxdiff {max_abs:.3e}  rms {rms:.3e}");

    let ok = enc_worst < 2e-2 && max_abs < 2e-2 && cache_worst < 1e-3;
    if ok {
        println!("\nboth the encoder windows and the head match the reference.");
        Ok(())
    } else {
        eprintln!(
            "\nstage 3 does NOT match the reference (encoder {enc_worst:.3e}, head {max_abs:.3e}, cache {cache_worst:.3e})"
        );
        std::process::exit(1);
    }
}
