//! Stage-3/4 validation: does the whole CPU path reproduce the reference's logits?
//!
//! Runs [`Model::run_offline`] and [`Model::run_streaming`] for all four modes on the
//! reference clip and diffs the result against `baseline/frames/<file>__<mode>.npy`.
//! It also re-derives the speech segments and diffs those against
//! `baseline/segments/<file>__<mode>.json`, because identical logits with different
//! segments (or vice versa) would point at a different bug than a plain value diff.
//!
//! The reference the diff is against is the **CUDA** baseline, so the tolerance also
//! absorbs PyTorch's own CPU/GPU gap (1.5e-4 on this clip) plus our summation order.
//! What must be exactly zero is the *decision* count: the number of frames whose
//! `sigmoid(logit) > 0.5` verdict differs from the reference.
//!
//! Usage:  cargo run --release --bin frame_check [../baseline] [file-stem]

use std::path::PathBuf;
use std::time::Instant;

use nemotron3_diarization_wgpu::{
    activity_stats, extract_speaker_dict, frame_duration, read_f32, Model, StreamingMode,
};

/// Max allowed |ours - reference| per logit. Comfortably above the 1.5e-4 CPU/CUDA gap:
/// 31 layers of hand-rolled fp32 accumulation accumulate their own rounding.
const TOLERANCE: f64 = 2e-2;

/// Live Python CUDA fp32 RTFx on this box (`run_offline.py` / `run_streaming.py`,
/// diarization_example, 2026-09-29). GPU mode must strictly exceed these.
const RTFX_BARS: [(&str, f64); 4] = [
    ("offline", 123.0),
    ("low_latency", 9.1),
    ("very_low_latency", 6.3),
    ("ultra_low_latency", 3.2),
];

const MODES: [(&str, StreamingMode); 4] = [
    ("offline", StreamingMode::Offline),
    ("low_latency", StreamingMode::LowLatency),
    ("very_low_latency", StreamingMode::VeryLowLatency),
    ("ultra_low_latency", StreamingMode::UltraLowLatency),
];

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../baseline".into()));
    let stem = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diarization_example".into());
    let ckpt = PathBuf::from(
        std::env::args().nth(3).unwrap_or_else(|| "../models/Nemotron-3-Diarization".into()),
    );

    let samples: Vec<f32> = read_f32(&root.join("waveforms").join(format!("{stem}.npy")))?
        .data
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let model = if std::env::var("FRAME_CHECK_GPU").is_ok() {
        let m = Model::load_gpu(&ckpt)?;
        if let Some(d) = m.gpu_describe() {
            println!("gpu: {d}");
        }
        m
    } else {
        Model::load(&ckpt)?
    };

    println!("{stem}: {} samples @ 16 kHz = {:.2} s", samples.len(), samples.len() as f64 / 16000.0);
    println!("speakers {} | encoder {} layers, hidden {} | upsample x{}",
        model.num_speakers(), model.cfg.audio_config.num_hidden_layers,
        model.cfg.audio_config.hidden_size, model.proc.subsampling_factor);
    println!("offline  : chunks of {} + {} right-context encoder frames",
        model.cfg.chunk_length, model.cfg.chunk_right_context);
    let scored = |m: StreamingMode| {
        let p = model.params(m);
        p.chunk_encoder_frames()
    };
    println!(
        "streaming: low {} / very {} / ultra {} encoder frames per step, {} / {} / {} ms latency\n",
        scored(StreamingMode::LowLatency),
        scored(StreamingMode::VeryLowLatency),
        scored(StreamingMode::UltraLowLatency),
        model.params(StreamingMode::LowLatency).latency_ms,
        model.params(StreamingMode::VeryLowLatency).latency_ms,
        model.params(StreamingMode::UltraLowLatency).latency_ms,
    );

    // `--plan` prints the chunk schedule and stops: the geometry is the easiest part
    // to get wrong and the slowest to notice, since a wrong schedule still produces
    // plausible-looking logits.
    if std::env::var("FRAME_CHECK_PLAN").is_ok() {
        for (name, mode) in MODES {
            if mode == StreamingMode::Offline {
                let fe = &model.proc.feature_extractor;
                let valid = nemotron3_diarization_wgpu::frame_count(
                    samples.len(),
                    fe,
                    nemotron3_diarization_wgpu::Padding::Centered,
                );
                println!(
                    "{name:<18} 1 chunk, {} mel frames (+1 zeroed) -> {} encoder frames",
                    valid,
                    valid.div_ceil(model.proc.subsampling_factor) + 1
                );
                continue;
            }
            let p = model.params(mode);
            let chunks = model.streaming_chunks(&samples, mode)?;
            let frames: usize = chunks.iter().map(|c| c.num_mel_frames).sum();
            let mut emitted = 0usize;
            for c in &chunks {
                emitted += (c.num_mel_frames.div_ceil(p.subsampling) - c.lookahead) * p.subsampling;
            }
            let last = chunks.last().unwrap();
            let last_samples = last.mel.len() / model.proc.feature_extractor.feature_size;
            println!(
                "{name:<18} {:>3} chunks | {frames} mel frames, {emitted} emitted \
| first {:>5} samples centred, later {:>5} uncentred \
| last {last_samples:>5} samples -> {} frames, no look-ahead | {} ms",
                chunks.len(),
                p.first_chunk_samples,
                p.samples_per_chunk,
                last.num_mel_frames,
                p.latency_ms,
            );
        }
        return Ok(());
    }

    let dur = frame_duration(model.proc.feature_extractor.hop_length, model.proc.feature_extractor.sampling_rate);
    let mut worst = 0.0f64;
    let mut all_ok = true;

    // `FRAME_CHECK_ONLY=offline,low_latency` limits the run; the streaming modes take
    // hours on a 4-core CPU box, so they are worth being able to run one at a time.
    let only: Vec<String> = std::env::var("FRAME_CHECK_ONLY")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();

    for (name, mode) in MODES {
        if !only.is_empty() && !only.iter().any(|m| m == name) {
            continue;
        }
        let t0 = Instant::now();
        let out = match mode {
            StreamingMode::Offline => model.run_offline(&samples)?,
            _ => model.run_streaming(&samples, mode)?,
        };
        let elapsed = t0.elapsed().as_secs_f64();

        let ref_path = root.join("frames").join(format!("{stem}__{name}.npy"));
        let reference = read_f32(&ref_path)?;
        let (rr, rc) = (reference.rows(), reference.cols());
        let ns = model.num_speakers();

        let audio_s = samples.len() as f64 / 16000.0;
        let rtfx = audio_s / elapsed.max(1e-9);
        print!(
            "{name:<18} {elapsed:6.2}s  RTFx {rtfx:6.1}  chunks {:>4}  frames {}/{}  ",
            out.num_chunks, out.num_frames, rr
        );
        if out.num_frames != rr {
            println!("FAIL (frame count differs)");
            all_ok = false;
            continue;
        }

        // --- logits ---
        let (mut max_abs, mut sum_sq) = (0.0f64, 0.0f64);
        let mut worst_frame = 0usize;
        let mut flips = 0usize;
        let mut sign_flips = 0usize;
        for t in 0..rr {
            for s in 0..rc {
                let got = out.logits[t * ns + s] as f64;
                let want = reference.data[t * rc + s];
                let d = (got - want).abs();
                sum_sq += d * d;
                if d > max_abs {
                    max_abs = d;
                    worst_frame = t;
                }
                if (got > 0.0) != (want > 0.0) {
                    sign_flips += 1;
                }
                let sg = 1.0 / (1.0 + (-got as f32).exp());
                let sw = 1.0 / (1.0 + (-want as f32).exp());
                if (sg > 0.5) != (sw > 0.5) {
                    flips += 1;
                }
            }
        }
        let n = (rr * rc) as f64;
        let rms = (sum_sq / n).sqrt();
        worst = worst.max(max_abs);

        let logits_ok = max_abs < TOLERANCE;
        let decisions_ok = flips == 0;
        all_ok &= logits_ok && decisions_ok;
        println!("maxdiff {max_abs:.3e} rms {rms:.3e}  flips {flips}/{n}  logit-sign {sign_flips}");
        if std::env::var("FRAME_CHECK_GPU").is_ok() {
            let bar = RTFX_BARS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, b)| *b)
                .unwrap_or(0.0);
            if rtfx <= bar {
                println!("   ^ FAIL: RTFx {rtfx:.1} does not exceed original CUDA bar {bar:.1}");
                all_ok = false;
            }
        }
        if !logits_ok {
            println!(
                "   ^ FAIL: worst at frame {worst_frame} ({:.2}s); tolerance {TOLERANCE:.1e}",
                worst_frame as f64 * dur as f64
            );
        }
        if sign_flips > 0 {
            println!("   ^ note: {sign_flips} logits crossed zero (well away from the 0.5 threshold)");
        }

        // --- segments ---
        let segs = extract_speaker_dict(&out.logits, out.num_frames, ns, dur, None, 0.5);
        let (pct, per_speaker) = activity_stats(&out.logits, out.num_frames, ns, None, 0.5);
        let seg_path = root.join("segments").join(format!("{stem}__{name}.json"));
        let ref_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&seg_path)?)?;
        let want_segs = ref_json["segments"].as_array().cloned().unwrap_or_default();
        let got_json: Vec<serde_json::Value> = segs.iter().map(|s| s.to_json()).collect();

        let identical = want_segs == got_json;
        let first_diff = want_segs.iter().zip(&got_json).position(|(a, b)| a != b);
        let spk: Vec<usize> = (0..ns)
            .filter(|&s| per_speaker[s] > 0)
            .collect();
        let want_spk = ref_json["num_speakers"].as_u64().unwrap_or(0) as usize;
        let seg_ok = identical;
        all_ok &= seg_ok;

        println!(
            "  segments {}  ours {} / ref {}  first diff {:?}  speakers {}/{}  speech {:.2}% (ref {:.2}%)",
            if identical { "IDENTICAL" } else { "DIFFER   " },
            got_json.len(),
            want_segs.len(),
            first_diff,
            spk.len(),
            want_spk,
            pct,
            ref_json["speech_frame_pct"].as_f64().unwrap_or(0.0),
        );
        if let Some(i) = first_diff {
            println!("    first differing segment #{i}");
            println!("      ref  {}", want_segs.get(i).unwrap_or(&serde_json::Value::Null));
            println!("      ours {}", got_json.get(i).unwrap_or(&serde_json::Value::Null));
        }
        let want_per: Vec<u64> = ref_json["per_speaker_frames"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect())
            .unwrap_or_default();
        if want_per.len() == per_speaker.len() && want_per.iter().zip(&per_speaker).any(|(a, b)| *a != *b as u64) {
            println!("    per-speaker frame counts differ: ref {want_per:?} ours {per_speaker:?}");
        }
    }

    println!("\nworst logit maxdiff = {worst:.3e} (tolerance {TOLERANCE:.1e})");
    if all_ok {
        println!("frame-level output matches the reference for every mode checked.");
        Ok(())
    } else {
        eprintln!("frame-level output does NOT match the reference");
        std::process::exit(1);
    }
}
