//! Command-line diarizer.
//!
//! Runs the ported model on a 16 kHz mono WAV (or on a `.npy` float32 waveform, which
//! is what the Python baseline stores) and writes the speaker segments as JSON in the
//! reference's `{Start, End, Speaker}` format.
//!
//! Audio decoding is deliberately *not* implemented: the reference pipeline resamples
//! to 16 kHz mono before anything else, and a port that resampled differently would
//! silently diverge from the baseline. Feed it audio that is already 16 kHz mono.
//!
//! ```text
//! cargo run --release --bin diarize -- <audio> [--mode offline|low_latency|...] \
//!     [--model DIR] [--out FILE] [--threshold 0.5]
//! ```

use std::path::PathBuf;

use nemotron3_diarization_wgpu::{
    extract_speaker_dict, frame_duration, read_f32, Model, StreamingMode,
};

const MODES: [(&str, StreamingMode); 4] = [
    ("offline", StreamingMode::Offline),
    ("low_latency", StreamingMode::LowLatency),
    ("very_low_latency", StreamingMode::VeryLowLatency),
    ("ultra_low_latency", StreamingMode::UltraLowLatency),
];

/// Where to look for `model.safetensors` when `--model` is not given.
///
/// The reference project (which owns the weights) is normally checked out as a
/// sibling directory, so try that first and fall back to an in-tree `models/`.
fn default_model_dir() -> PathBuf {
    const NAME: &str = "Nemotron-3-Diarization";
    for cand in [
        PathBuf::from("../nemotron3-diarization/models").join(NAME),
        PathBuf::from("../models").join(NAME),
    ] {
        if cand.join("model.safetensors").exists() {
            return cand;
        }
    }
    PathBuf::from("../models").join(NAME)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut audio: Option<PathBuf> = None;
    let mut mode = StreamingMode::Offline;
    // Default to a sibling checkout of the reference project, which is where the
    // weights live; fall back to an in-tree `models/` for older layouts. `--model`
    // overrides both.
    let mut model_dir = default_model_dir();
    let mut out: Option<PathBuf> = None;
    let mut threshold = 0.5f32;
    let mut use_gpu = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                let name = args.get(i).ok_or_else(|| anyhow::anyhow!("--mode needs a value"))?;
                mode = MODES
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, m)| *m)
                    .ok_or_else(|| anyhow::anyhow!("unknown mode {name}"))?;
            }
            "--model" => {
                i += 1;
                model_dir = PathBuf::from(args.get(i).ok_or_else(|| anyhow::anyhow!("--model needs a value"))?);
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).ok_or_else(|| anyhow::anyhow!("--out needs a value"))?));
            }
            "--threshold" => {
                i += 1;
                threshold = args.get(i).ok_or_else(|| anyhow::anyhow!("--threshold needs a value"))?.parse()?;
            }
            "--gpu" => use_gpu = true,
            "-h" | "--help" => {
                println!("usage: diarize <audio.wav|audio.npy> [--mode MODE] [--model DIR] [--out FILE] [--gpu]");
                println!("modes: offline, low_latency, very_low_latency, ultra_low_latency");
                return Ok(());
            }
            other if other.starts_with("--") => anyhow::bail!("unknown flag {other}"),
            other => audio = Some(PathBuf::from(other)),
        }
        i += 1;
    }
    let audio = audio.ok_or_else(|| anyhow::anyhow!("no audio file given (try --help)"))?;

    let t0 = std::time::Instant::now();
    let samples = read_audio(&audio)?;
    let model = if use_gpu {
        let m = Model::load_gpu(&model_dir)?;
        if let Some(d) = m.gpu_describe() {
            eprintln!("gpu: {d}");
        }
        m
    } else {
        Model::load(&model_dir)?
    };
    let loaded = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let result = match mode {
        StreamingMode::Offline => model.run_offline(&samples)?,
        _ => model.run_streaming(&samples, mode)?,
    };
    let infer = t1.elapsed().as_secs_f64();

    let fe = &model.proc.feature_extractor;
    let dur = frame_duration(fe.hop_length, fe.sampling_rate);
    let segments = extract_speaker_dict(
        &result.logits,
        result.num_frames,
        model.num_speakers(),
        dur,
        None,
        threshold,
    );
    let speakers: Vec<usize> = (0..model.num_speakers())
        .filter(|&s| segments.iter().any(|g| g.speaker == s))
        .collect();
    let duration_s = samples.len() as f64 / fe.sampling_rate as f64;
    let rtfx = if infer > 0.0 { duration_s / infer } else { f64::INFINITY };

    let report = serde_json::json!({
        "audio": audio.display().to_string(),
        "mode": format!("{mode:?}").to_lowercase(),
        "duration_s": (duration_s * 1000.0).round() / 1000.0,
        "num_frames": result.num_frames,
        "num_chunks": result.num_chunks,
        "num_segments": segments.len(),
        "num_speakers": speakers.len(),
        "speakers": speakers,
        "cache_compressed": result.cache_compressed,
        "cache_frames": result.cache_frames,
        "fifo_frames": result.fifo_frames,
        "load_s": (loaded * 1000.0).round() / 1000.0,
        "forward_s": (infer * 1000.0).round() / 1000.0,
        "rtfx": (rtfx * 100.0).round() / 100.0,
        "segments": segments.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&report)?;
    match &out {
        Some(p) => {
            std::fs::write(p, format!("{text}\n"))?;
            eprintln!(
                "{:.2}s audio, {} mode: {} segments, {} speakers in {:.2}s (RTFx {:.1}) -> {}",
                duration_s, format!("{mode:?}").to_lowercase(), segments.len(),
                speakers.len(), infer, rtfx, p.display()
            );
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// 16 kHz mono in, `Vec<f32>` out. `.npy` is the baseline's own waveform dump; `.wav`
/// is read with `hound` and downmixed to mono.
fn read_audio(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    if path.extension().is_some_and(|e| e == "npy") {
        return Ok(read_f32(path)?.data.into_iter().map(|v| v as f32).collect());
    }
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels >= 1,
        "{} has no channels",
        path.display()
    );
    anyhow::ensure!(
        spec.sample_format == hound::SampleFormat::Float || spec.bits_per_sample == 16,
        "{} must be 16-bit PCM or float32",
        path.display()
    );
    let ch = spec.channels as usize;
    let float = spec.sample_format == hound::SampleFormat::Float;
    // hound only lets you read samples in the encoding the file actually uses, so
    // 16-bit PCM has to be read as `i16` and scaled here. Reading it as `f32` is the
    // single most common wav format and used to fail with "The sample format differs
    // from the destination format".
    let mut out = Vec::new();
    if float {
        for s in reader.samples::<f32>() {
            out.push(s?);
        }
    } else {
        for s in reader.samples::<i16>() {
            out.push(s? as f32 / 32768.0);
        }
    }
    anyhow::ensure!(out.len() % ch == 0, "{} is not a whole number of frames", path.display());
    if ch == 1 {
        return Ok(out);
    }
    Ok(out.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect())
}
