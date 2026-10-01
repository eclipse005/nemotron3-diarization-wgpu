# Nemotron 3 Diarization wgpu

**Speaker diarization in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration. Hand-written CPU + wgpu inference — no Python, no torch, no CUDA toolkit.

The goal is simple: work out **who spoke when**, locally and natively, on any GPU that wgpu can drive (Vulkan, DX12, Metal, OpenGL) or on the CPU — while reproducing the official transformers output **frame for frame**.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Vulkan / DX12 / Metal / OpenGL
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* 🗣️ Up to 8 concurrent speakers
* 🔁 Offline plus three streaming modes (320–1040 ms algorithmic latency)
* ✅ Bit-accurate against the official PyTorch runtime
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
nemotron3-diarization-wgpu = { git = "https://github.com/eclipse005/nemotron3-diarization-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/nemotron3-diarization-wgpu.git
cd nemotron3-diarization-wgpu
cargo build --release        # target/release/diarize
cargo test --release --lib   # 29 unit tests
```

No CUDA, no Python, no external toolkit — `wgpu` links against the system graphics driver (Vulkan / DX12 / GL / Metal) at runtime.

### Model download

Weights are **not** included in this repository. Download the checkpoint from Hugging Face (rights remain with the original authors) and pass the downloaded directory to `--model` unchanged:

- [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

If you keep a checkout of the reference project as a sibling directory, `--model` finds it automatically and you can omit the flag.

### Quick Start

```bash
diarize meeting.wav --mode offline --gpu --model ./Nemotron-3-Diarization
```

| Option | Description |
|--------|-------------|
| `--mode <name>` | `offline`, `low_latency`, `very_low_latency` or `ultra_low_latency`. Default `offline` |
| `--gpu` | Run the compute kernels through wgpu. Without it the pure-Rust CPU path runs |
| `--model <dir>` | Directory holding `model.safetensors` |
| `--out <file>` | Write the JSON report to a file instead of stdout |

Input is 16 kHz audio. `.wav` (16-bit PCM or float32, any channel count — downmixed to mono) and `.npy` are both accepted. The report carries the speaker segments plus per-run statistics (`rtfx`, `forward_s`, `cache_frames`, `num_speakers`, …).

### Modes

| mode | encoder frames / step | algorithmic latency |
|---|---|---|
| `offline` | full-length, chunked | none |
| `low_latency` | 9 | 1040 ms |
| `very_low_latency` | 6 | 640 ms |
| `ultra_low_latency` | 3 | 320 ms |

All four produce the same frame count. They differ in how much right-context each frame sees, so the streaming modes trade a little accuracy for latency.

### Accuracy

This is a **port**, not a re-implementation: it loads the same `model.safetensors` and reproduces the official transformers output.

Validated against the official PyTorch CUDA runtime over a curated test set — 10 files covering 2/3/4 speakers, 0–61% overlap, Chinese and English, telephone band-pass and SNR-8 dB reverb, plus a 34.4-minute recording:

| | result |
|---|---|
| mode-runs compared | **40 / 40** (10 files × 4 modes) |
| segments IDENTICAL | **40 / 40** |
| `sigmoid(logit) > 0.5` decision flips | **0** (including 206 553 frames of the 34.4-min clip) |
| worst per-logit `maxdiff` | **1.041e-3** (tolerance 2e-2 — a 19× margin) |

The speaker timeline re-derived from the logits matches the reference segment for segment, and no frame's diarization verdict ever flips.

Offline has no duration limit: the recording is embedded window by window instead of being materialised whole, so hour-long meetings work unchanged.

### Library

```rust
use nemotron3_diarization_wgpu::{Model, StreamingMode, extract_speaker_dict};

// GPU path
let model = Model::load_gpu("Nemotron-3-Diarization".as_ref())?;

// Offline
let out = model.run_offline(&samples_16k_mono)?;
let dur = samples_16k_mono.len() as f32 / 16_000.0;
let segs = extract_speaker_dict(
    &out.logits, out.num_frames, model.num_speakers(), dur, None, 0.5,
);

// Streaming, 320 ms algorithmic latency
let out = model.run_streaming(&samples_16k_mono, StreamingMode::UltraLowLatency)?;
```

`Model::load` gives the same API on the pure-Rust CPU path. See `cargo doc` for the full surface.

### Verification

`frame_check` runs a clip through the engine and diffs it against reference logits produced by the official PyTorch implementation:

```bash
FRAME_CHECK_GPU=1 cargo run --release --bin frame_check <reference-dir> <file-stem> <model-dir>
```

It reports, per mode, the worst per-logit `maxdiff`, the number of `sigmoid > 0.5` decision flips, and whether the re-derived segments are `IDENTICAL`. Three gates must hold: `maxdiff < 2e-2`, `flips == 0`, segments `IDENTICAL`.

> `FRAME_CHECK_GPU=1` is not optional. It selects the GPU path (`Model::load_gpu`); without it the binary quietly runs the CPU path, and the only symptom is that it becomes roughly 40× slower.

`gpu_check` goes one level down and probes individual kernels — GEMM, attention, QKV, rope, softmax, and the encoder's range embedding. `front_end_check`, `encoder_check`, `head_check` and `compress_check` cover the stages in between.

The reference tree is produced by a separate project that wraps the official transformers code; it is not needed to run inference, only to verify against it.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction.

This makes it possible to build a single Rust-based diarization runtime for different platforms and GPU vendors.

The GEMM kernel is hand-written WGSL. On sm_61 it sustains 1.155 TFLOP/s at `m = 380` — 43.7% of that card's fp32 peak, and 92% of what a hand-written CUDA kernel reaches on the same shapes (cuBLAS reaches 83.3%). It is not a cuBLAS-beating kernel; it is a cross-vendor one that runs everywhere.

### Project Status

✅ **Bit-accurate, feature-complete**

Verified against the official runtime across 10 files × 4 modes with zero decision flips. Throughput on low-end cards is hardware-bound (see *Why wgpu?*), so further performance work targets newer GPUs.

### Related

* [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) — the official model
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — speech recognition
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — word-level timestamps

### License

Apache-2.0, matching the upstream model.

This repository is an **independent Rust inference implementation** for loading and running the officially released Nemotron-3-Diarization weights — not an official NVIDIA release, and not affiliated with the original authors. Model weights remain under the terms of their respective owners.
