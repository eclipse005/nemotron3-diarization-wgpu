# Nemotron 3 Diarization — wgpu

A native Rust implementation of [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
that runs on **wgpu**, so it works on NVIDIA, Intel and AMD GPUs, and on Apple silicon —
Vulkan, Metal, DX12 or GL — from a single binary. No CUDA, no PyTorch, no Python at runtime.

It loads the **same `model.safetensors`** as the official HuggingFace transformers code and
reproduces its output frame for frame.

---

## Accuracy

The implementation is validated against the official PyTorch CUDA runtime over a
curated test set covering 2/3/4 speakers, 0–61% overlap, Chinese and English, telephone
band-pass and SNR-8 dB reverb, plus a 34.4-minute recording:

| | result |
|---|---|
| mode-runs compared | **40 / 40** (10 files × 4 modes) |
| segments IDENTICAL | **40 / 40** |
| `sigmoid(logit) > 0.5` decision flips | **0** (including 206 653 frames of the 34.4-min clip) |
| worst per-logit `maxdiff` | **1.041e-3** (tolerance 2e-2 — a 19× margin) |

Re-deriving the speaker timeline from the logits reproduces the reference segment for
segment. See *Verification* below for how to reproduce this.

## Build

```bash
cargo build --release
```

Requires a Rust toolchain and a wgpu-compatible driver. The GPU path uses the
`Vulkan / Metal / DX12` backends; the same binary also has a CPU path for machines
without a usable GPU adapter.

## Usage

```bash
diarize <audio.wav|audio.npy> [--mode MODE] [--model DIR] [--out FILE] [--gpu]
```

- `--gpu` runs the compute kernels through wgpu. Without it, the pure-Rust CPU path runs.
- `--model` points at the directory holding `model.safetensors` (default: next to the
  checkpoint the binary was built against).
- `--out` writes a JSON report; without it the report goes to stdout.

The report contains the speaker segments plus per-run statistics (`rtfx`, `forward_s`,
`cache_frames`, …).

```bash
# offline diarization on the GPU
diarize meeting.wav --mode offline --gpu --model /path/to/Nemotron-3-Diarization

# same thing through the CPU path
diarize meeting.wav --mode offline
```

### Modes

| mode | encoder frames / step | algorithmic latency |
|---|---|---|
| `offline` | full-length, chunked | none |
| `low_latency` | 9 | 1040 ms |
| `very_low_latency` | 6 | 640 ms |
| `ultra_low_latency` | 3 | 320 ms |

All four produce the same frame count; they differ in how much context each frame sees.

## Layout

```
src/
  lib.rs        front-end, encoder, head, streaming glue
  config.rs     checkpoint / processor config
  mel.rs        preemphasis, STFT, slaney mel, log
  encoder.rs    31-layer pre-LN transformer tower
  head.rs       speaker classification head
  streaming.rs  AOSC + FIFO speaker-cache state machine
  segments.rs   logits -> speaker segments
  gpu.rs        compute pipelines (GEMM, attention, softmax, rope)
  gpu_engine.rs the wgpu engine: buffers, dispatch, readback
  npy.rs        .npy reader (used by the validation binaries)
  bin/          CLI and validation binaries
```

### Binaries

| binary | what it does |
|---|---|
| `diarize` | the CLI above |
| `frame_check` | diffs a whole run against the Python reference (logits, flips, segments) |
| `gpu_check` | probes individual kernels — GEMM, attention, QKV, rope, softmax, encoder range |
| `front_end_check`, `encoder_check`, `head_check`, `compress_check` | per-stage checks |
| `head_trace` | dumps speaker-cache internals for debugging |
| `spike` | scratch binary for one-off experiments |

## Verification

`frame_check` runs a clip through the engine and diffs it against reference logits
produced by the official PyTorch implementation:

```bash
FRAME_CHECK_GPU=1 cargo run --release --bin frame_check <reference-dir> <file-stem> <model-dir>
```

It reports, per mode, the worst per-logit `maxdiff`, the number of `sigmoid > 0.5`
decision flips, and whether the re-derived segments are `IDENTICAL` to the reference.
Three gates must hold: `maxdiff < 2e-2`, `flips == 0`, segments `IDENTICAL`.

> `FRAME_CHECK_GPU=1` is not optional. It selects the GPU path (`Model::load_gpu`);
> without it the binary quietly runs the CPU path, and the only symptom is that it
> becomes roughly 40× slower.

The reference tree itself is produced by the sibling `nemotron3-diarization` project,
which wraps the official transformers code. That project also ships
`regress_testset.py`, a one-command regression over the 10-file test set.

## Notes

- The GEMM kernel is hand-written WGSL. On sm_61 it sustains 1.155 TFLOP/s at
  `m = 380` — 43.7% of the card's fp32 peak, and 92% of what a hand-written CUDA
  kernel reaches on the same shapes. (cuBLAS reaches 83.3%.) It is **not** a
  cuBLAS-beating kernel; it is a cross-vendor one that runs everywhere.
- `silence_embeds` is not added to the encoder output. It only fills the reserved
  silence slots when the speaker cache is compressed.
- fp32 throughout.

## License

Apache-2.0, matching the upstream model.
