# wgpu/ — Rust port of Nemotron 3 Diarization

Target: a native Rust implementation that loads the **same `model.safetensors`** as the Python
reference and reproduces its output, so it can run without CUDA (and without Python).

This directory is intentionally empty of code — it holds the contract. Start here.

## Where the reference lives

All paths relative to the project root (`/home/mh/nemotron3-diarization`):

| what | path | size |
|---|---|---|
| weights (the only thing you load) | `models/Nemotron-3-Diarization/model.safetensors` | 397 MB, 417 tensors, all F32 |
| model config | `models/Nemotron-3-Diarization/config.json` | |
| front-end / streaming config | `models/Nemotron-3-Diarization/processor_config.json` | |
| tensor inventory + derived dims | `models/Nemotron-3-Diarization/port_manifest.json` | |
| file checksums | `models/Nemotron-3-Diarization/manifest.json` | |
| **Python baseline** | `baseline/` | see below |

Verify you have the right weights: `model.safetensors` sha256 starts with `c074d86335b3b794`
(full value in `models/Nemotron-3-Diarization/manifest.json`).

## The baseline you must match

Built by `uv run build_baseline.py` over all 29 official test files × 4 inference modes.

```
baseline/
├── baseline.json                     everything: per file, per mode, stats + all segments
├── waveforms/<name>.npy              the exact 16 kHz mono f32 waveform the model saw
├── frames/<name>__<mode>.npy         per-frame logits (T, 8) f32, PRE-sigmoid
├── segments/<name>__<mode>.json      extract_speaker_dict output + stats
├── frontend/<name>__<mode>.npz       first 104 mel frames, attention mask, chunk sizes
└── hidden/<name>__offline__layer*.npy   every encoder hidden state, reference file only
```

`<mode>` ∈ `offline`, `low_latency`, `very_low_latency`, `ultra_low_latency`.

### The front-end, exactly

This is where ports usually go wrong. Copied from
`transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py`:

```python
# 1. preemphasis on the RAW WAVEFORM, before the STFT (not on the mel)
y[0] = x[0]
y[i] = x[i] - 0.97 * x[i-1]          # samples past the end are zeroed

# 2. STFT
window = hann_window(400, periodic=False)   # NOT periodic!
stft = stft(y, n_fft=512, hop=160, win=400, window, center=<first chunk|offline>)
power = |stft|^2                              # sqrt(re^2+im^2) then squared

# 3. mel
mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney")
feat = log(mel @ power + 2**-24)             # natural log, guard 2^-24 ≈ 5.96e-8
feat = feat.T                                # -> (frames, 128)
```

Three traps:

1. **`norm="slaney"`** — the HF `mel_filter_bank` helper defaults to HTK and float64, and the
   source even has a commented-out block noting that they switched to librosa *because* the two
   disagree numerically. Use slaney/slaney (`htk=False`).
2. **preemphasis is on the waveform**, before the STFT. Applying it to mel output is wrong.
3. **natural log, not log10 and not dB.** `-16.01` is the floor (`log(2^-24)`), not a dB value.

Frame counts (this is how the processor decides a chunk is valid):

| `center` | valid frames |
|---|---|
| `True` (offline + first streaming chunk) | `floor(L / hop)` |
| `False` (later streaming chunks) | `floor((L - n_fft) / hop) + 1` |

with `L` the chunk's sample count. `center=True` pads `n_fft//2 = 256` zeros on both sides
(`pad_mode="constant"`); `center=False` does not pad, which is why a later chunk must start at
`frame*hop - n_fft//2` to line up frame-for-frame with a full pass.

## Offline mode is NOT a stateless full-length forward

The most important structural fact, and the one most likely to be missed. From
`modeling_nemotron3_diarization.py`:

```python
chunk_length, chunk_right_context = config.chunk_length, config.chunk_right_context   # 340, 40

for start_idx in range(0, num_chunk_embeds, chunk_length):
    chunk_embeds = inputs_embeds[:, start_idx : min(start_idx + 340 + 40, num_embeds)]
    cached = speaker_cache.get_embeds(chunk_embeds)          # AOSC + FIFO prepended
    chunk_input_embeds = torch.cat([cached, chunk_embeds], dim=1)
    position_ids = torch.arange(chunk_input_embeds.shape[1]) # RESTART AT 0 EVERY CHUNK
    outputs = encoder(chunk_input_embeds, attention_mask=..., position_ids=position_ids)
```

So the encoder runs **once per 340-frame chunk (27.2 s)**, each call seeing
`[AOSC 264 | FIFO 40 | chunk 340 | right-context 40]`, and **RoPE positions restart at 0 on every
call — with the cached frames occupying positions *before* the current chunk.** Offline uses the
same speaker-cache code as streaming, just with offline sizes: `fifo_length=40`,
`speaker_cache_update_period=300` (streaming uses 264 / 222).

Measured encoder sequence lengths per call:

| audio | encoder frames | calls |
|---|---|---|
| ≤ 30.4 s | ≤ 380 | **one single call**, no cache |
| 60 s | 751 | `[380, 684, 375]` |
| 100 s | 1251 | `[380, 684, 684, 535]` |
| 200 s | 2501 | `[380, 684 ×6, 425]` |

`684 = 264 + 40 + 380`; the trailing call is the remainder. A file of ≤ 30.4 s is a single
stateless forward, which is why `two_speakers.wav` (60 s) still exercises the cache.

**If your port does one full-length forward for offline, the numbers will not match.** Either
replicate the chunked loop with the cache, or validate only against ≤ 30.4 s files.

## How to use it while developing

- **While iterating on the port**: one file is enough —
  `samples/official/two_speakers.wav` (60 s, ground truth = exactly 2 speakers, stable across all
  four modes). `baseline/frames/two_speakers__*.npy` is your target.
- **Before calling the port done**: re-run every file and every mode and diff against `baseline/`.
  A per-layer check on the reference file will localise any divergence immediately.

### Suggested order of implementation

Each stage is independently checkable against a file in `baseline/`:

1. **decode + resample** → `waveforms/<name>.npy` (16 kHz mono f32)
2. **mel front-end** → `frontend/<name>__<mode>.npz` (`input_features` is 128-bin log-mel,
   10 ms hop, preemphasis 0.97). This is where porting bugs usually hide; check it first.
3. **chunking** → the per-mode sample counts in the `.npz` (`first_chunk_samples`,
   `samples_per_audio_chunk`, `mel_frames_per_step`, `num_lookahead_frames`)
4. **embedder** — ×8 frame stacking `[512, 1024]` → 512, then `input_layer_norm`
5. **31 encoder blocks** — pre-LN, q/k/v have **no bias**, o_proj has bias, MHA with 8 kv heads
   (no GQA), RoPE theta 10000, GELU FFN 2048 → `hidden/<name>__offline__layer*.npy`
6. **head** — `+silence_embeds` → `proj` → `upsampler` Conv1d k=3 → `classifier` → `frames/*.npy`
7. **streaming state (AOSC + FIFO)** — only needed for the 3 streaming modes

## Tolerances

Everything is fp32. wgpu compute shaders reorder reductions, so expect drift, not equality.
Roughly:

| quantity | expected max abs diff vs torch |
|---|---|
| mel `input_features` | 1e-3 (different STFT/mel summation order) |
| encoder hidden states | 1e-2 early layers, ~1e-1 by layer 31 |
| final logits | ~1e-1 |
| `sigmoid(logit) > 0.5` speaker decision | should be identical except on a handful of frames |

Judge correctness on the **derived output** (speaker activity, segments, speaker count), not on
logit equality. A frame is "correct" if the argmax/binarised speaker matches; a logit differing by
0.05 that does not flip any decision is fine. Logit differences large enough to flip many decisions
mean a real bug — most often the mel front-end or a transposed weight.

## Suggested tooling

- [`candle`](https://github.com/huggingface/candle) — has `safetensors` loading and a `cublas`/
  Metal/CPU backend, and a `wgpu` backend is a natural fit. Safest choice for matching HF semantics.
- [`burn`](https://github.com/tracel-ai/burn) — if you want a training-capable framework; wgpu
  backend is first class, but you will be writing more of the glue yourself.
- Plain `wgpu` + `safetensors` crate — most control, most work. Fine for this model: it is a
  31-layer 512-dim encoder, no exotic ops (no GQA, no bias-fused QKV, no flash-attn requirement).

`silence_embeds`, `classifier.out_proj` and the AOSC bookkeeping are the pieces HF hides behind
higher-level abstractions — those are where "it runs but the numbers differ" usually comes from.
