# Nemotron 3 Diarization wgpu

**用 Rust + wgpu 实现的说话人日志分离。**

[English](README.md) · **简体中文**

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的轻量级、跨平台 Rust 实现，对标官方
[Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)。
手写 CPU + wgpu 推理——不依赖 Python、torch 或 CUDA toolkit。

目标很简单：在任何 wgpu 能驱动的显卡（Vulkan / DX12 / Metal / OpenGL）上，或者纯 CPU 上，
本地、原生地算出**谁在什么时候说话**，同时逐帧复现官方 transformers 的输出。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 Vulkan / DX12 / Metal / OpenGL
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 完全离线本地推理
* 🗣️ 最多 8 个并发说话人
* 🔁 offline 加三种流式模式（算法延迟 320–1040 毫秒）
* ✅ 与官方 PyTorch 运行时逐位一致
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
nemotron3-diarization-wgpu = { git = "https://github.com/eclipse005/nemotron3-diarization-wgpu.git" }
```

或者从源码编译 CLI：

```bash
git clone https://github.com/eclipse005/nemotron3-diarization-wgpu.git
cd nemotron3-diarization-wgpu
cargo build --release        # target/release/diarize
cargo test --release --lib   # 29 个单测
```

不需要 CUDA、Python 或任何外部工具链——`wgpu` 在运行时直接链接系统的图形驱动
（Vulkan / DX12 / GL / Metal）。

### 模型下载

本仓库**不包含**权重。请从 Hugging Face 下载 checkpoint（权利归属原作者），
把下载好的目录原样传给 `--model`：

- [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

如果你把参考项目作为同级目录检出，`--model` 会自动找到它，可以省略这个参数。

### 快速开始

```bash
diarize meeting.wav --mode offline --gpu --model ./Nemotron-3-Diarization
```

| 参数 | 说明 |
|---|---|
| `--mode <name>` | `offline`、`low_latency`、`very_low_latency` 或 `ultra_low_latency`，默认 `offline` |
| `--gpu` | 用 wgpu 跑计算核；不加则走纯 Rust CPU 路径 |
| `--model <dir>` | 存放 `model.safetensors` 的目录 |
| `--out <file>` | 把 JSON 报告写入文件，而不是打到标准输出 |

输入为 16 kHz 音频。`.wav`（16-bit PCM 或 float32，任意声道数，会下混为单声道）和
`.npy` 都能读。报告里包含说话人分段，以及每次运行的统计信息
（`rtfx`、`forward_s`、`cache_frames`、`num_speakers` 等）。

### 运行模式

| 模式 | 每步 encoder 帧数 | 算法延迟 |
|---|---|---|
| `offline` | 整段前向，内部再分块 | 无 |
| `low_latency` | 9 | 1040 毫秒 |
| `very_low_latency` | 6 | 640 毫秒 |
| `ultra_low_latency` | 3 | 320 毫秒 |

四种模式产出的帧数完全相同，区别在于每一帧能看到多少右侧上下文——流式档位用一点精度换延迟。

### 精度

这是一个**移植**，不是重写：加载同一份 `model.safetensors`，复现官方 transformers 的输出。

在一组精选测试集上与官方 PyTorch CUDA 运行时逐帧对拍——10 个文件，覆盖 2/3/4 说话人、
0–61% 重叠、中英文、电话带通与 SNR 8 dB 混响，外加一条 34.4 分钟的长录音：

| | 结果 |
|---|---|
| 对拍运行次数 | **40 / 40**（10 文件 × 4 模式） |
| 分段完全一致 | **40 / 40** |
| `sigmoid(logit) > 0.5` 判决翻转 | **0**（含 34.4 分钟那条的 206 553 帧） |
| 最大逐 logit 偏差 | **1.041e-3**（容差 2e-2，余量 19 倍） |

从 logits 重新推导出的说话人时间线与参考逐条相同，且没有任何一帧的判决发生翻转。

offline **没有时长上限**：音频按窗口逐段 embed，而不是整段物化在显存里，所以几小时的会议录音可以直接跑。

### 作为库使用

```rust
use nemotron3_diarization_wgpu::{Model, StreamingMode, extract_speaker_dict};

// GPU 路径
let model = Model::load_gpu("Nemotron-3-Diarization".as_ref())?;

// Offline
let out = model.run_offline(&samples_16k_mono)?;
let dur = samples_16k_mono.len() as f32 / 16_000.0;
let segs = extract_speaker_dict(
    &out.logits, out.num_frames, model.num_speakers(), dur, None, 0.5,
);

// 流式，算法延迟 320 毫秒
let out = model.run_streaming(&samples_16k_mono, StreamingMode::UltraLowLatency)?;
```

`Model::load` 提供完全相同的 API，走纯 Rust CPU 路径。完整接口见 `cargo doc`。

### 验证

`frame_check` 把一段音频跑过引擎，再与官方 PyTorch 实现产出的参考 logits 逐帧比对：

```bash
FRAME_CHECK_GPU=1 cargo run --release --bin frame_check <参考目录> <文件名> <模型目录>
```

它会按模式报告最大逐 logit 偏差、`sigmoid > 0.5` 的判决翻转数，以及重新推导出的分段是否
`IDENTICAL`。三道门禁必须同时成立：`maxdiff < 2e-2`、`flips == 0`、分段 `IDENTICAL`。

> `FRAME_CHECK_GPU=1` 不是可选项。它决定走 GPU 路径（`Model::load_gpu`）；不加的话程序会
> 悄悄退回 CPU 路径，唯一的表现就是慢约 40 倍。

`gpu_check` 更细一层，逐个核做对拍——GEMM、attention、QKV、rope、softmax，以及 encoder 的
区间 embedding。`front_end_check`、`encoder_check`、`head_check`、`compress_check` 覆盖中间的各个阶段。

参考树由另一个封装了官方 transformers 代码的项目产出；跑推理不需要它，只有对拍才需要。

### 为什么用 wgpu？

不用 CUDA、ROCm 这类厂商专属运行时，而是把 **wgpu** 当作统一的 GPU 抽象层。

这样才能为不同平台、不同显卡厂商构建**同一套** Rust 分离运行时。

GEMM 核是手写 WGSL。在 sm_61 上，`m = 380` 时可达 1.155 TFLOP/s——是该卡 fp32 峰值的 43.7%，
也达到了同形状下手写 CUDA 核的 92%（cuBLAS 是 83.3%）。它不是一个打败 cuBLAS 的核，
而是一个跨厂商、到处都能跑的核。

### 项目状态

✅ **逐位对齐，功能完整**

已在 10 文件 × 4 模式上与官方运行时对拍，判决翻转为零。低端显卡上的吞吐受硬件限制
（见「为什么用 wgpu」），后续性能工作面向更新的显卡。

### 相关项目

* [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) — 官方模型
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — 语音识别
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — 词级时间戳

### 许可

Apache-2.0，与上游模型保持一致。

本仓库是用于加载和运行官方发布的 Nemotron-3-Diarization 权重的**独立 Rust 推理实现**，
并非 NVIDIA 官方发布，与原作者无隶属关系。模型权重权利归属各自所有者。
