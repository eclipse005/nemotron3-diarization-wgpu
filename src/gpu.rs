//! wgpu device and the f32 compute kernels the GPU encoder / head dispatch.
//!
//! Every kernel is a literal transcription of the matching CPU helper in
//! [`crate::tensor`]: `linear` is `y = x @ W^T + bias` with W stored `[out, in]`,
//! layer-norm uses the population variance, GELU is the erf form. Reduction order
//! differs (workgroup tiles vs a scalar loop), so the GPU path is allowed the
//! wgpu README's ~1e-2 hidden-state budget against the CPU reference.

use crate::error::{DiarizationError, Result};

const WG_LN: u32 = 256;
/// GEMM output tile: 8×16 threads × 16×8 registers = 128×128.
pub const GEMM_BM: u32 = 128;
pub const GEMM_BN: u32 = 128;
/// Row tile of the *linear* GEMM dispatch, which `GEMM_BM=64` shrinks to 64 (a
/// 4×8 register tile) so the grid reaches 2·gx·gy128 tiles and the ragged last
/// wave is a smaller share of the work. The strided attention kernels keep
/// `GEMM_BM` — they are separate pipelines and their workgroups are sized by the
/// head layout, not by this choice.
///
/// Cached, and that is not a micro-optimisation: this sits on the per-dispatch
/// path (~190 calls per step), and a plain `std::env::var` there allocates a
/// `String` and walks the whole environment. Leaving it uncached cost ~10% of
/// end-to-end RTFx — more than the kernel change was worth.
pub fn gemm_bm() -> u32 {
    static BM: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BM.get_or_init(|| match std::env::var("GEMM_BM").ok().as_deref() {
        Some("64") => 64,
        _ => GEMM_BM,
    })
}
const ATTN_WG: u32 = 64;
const MAX_ATTN_SEQ: u32 = 1024;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        // Vulkan, Metal, DX12, GL — the same set qwen3-asr-wgpu uses, so one
        // binary covers NVIDIA, AMD, Intel and Apple GPUs.
        let backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let mut adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(backends).await;
        if adapters.is_empty() {
            return Err(DiarizationError::Gpu("no wgpu adapters (Vulkan/GL)".into()));
        }
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0u8,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        });
        let adapter = &adapters[0];
        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();
        let ts = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nemotron3-diarization"),
                required_features: features & ts,
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| DiarizationError::Gpu(format!("request_device: {e}")))?;
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e}");
        }));
        Ok(Self {
            device,
            queue,
            info,
            limits,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} | binding {} MiB, workgroup storage {} B",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024),
            self.limits.max_compute_workgroup_storage_size,
        )
    }

    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str) -> wgpu::Buffer {
        self.uniform_sized(label, 16)
    }

    /// A uniform buffer big enough for a struct other than [`U4`]. The strided GEMM
    /// needs 12 `u32` for its strides and bases.
    pub fn uniform_sized(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    /// Record a `dst <- data` copy through a freshly mapped staging buffer.
    ///
    /// [`Queue::write_buffer`](wgpu::Queue::write_buffer) is fine for the one-off
    /// weight uploads, but on this box the per-window encoder input costs about 90 ms
    /// for 1.4 MB — ~25% of the whole offline run. A reused [`Uploader`] moves the
    /// bytes with a plain memcpy instead.
    pub fn stage_copy(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        data: &[f32],
    ) {
        let bytes = (data.len() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload"),
            size: bytes.div_ceil(4).max(4) * 4,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = staging.slice(..).get_mapped_range_mut().expect("map upload staging");
            view.copy_from_slice(bytemuck::cast_slice(data));
        }
        staging.unmap();
        enc.copy_buffer_to_buffer(&staging, 0, dst, 0, bytes);
    }

    pub fn upload_f32(&self, buf: &wgpu::Buffer, data: &[f32]) {
        self.upload(buf, bytemuck::cast_slice(data));
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| DiarizationError::Gpu(format!("poll: {e}")))?;
        rx.recv()
            .map_err(|e| DiarizationError::Gpu(format!("map callback: {e}")))?
            .map_err(|e| DiarizationError::Gpu(format!("map: {e}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| DiarizationError::Gpu(format!("mapped range: {e}")))?
            .to_vec();
        staging.unmap();
        let mut data = data;
        data.truncate(bytes as usize);
        Ok(data)
    }

    pub fn readback_f32(&self, buf: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let bytes = self.readback(buf, (n * 4) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    fn pipeline(&self, label: &str, wgsl: &str, entry: &str) -> Result<wgpu::ComputePipeline> {
        if std::env::var_os("DUMP_WGSL").is_some() {
            eprintln!("===== {label} =====\n{wgsl}");
        }
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(guard.pop()) {
            return Err(DiarizationError::Gpu(format!("pipeline {label}: {e}")));
        }
        Ok(pipe)
    }
}

/// A small ring of staging buffers kept mapped and reused for host-to-device copies.
///
/// Allocating a mapped buffer per call is *slower* than the copy itself on this
/// driver, so buffers are created once and re-mapped between uses. The ring matters
/// because two copies recorded into the **same** command encoder would otherwise
/// share one staging buffer, and both would see whatever was written last — the
/// embedder uploads one mel chunk per pass. Re-mapping a slot is only safe because
/// every window ends in a `device.poll`, so earlier copies have retired.
pub struct Uploader {
    slots: Vec<(wgpu::Buffer, bool)>,
    bytes: u64,
    next: usize,
}

impl Uploader {
    pub fn new(gpu: &Gpu, bytes: u64, depth: usize) -> Self {
        let bytes = bytes.max(16);
        let slots = (0..depth.max(1))
            .map(|i| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("uploader{i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                (b, false)
            })
            .collect();
        Self { slots, bytes, next: 0 }
    }

    /// `dst <- data`, recorded into `enc` so it is ordered before that encoder's
    /// compute pass.
    pub fn copy(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        data: &[f32],
    ) -> Result<()> {
        let bytes = (data.len() * 4) as u64;
        assert!(bytes <= self.bytes, "upload {bytes} > staging {}", self.bytes);
        let i = self.next % self.slots.len();
        self.next = self.next.wrapping_add(1);
        if !self.slots[i].1 {
            let (tx, rx) = std::sync::mpsc::channel();
            let buf = self.slots[i].0.clone();
            buf.slice(..).map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv()
                .map_err(|e| DiarizationError::Gpu(format!("map callback: {e}")))?
                .map_err(|e| DiarizationError::Gpu(format!("map: {e}")))?;
            self.slots[i].1 = true;
        }
        {
            let mut view = self.slots[i]
                .0
                .slice(..bytes)
                .get_mapped_range_mut()
                .map_err(|e| DiarizationError::Gpu(format!("mapped range: {e}")))?;
            view.copy_from_slice(bytemuck::cast_slice(data));
        }
        self.slots[i].0.unmap();
        self.slots[i].1 = false;
        enc.copy_buffer_to_buffer(&self.slots[i].0, 0, dst, 0, bytes);
        Ok(())
    }
}

/// A [`BindingResource`](wgpu::BindingResource) covering exactly `s`, for binding a
/// sub-range of a buffer (the fused q/k/v projections live side by side).
fn slice_binding<'a>(s: &'a wgpu::BufferSlice<'a>) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: s.buffer(),
        offset: s.offset(),
        size: std::num::NonZeroU64::new(s.size()),
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// Uniform for the strided GEMM: `C[m,n] = A[m,k] @ W[n,k]^T` with every operand
/// carrying its own row stride, column stride and base. `*_zs` is added to the
/// base scaled by `workgroup_id.z`, so one dispatch can cover a whole batch — the
/// attention passes put the head index there instead of looping 8 times.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct G16 {
    m: u32,
    n: u32,
    k: u32,
    a_rs: u32,
    a_base: u32,
    a_cs: u32,
    a_zs: u32,
    w_rs: u32,
    w_base: u32,
    w_cs: u32,
    w_zs: u32,
    c_rs: u32,
    c_base: u32,
    c_zs: u32,
    flags: u32,
    _pad: u32,
}

/// Argument bundle for [`Kernels::gemm_strided`].
#[derive(Clone, Copy)]
pub struct StridedGemm {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_rs: u32,
    pub a_base: u32,
    pub a_cs: u32,
    pub a_zs: u32,
    pub w_rs: u32,
    pub w_base: u32,
    pub w_cs: u32,
    pub w_zs: u32,
    pub c_rs: u32,
    pub c_base: u32,
    pub c_zs: u32,
    pub gx: u32,
    pub gy: u32,
    pub gz: u32,
}

pub struct Pipes {
    pub gemm: wgpu::ComputePipeline,
    /// The 64-row-row-tile instantiation, see `gemm_wave_bm`.
    pub gemm_m64: wgpu::ComputePipeline,
    pub gemm_n64: wgpu::ComputePipeline,
    pub gemm_strided: wgpu::ComputePipeline,
    pub ln: wgpu::ComputePipeline,
    pub unary: wgpu::ComputePipeline,
    pub add: wgpu::ComputePipeline,
    pub rope: wgpu::ComputePipeline,
    pub attn: wgpu::ComputePipeline,
    pub attn_sm: wgpu::ComputePipeline,
    pub conv: wgpu::ComputePipeline,
    pub col3: wgpu::ComputePipeline,
}

impl Pipes {
    fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            gemm: gpu.pipeline("gemm", &gemm_wgsl(), "gemm")?,
            // Same kernel at 64x128, selected per dispatch by `gemm_wave_bm`.
            gemm_m64: gpu.pipeline("gemm_m64", &gemm_wgsl_wide(), "gemm")?,
            // PV still runs the old 8x8 kernel: it reads `V` transposed, which the
            // 16x8 staging path does not cover, and its `k` is `seq`, which needs a
            // per-element k guard. It is ~1/6 of the tower, so the linear layers
            // went first.
            gemm_n64: gpu.pipeline("gemm_n64", &gemm_wgsl_t(8, 4, true, 0, false), "gemm")?,
            // The strided instantiation has its own operand/result strides, so
            // `DUMP_WGSL` (which prints the linear one) does not cover it.
            gemm_strided: {
                let src = gemm_strided_wgsl();
                if std::env::var_os("DUMP_STRIDED").is_some() {
                    eprintln!("===== gemm_strided =====\n{src}");
                }
                gpu.pipeline("gemm_strided", &src, "gemm")?
            },
            ln: gpu.pipeline("ln", LN_WGSL, "ln")?,
            unary: gpu.pipeline("unary", UNARY_WGSL, "unary")?,
            add: gpu.pipeline("add", ADD_WGSL, "add")?,
            rope: gpu.pipeline("rope", ROPE_WGSL, "rope")?,
            attn: gpu.pipeline("attn", ATTN_WGSL, "attn")?,
            attn_sm: gpu.pipeline("attn_sm", ATTN_SM_WGSL, "softmax_rows")?,
            conv: gpu.pipeline("conv", CONV_WGSL, "conv")?,
            col3: gpu.pipeline("col3", COL3_WGSL, "col3")?,
        })
    }
}

/// Scratch + pipelines + a ring of uniform buffers so several dispatches can
/// share one command encoder without stomping each other's uniforms.
pub struct Kernels {
    gpu: Gpu,
    pipes: Pipes,
    uniforms: Vec<wgpu::Buffer>,
    uni_used: usize,
    uniforms8: Vec<wgpu::Buffer>,
    uni_used8: usize,
    uniforms16: Vec<wgpu::Buffer>,
    uni_used16: usize,
    dummy_bias: wgpu::Buffer,
}

impl Kernels {
    pub fn new() -> Result<Self> {
        let gpu = Gpu::new()?;
        let pipes = Pipes::new(&gpu)?;
        let dummy_bias = gpu.storage("dummy_bias", 2048 * 4);
        gpu.upload_f32(&dummy_bias, &[0.0f32; 2048]);
        Ok(Self {
            gpu,
            pipes,
            uniforms: Vec::new(),
            uniforms8: Vec::new(),
            uni_used: 0,
            uni_used8: 0,
            uniforms16: Vec::new(),
            uni_used16: 0,
            dummy_bias,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn dummy_bias(&self) -> &wgpu::Buffer {
        &self.dummy_bias
    }

    pub fn make_uniform(&self, label: &str) -> wgpu::Buffer {
        self.gpu.uniform(label)
    }

    /// 32 bytes, for the RoPE uniform.
    pub fn make_uniform32(&self, label: &str) -> wgpu::Buffer {
        self.gpu.uniform_sized(label, 32)
    }

    /// 64-byte uniform, wide enough for [`StridedGemm`]'s shader struct.
    pub fn make_uniform64(&self, label: &str) -> wgpu::Buffer {
        self.gpu.uniform_sized(label, 64)
    }

    pub fn describe(&self) -> String {
        self.gpu.describe()
    }

    pub fn storage(&self, label: &str, n_f32: usize) -> wgpu::Buffer {
        self.gpu.storage(label, (n_f32 * 4) as u64)
    }

    pub fn upload_new(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let b = self.storage(label, data.len().max(4));
        self.gpu.upload_f32(&b, data);
        b
    }

    pub fn begin(&mut self) {
        self.uni_used = 0;
        self.uni_used8 = 0;
        self.uni_used16 = 0;
    }

    /// 32-byte uniform, for the RoPE kernel's column offset + row stride.
    pub fn uniform8(&mut self, v: [u32; 8]) -> wgpu::Buffer {
        if self.uni_used8 >= self.uniforms8.len() {
            let n = self.uniforms8.len();
            self.uniforms8.push(self.gpu.uniform_sized(&format!("u8_{n}"), 32));
        }
        self.gpu.queue.write_buffer(&self.uniforms8[self.uni_used8], 0, bytemuck::cast_slice(&v));
        let b = self.uniforms8[self.uni_used8].clone();
        self.uni_used8 += 1;
        b
    }

    fn uniform(&mut self, u: U4) -> wgpu::Buffer {
        if self.uni_used >= self.uniforms.len() {
            let n = self.uniforms.len();
            self.uniforms.push(self.gpu.uniform(&format!("u{n}")));
        }
        self.gpu
            .queue
            .write_buffer(&self.uniforms[self.uni_used], 0, bytemuck::bytes_of(&u));
        let i = self.uni_used;
        self.uni_used += 1;
        self.uniforms[i].clone()
    }

    fn uniform_g16(&mut self, u: G16) -> wgpu::Buffer {
        if self.uni_used16 >= self.uniforms16.len() {
            let n = self.uniforms16.len();
            self.uniforms16.push(self.gpu.storage(&format!("g{n}"), 12));
        }
        self.gpu
            .queue
            .write_buffer(&self.uniforms16[self.uni_used16], 0, bytemuck::bytes_of(&u));
        let i = self.uni_used16;
        self.uni_used16 += 1;
        self.uniforms16[i].clone()
    }

    pub fn pipes(&self) -> &Pipes {
        &self.pipes
    }

    /// RoPE's uniform: sequence, heads, head_dim, the column block being rotated
    /// and the packed buffer's row stride. Five values, so it cannot ride the
    /// four-word `U4` path the other kernels share.
    pub fn write_uniform6(&self, buf: &wgpu::Buffer, seq: u32, heads: u32, hd: u32, col: u32, row: u32) {
        self.gpu
            .queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&[seq, heads, hd, col, row, 0u32, 0u32, 0u32]));
    }

    pub fn write_uniform(&self, buf: &wgpu::Buffer, a: u32, b: u32, c: u32, d: u32) {
        self.gpu
            .queue
            .write_buffer(buf, 0, bytemuck::bytes_of(&U4 { a, b, c, d }));
    }

    /// Rewrite a 48-byte strided-GEMM uniform. Safe to call once per submit: the
    /// attention shapes are the same for every layer, so one buffer each is enough
    /// and never aliased inside a command encoder.
    pub fn write_uniform_g16(&self, buf: &wgpu::Buffer, p: &StridedGemm) {
        self.gpu.queue.write_buffer(
            buf,
            0,
            bytemuck::bytes_of(&G16 {
                m: p.m,
                n: p.n,
                k: p.k,
                a_rs: p.a_rs,
                a_base: p.a_base,
                a_cs: p.a_cs,
                a_zs: p.a_zs,
                w_rs: p.w_rs,
                w_base: p.w_base,
                w_cs: p.w_cs,
                w_zs: p.w_zs,
                c_rs: p.c_rs,
                c_base: p.c_base,
                c_zs: p.c_zs,
                flags: 0,
                _pad: 0,
            }),
        );
    }

    pub fn bind5(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
            ],
        })
    }

    pub fn bind4(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            ],
        })
    }

    pub fn bind3(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uni.as_entire_binding() },
            ],
        })
    }

    /// [`bind5`](Self::bind5) with `b0` and `b1` given as sub-ranges of buffers.
    pub fn bind5_slice<'x>(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::BufferSlice<'x>,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slice_binding(b1),
                },
                wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
            ],
        })
    }

    /// [`bind5`](Self::bind5) with both operands given as sub-ranges (QK^T).
    pub fn bind5_view<'x>(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::BufferSlice<'x>,
        b1: &wgpu::BufferSlice<'x>,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: slice_binding(b0) },
                wgpu::BindGroupEntry { binding: 1, resource: slice_binding(b1) },
                wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
            ],
        })
    }

    /// [`bind4`](Self::bind4) with `b0` given as a sub-range of a buffer.
    pub fn bind4_slice<'x>(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::BufferSlice<'x>,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slice_binding(b0),
                },
                wgpu::BindGroupEntry { binding: 1, resource: b1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            ],
        })
    }

    pub fn bind2(
        &self,
        pipe: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        uni: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uni.as_entire_binding() },
            ],
        })
    }

    /// `C[M,N] = A[M,K] @ W[N,K]^T (+ bias[N])`.
    pub fn gemm(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let has_bias = bias.is_some() as u32;
        let bias = bias.unwrap_or(&self.dummy_bias).clone();
        let uni = self.uniform(U4 { a: m, b: n, c: k, d: has_bias });
        let bg = self.bind5(&self.pipes.gemm, a, w, &bias, c, &uni);
        pass.set_pipeline(&self.pipes.gemm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(GEMM_BN), m.div_ceil(gemm_bm()), 1);
    }

    /// QK^T / PV for one attention step. `narrow` picks the 128-wide tile (QK,
    /// which has `seq` output columns) over the 64-wide one (PV, which has
    /// `head_dim`).
    pub fn gemm_strided(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        narrow: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        p: StridedGemm,
    ) {
        let pipe = if narrow { self.pipes.gemm_n64.clone() } else { self.pipes.gemm_strided.clone() };
        let uni = self.uniform_g16(G16 {
            m: p.m,
            n: p.n,
            k: p.k,
            a_rs: p.a_rs,
            a_base: p.a_base,
            a_cs: p.a_cs,
            a_zs: p.a_zs,
            w_rs: p.w_rs,
            w_base: p.w_base,
            w_cs: p.w_cs,
            w_zs: p.w_zs,
            c_rs: p.c_rs,
            c_base: p.c_base,
            c_zs: p.c_zs,
            flags: 0,
            _pad: 0,
        });
        let bg = self.bind5(&pipe, a, w, &self.dummy_bias, c, &uni);
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(p.gx, p.gy, p.gz);
    }

    /// In-place scaled softmax over the `valid` leading columns of each
    /// `[rows, cols]` row. Columns `valid..cols` are zeroed, which is how the
    /// attention mask keeps padded keys out of the PV product.
    pub fn attn_softmax(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        s: &wgpu::Buffer,
        rows: u32,
        cols: u32,
        valid: u32,
        scale: f32,
    ) {
        let uni = self.uniform(U4 { a: rows, b: cols, c: valid, d: scale.to_bits() });
        let bg = self.bind2(&self.pipes.attn_sm, s, &uni);
        pass.set_pipeline(&self.pipes.attn_sm);
        pass.set_bind_group(0, &bg, &[]);
        // 8 rows per workgroup, see `ATTN_SM_WGSL`
        pass.dispatch_workgroups(rows.div_ceil(8), 1, 1);
    }

    pub fn layer_norm(        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: u32,
        cols: u32,
    ) {
        let uni = self.uniform(U4 {
            a: cols,
            b: 1e-5f32.to_bits(),
            c: 0,
            d: 0,
        });
        let bg = self.bind5(&self.pipes.ln, x, w, b, y, &uni);
        pass.set_pipeline(&self.pipes.ln);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(rows, 1, 1);
    }

    /// `mode=0` GELU (erf), `mode=1` ReLU. In-place.
    pub fn unary(&mut self, pass: &mut wgpu::ComputePass<'_>, x: &wgpu::Buffer, n: u32, mode: u32) {
        let uni = self.uniform(U4 { a: n, b: mode, c: 0, d: 0 });
        let bg = self.bind2(&self.pipes.unary, x, &uni);
        pass.set_pipeline(&self.pipes.unary);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(WG_LN), 1, 1);
    }

    /// In-place `c[i] += b[i]`. `c` is read-write, `b` is read-only — wgpu
    /// forbids binding the same buffer as both in one dispatch.
    pub fn add(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        c: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: u32,
    ) {
        let uni = self.uniform(U4 { a: n, b: 0, c: 0, d: 0 });
        let bg = self.bind_add(c, b, &uni);
        pass.set_pipeline(&self.pipes.add);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(WG_LN), 1, 1);
    }

    pub fn bind_add(&self, c: &wgpu::Buffer, b: &wgpu::Buffer, uni: &wgpu::Buffer) -> wgpu::BindGroup {
        let layout = self.pipes.add.get_bind_group_layout(0);
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uni.as_entire_binding() },
            ],
        })
    }

    pub fn rope(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        cos: &wgpu::Buffer,
        sin: &wgpu::Buffer,
        seq: u32,
        heads: u32,
        hd: u32,
        col: u32,
        row_stride: u32,
    ) {
        let uni = self.uniform8([seq, heads, hd, col, row_stride, 0, 0, 0]);
        let bg = self.bind4(&self.pipes.rope, x, cos, sin, &uni);
        pass.set_pipeline(&self.pipes.rope);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((seq * heads * (hd / 2)).div_ceil(64), 1, 1);
    }

    /// RoPE over a sub-range of `qkv` (q and k sit side by side in one buffer).
    pub fn rope_slice(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::BufferSlice<'_>,
        cos: &wgpu::Buffer,
        sin: &wgpu::Buffer,
        seq: u32,
        heads: u32,
        hd: u32,
        col: u32,
        row_stride: u32,
    ) {
        let uni = self.uniform8([seq, heads, hd, col, row_stride, 0, 0, 0]);
        let layout = self.pipes.rope.get_bind_group_layout(0);
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: slice_binding(x) },
                wgpu::BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipes.rope);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((seq * heads * (hd / 2)).div_ceil(64), 1, 1);
    }

    pub fn attn(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        out: &wgpu::Buffer,
        seq: u32,
        heads: u32,
        hd: u32,
        valid: u32,
    ) {
        debug_assert!(seq <= MAX_ATTN_SEQ && hd == ATTN_WG);
        let uni = self.uniform(U4 { a: seq, b: heads, c: hd, d: valid });
        let bg = self.bind5(&self.pipes.attn, q, k, v, out, &uni);
        pass.set_pipeline(&self.pipes.attn);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(seq, heads, 1);
    }

    pub fn conv1d_k3(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        frames: u32,
        hh: u32,
        out_c: u32,
    ) {
        let uni = self.uniform(U4 { a: frames, b: hh, c: out_c, d: 0 });
        let bg = self.bind5(&self.pipes.conv, x, w, b, y, &uni);
        pass.set_pipeline(&self.pipes.conv);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(frames.div_ceil(8), out_c.div_ceil(8), 1);
    }

    /// `[frames, hh] -> [frames, hh*3]` im2col for the sub-pixel conv.
    pub fn col3(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        a: &wgpu::Buffer,
        frames: u32,
        hh: u32,
    ) {
        let uni = self.uniform(U4 { a: frames, b: hh, c: 0, d: 0 });
        let bg = self.bind3(&self.pipes.col3, x, a, &uni);
        pass.set_pipeline(&self.pipes.col3);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((frames * hh * 3).div_ceil(256), 1, 1);
    }

    pub fn submit(&self, enc: wgpu::CommandEncoder) {
        self.gpu.queue.submit([enc.finish()]);
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.gpu.device.create_command_encoder(&Default::default())
    }
}

/// fp32 port of the qwen3-asr prefill GEMM: 128×128 tile, 8×8 registers,
/// BK=16, shared tile stored as k-major `vec4` chunks (four rows at one k).
/// `C[m,n] = A[m,k] @ W[n,k]^T (+ bias)`.
///
/// The staging side loads *four rows at one k* with scalar loads and packs them
/// into one `vec4` store. That looks badly uncoalesced — a warp instruction
/// touches 16 distinct 32-byte sectors — and it was worth trying to fix: a
/// replacement that loads 4(row) × 4(k) blocks with coalesced `vec4` loads along
/// k and transposes them in registers measured **0.71 TFLOP/s against this
/// kernel's 0.85**, and 1.65 s against 1.06 s on the real offline tower. Widening
/// the register tile to 16×8 (0.75 B/FMA of shared traffic instead of 1.0, so
/// shared bandwidth is no longer the limit) was also slower, 0.67. Occupancy wins
/// here: 8×8 at 64 accumulators is the only shape measured that holds enough
/// warps per SM to cover the barrier and global-load latency. Do not re-try the
/// coalesced-staging rewrite without new evidence.
///
/// The one staging/read fix that did land: the B tile is consumed by `tx`, so
/// group `v` of thread `tx` takes shared slot `v * TX + tx` rather than
/// `tx * VGN + v`. The contiguous grouping put lanes 0 and 4 of every 8-lane phase
/// on the same four banks.
///
/// The tile shape is a parameter so the attention pipeline can ask for the same
/// kernel with a 128×**64** output tile (`PV` only ever has 64 columns per head)
/// without a second, differently-shaped shader. `strided` swaps the assumed
/// `row_stride == k` layout for explicit `a_rs` / `a_base` / `a_cs` / `w_rs` /
/// `w_base` / `w_cs` / `c_rs` / `c_base` so Q and K can be read straight out of
/// the `[seq, heads*hd]` activation layout instead of being repacked first.
///
/// The non-strided instantiation emits the exact index expressions the linear
/// layers were validated with; the shared-memory layout is the same k-major
/// `vec4` arrangement, so its results are bit-identical.
fn gemm_wgsl_t(tm: usize, tn: usize, strided: bool, abl: u8, lds_pipe: bool) -> String {
    // `half_split` is on by default for every 8x8 instantiation. The linear one is
    // covered by GEMM_CHECK (eight shapes) and the strided attention pair by
    // ATTN_GEMM_CHECK (six seq values each); both are worth re-running after any
    // change here. `gemm_n64` is 8x4, so `bm == 128 && bn == 128` fails and it keeps
    // the packed read automatically.
    // BK is only tunable for the linear instantiation *by default*: the strided
    // ones have `k == head_dim` (QK) or `k == seq` (PV), neither of which is a
    // multiple of BK in general, and the kernel carries no per-element k guard.
    //
    // `GEMM_BK_STRIDED` opts in anyway, and it is safe for both after all: the
    // staging load is guarded (`if (... && ka < g.k) la = 0.0`) and the store
    // copies those zeros into the shared tile, so a k-tile that runs past the end
    // of K multiplies by zero and contributes nothing. What it costs is the wasted
    // FFMA of the overhang. The motivation is that QK's `k == head_dim == 64`
    // gives only 4 k-tiles at BK = 16, so the two `workgroupBarrier`s per k-tile
    // are amortised over 8x less work than in the linear GEMM (`k == 512`, 32
    // k-tiles) -- which is the whole of the measured QK-vs-PV gap
    // (0.449 vs 0.655 TFLOP/s, PV having 24 k-tiles to QK's 4).
    let bk = if strided {
        std::env::var("GEMM_BK_STRIDED").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
    } else {
        std::env::var("GEMM_BK").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
    };
    gemm_wgsl_bk(tm, tn, strided, abl, lds_pipe, bk, std::env::var_os("GEMM_NOHALF").is_none(), std::env::var_os("GEMM_DB").is_some(), std::env::var_os("GEMM_PIPE").is_some(), std::env::var_os("GEMM_NOPADCOAL").is_none())
}

/// `half_split` moves a thread's 8x8 register tile to the two 4-wide halves of the
/// 128x128 output tile (rows `4ty..+3` and `64+4ty..+3`) instead of one 8-wide run.
/// Both the packed and the half-split form read one `vec4` per operand per k-step
/// and both satisfy the staging invariant "shared slot `c` holds rows `4c..4c+3`",
/// so **the staging path is identical** -- only the read-side slot and the epilogue
/// row/column change. The reason it pays: packed, a warp's first eight lanes read
/// slots `2tx`, i.e. float4 indices 0,2,4,...,14, which is bank groups
/// 0,8,16,24 twice over -> 4-way conflict. Half-split reads slots `tx` and
/// `tx+16`, so those same eight lanes cover float4 indices 0..7 = all 32 banks.
/// Measured on sm_61 (see HANDOFF.md): +11% alone, +26% together with BK=8 and a
/// second shared buffer. Requires `tm == tn == 8` (BM = BN = 128).
///
/// `padcoal` (`GEMM_NOPADCOAL` turns it off) is default **on**: the WGSL port of
/// `v25` in sgemm_hs.cu -- a padded plane stride plus the one staging lane map
/// that padding makes possible. See the long comment on it below for the bank
/// algebra and for the premise round 16's exclusion proof was missing. Bit-exact
/// with the old path on all eight `GEMM_CHECK` shapes, and **+6..8% TFLOP/s across
/// the whole seq sweep** (six interleaved rounds, order swapped, no overlap at any
/// seq): 0.565 / 1.058 / 1.087 / 1.080 / 1.105 / 1.002 -> 0.613 / 1.117 / 1.163 /
/// 1.160 / 1.175 / 1.073 TFLOP/s for seq 64 / 128 / 256 / 380 / 512 / 684.
fn gemm_wgsl_bk(tm: usize, tn: usize, strided: bool, abl: u8, lds_pipe: bool, bk: usize, half_split: bool, dbuf: bool, pipe: bool, padcoal: bool) -> String {
    const TX: usize = 16;
    const TY: usize = 16;
    // BK is pinned at 16: the staging lane map is `kid = (pos/128)*(BK/2) + (pos%128)/16`
    // with two float4 stores per thread, which tiles exactly 16 k-planes x 32 slots.
    // BK = 8 would make the second half address planes 8..11, i.e. past the tile.
    let BK: usize = bk;
    let vgm = tm / 4;
    let vgn = tn / 4;
    let bm = TX * tm;
    let bn = TX * tn;
    let aq = bm / 4;
    let bq = bn / 4;
    // Half-split needs `BN == 128` to have two 64-wide halves, and `TN == 8` to
    // fill one. `TM` may be 8 (128x128) or 4 (the `GEMM_BM=64` probe, where the
    // row side only ever emits the `i < 4` branch), and `VG_M` then collapses to a
    // single `vec4` read at slot `ty` -- which is exactly slot 0..15 of the 16-slot
    // A tile, so the "slot c holds rows 4c..4c+3" invariant still holds.
    let half = half_split && tn == 8 && bn == 128 && (tm == 8 || tm == 4);
    // `padcoal` (`GEMM_PADCOAL`) is the WGSL port of kernel `v25` in sgemm_hs.cu,
    // and it is the port of the *corrected* version of the write/read mutual
    // exclusion, not of the one recorded in round 16.
    //
    // Round 16 proved that for a tile whose plane stride is exactly 32 the two
    // sides of the shared tile cannot both be conflict-free: the *store* wants
    // `f mod 8` to follow the lane, the *read* wants it to follow the slot, and
    // with a stride of 32 (== 0 mod 8) the two are the same quantity. That proof
    // is correct **as stated** -- and that is exactly the hole: it assumed the
    // stride is 32. Pad the plane stride to 33 and the two separate again:
    //
    //   store  f = plane*33 + slot, lanes 0..7 of a warp have eight distinct
    //          planes at one slot, so f mod 8 = (plane + slot) = 0..7
    //   read   f = q*33 + slot,  half-split makes the A side a broadcast and
    //          gives the B side slot = tx = 0..7, so f mod 8 = (q + tx) = 0..7
    //
    // Both are then conflict-free at the same time, which is what unlocks the
    // coalesced staging map (`GEMM_COAL`, measured -10% in WGSL because its
    // stores went 8 ways). With the pad they do not. Measured on sm_61 as
    // `v25`: bit-identical to `v0` on all four shapes, 48.1% -> 58.6% of cuBLAS
    // (+21.7%), stable over three rounds, and nothing in it changes the
    // instruction count -- v23 had already attributed 10.5 of those points to
    // halving the sector count alone.
    //
    // Gated to the 128x128 non-strided shape, which is the only one whose tile
    // has 32 slots per plane: the map assigns 4 slots per warp, so it cannot
    // tile a 64-row or 64-column tile (`aq`/`bq` of 16).
    let padcoal = padcoal && !strided && bm == 128 && bn == 128 && half;
    let astride = if padcoal { aq + 1 } else { aq };
    let bstride = if padcoal { bq + 1 } else { bq };
    // `dbuf` gives the shared tile two buffers. Writing buffer `buf^1` while every
    // thread is still reading `buf` is safe because the single barrier at the end of
    // the previous iteration already separated the reads of `buf^1` from these
    // writes, so one barrier per k-tile suffices instead of two.
    let nak = BK * astride;
    let nbk = BK * bstride;
    let nbuf = if dbuf { 2usize } else { 1usize };
    let nakN = nak * nbuf;
    let nbkN = nbk * nbuf;
    // Staging lane map. 256 threads each stage `sgroups` 4-row groups, so a k-plane
    // is covered by `TPI = 256/BK` threads and each of them owns `Q/TPI` slots.
    //   BK = 16: TPI 16, 2 groups -> plane = pos/16, slot = (pos%16)*2 + v   (was
    //            `(pos>>7)*(BK/2) + (pos&127)/16` and `((pos&127)%16)*VGM`; the two
    //            forms are identical for BK = 16, so this is not a behaviour change)
    //   BK =  8: TPI 32, 1 group  -> plane = pos/32, slot = pos%32
    // The old hardcoded form only ever tiled 16 planes x 32 slots; at BK = 8 the
    // second half addressed planes 8..11, past the end of the tile.
    //
    // `padcoal` uses the third map, the one `v25` needs: the plane is the *lane*
    // (`pos % 8`) and the slot is the *warp* (`4*(pos/32) + (pos/8 % 4)`), with
    // each thread's `sga = BK/8` groups walking eight planes apart rather than
    // one slot apart. `pos/8 % 4 == (pos % 32) / 8`, so this is the same thing
    // `v25` writes as `s0 = 4*w + (p >> 3)` with `w = pos/32`. One warp covers
    // four slots x eight consecutive k, so a staging load touches four sectors
    // (4 rows x 32 B) instead of 32 and every byte fetched is used -- the
    // coalescing win that `GEMM_COAL` could not keep, because without the pad
    // these stores go 8 ways. Bijective on 16 planes x 32 slots: for a fixed
    // plane the 32 `pos` values with `pos % 8 == plane % 8` and `v == plane / 8`
    // sweep `j = pos / 8` over 0..31 and hit every slot exactly once.
    //
    // `coalesce` flips which axis the 32 lanes of a warp spread over. The original
    // map gives one warp 16 *rows* at 2 adjacent k, so every scalar load retires
    // 16 sectors carrying 8 B each -- a quarter of what the sector moved. But the
    // k-planes of a tile are 16 *consecutive* floats of one row in global memory, so
    // swapping the roles (plane = pos % TPI, slot = pos / TPI, v groups strided by
    // BK) puts 16 lanes on 16 consecutive k of two rows: two full sectors per load.
    //
    // It is correct, bit-exact on all eight GEMM_CHECK shapes, and **+6% on sm_61**,
    // but it is **-10% under WGSL/Vulkan** (three interleaved rounds, every seq in the
    // sweep), so it ships off. The reason is a structural conflict, not tuning: the
    // shared tile is a `plane x slot` matrix and a warp can only be contiguous along
    // one of those axes. The *store* side wants 16 planes x 2 slots contiguous, i.e.
    // slot-major `A4[slot*BK+plane]`; the *read* side wants 32 slots contiguous, i.e.
    // plane-major `A4[plane*AST+slot]`. The swapped map is slot-major, so its stores go
    // 8 ways (vec4 indices 0,32,64,...,224, 512 B apart) and the global win is smaller
    // than the store loss. Keeping the store bank-conflict-free by transposing the
    // tile instead (slot-major) was measured on sm_61 as the kernel `v15` in
    // sgemm_hs.cu: -60%, because the read side then walks 256 B per step. The two
    // requirements are mutually exclusive, so 8B/32B = 25% is the ceiling for this
    // layout and the existing map already sits on it. **This paragraph is what
    // `padcoal` overturns** -- not the mutual exclusion itself, which still holds
    // for an unpadded stride, but the conclusion that the layout has to stay
    // plane-major with a 32-stride. Re-deriving this costs a day; don't re-try
    // `GEMM_COAL` on its own, and don't re-derive the exclusion without checking
    // whether the stride is still 32.
    // `tpi` is the threads that cover one k-plane, and it must not exceed the
    // number of slots in that plane or the lane map cannot tile the tile. BK=8
    // makes `tpi = 32`, which is fine for the 128-wide instantiations
    // (`aq = bq = 32`) but **not** for `gemm_n64` (128x64, `bq = 16`): the
    // assert below fires at pipeline creation. So `GEMM_BK=8` does not currently
    // run at all -- whatever BK=8 numbers exist in the history predate that
    // conflict and cannot be reproduced from this tree.
    let tpi = 256 / BK;
    let coalesce = !padcoal
        && !strided
        && std::env::var_os("GEMM_COAL").is_some()
        && aq % BK == 0
        && bq % BK == 0;
    let (sga, sgb) = if padcoal {
        (BK / 8, BK / 8)
    } else if coalesce {
        (aq / BK, bq / BK)
    } else {
        (aq / tpi, bq / tpi)
    };
    assert!(
        if padcoal {
            BK % 8 == 0 && 256 * sga == BK * aq && 256 * sgb == BK * bq
        } else if coalesce {
            BK * sga == aq && BK * sgb == bq
        } else {
            tpi * sga == aq && tpi * sgb == bq
        },
        "staging lane map does not tile the shared tile"
    );
    // A element: row `r`, k `k`. Non-strided is `r * g.k + k`.
    let a_idx = |r: &str, k: &str| -> String {
        if strided {
            format!("({r}) * g.a_rs + abase + ({k}) * g.a_cs")
        } else {
            format!("({r}) * g.k + ({k})")
        }
    };
    let w_idx = |r: &str, k: &str| -> String {
        if strided {
            format!("({r}) * g.w_rs + wbase + ({k}) * g.w_cs")
        } else {
            format!("({r}) * g.k + ({k})")
        }
    };
    let c_idx = |r: &str, n: &str| -> String {
        if strided {
            format!("({r}) * g.c_rs + cbase + ({n})")
        } else {
            format!("({r}) * g.n + ({n})")
        }
    };

    let mut s = String::new();
    if strided {
        s.push_str(
            "struct G { m: u32, n: u32, k: u32, a_rs: u32, a_base: u32, a_cs: u32, a_zs: u32,\
             w_rs: u32, w_base: u32, w_cs: u32, w_zs: u32, c_rs: u32, c_base: u32, c_zs: u32,\
             flags: u32, _pad: u32 }\n",
        );
    } else {
        s.push_str("struct G { m: u32, n: u32, k: u32, has_bias: u32 }\n");
    }
    s.push_str(
        "@group(0) @binding(0) var<storage, read> A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> Bias: array<f32>;\n\
         @group(0) @binding(3) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(4) var<uniform> g: G;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {BK}u;\nconst TPI: u32 = {tpi}u;\n\
         const TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const VGM: u32 = {vgm}u;\nconst VGN: u32 = {vgn}u;\n\
         const AST: u32 = {astride}u;\nconst BST: u32 = {bstride}u;\n\
         var<workgroup> A4: array<vec4<f32>, {nakN}>;\n\
         var<workgroup> B4: array<vec4<f32>, {nbkN}>;\n"
    ));
    let zbases = if strided {
        "let abase = g.a_base + wgid.z * g.a_zs;\nlet wbase = g.w_base + wgid.z * g.w_zs;\nlet cbase = g.c_base + wgid.z * g.c_zs;\n"
    } else {
        ""
    };
    s.push_str(&format!(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
         let tx = lid.x;\nlet ty = lid.y;\n\
         let m0 = wgid.y * BM;\nlet n0 = wgid.x * BN;\n{zbases}",
    ));
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    // Shared-memory slot `kid * Q + g` always holds rows `4g .. 4g+3` of the tile at
    // k = `kid`, so the read side can address it directly. The *write* side, however,
    // has to give each of the 32 lanes of a warp 32 consecutive slots or the stores
    // collide across banks. Threads are therefore numbered `pos` and split as
    // `kid = (pos / 128) * (BK/2) + (pos % 128) / TX`, `g = (pos % 128) % TX * V`,
    // which needs `TX == TX` in both cases and `BK/2 * Q == TX * TY / 2 * V`.
    // B-side slot assignment. For the linear instantiation the B tile is consumed
    // by `tx`, so group `v` of thread `tx` takes slot `v * TX + tx` rather than
    // `tx * VGN + v`. A warp's first eight lanes then hold eight *distinct* slots
    // and the `vec4` read covers all 32 banks; the contiguous grouping put lanes
    // 0 and 4 on the same four banks. The strided variants keep the packed layout,
    // which their own `vgn` accounting depends on.
    // A-side slot assignment. The invariant tying the three expressions together
    // is only "slot `c` holds rows `4c .. 4c+3`"; staging and reading are free to
    // reach that slot by different routes, and they have to, because the staging
    // lane map pins the k-plane to `ty` (`akid == ty`) while the read is indexed by
    // `ty` and the staging sweep is indexed by `tx`.
    //
    //   staging slot  v*TX + tx  — for a fixed plane the 16 `tx` values must sweep
    //                               all 32 slots or most of the tile is never
    //                               written. Also bank-conflict-free: a warp's
    //                               first eight lanes hit eight consecutive slots.
    //   read slot     v*TY + ty  — a warp's first eight lanes share one `ty`, so
    //                               the read is a broadcast.
    // Packed, staging was `tx*VGM + v` (stride two: four banks hit twice, four
    // never touched, so every A store ran at half rate) and the read was
    // `ty*VGM + v`.
    // `padcoal` inverts what `v` means: the group index selects the k-*round*
    // (eight planes apart), not a neighbouring slot, so the slot is the same for
    // every round and the row it stages is fixed by `aslot` alone. That is also
    // what makes the A read a broadcast -- `aslot` is a function of the warp and
    // the lane's high three bits, and the read side addresses the slot by `ty`.
    let a_stage_slot = |v: usize| -> String {
        if padcoal {
            "aslot".to_string()
        } else if coalesce {
            format!("aslot + {}u", v * BK)
        } else {
            format!("aslot + {v}u")
        }
    };
    let a_read_slot = |v: usize| -> String {
        if half {
            if v == 0 { "ty".to_string() } else { format!("ty + {}u", bm / 8) }
        } else {
            format!("ty * VGM + {v}u")
        }
    };
    let a_row = |v: usize, e: usize| -> String {
        if padcoal {
            format!("aslot * 4u + {e}u")
        } else if coalesce {
            format!("(aslot + {}u) * 4u + {e}u", v * BK)
        } else {
            format!("(aslot + {v}u) * 4u + {e}u")
        }
    };
    // Output tile position of accumulator `i`/`j`. Must agree with the read-side
    // slot above: packed, `i` is at `ty*tm + i`; half-split, `i < 4` is at
    // `4*ty + i` and `i >= 4` at `64 + 4*ty + (i-4)`.
    let hs_row = |i: usize| -> String {
        if half {
            if i < 4 { format!("ty * 4u + {i}u") } else { format!("{}u + ty * 4u + {}u", bm / 2, i - 4) }
        } else {
            format!("ty * {tm}u + {i}u")
        }
    };
    let hs_col = |j: usize| -> String {
        if half {
            if j < 4 { format!("tx * 4u + {j}u") } else { format!("{}u + tx * 4u + {}u", bn / 2, j - 4) }
        } else {
            format!("tx * {tn}u + {j}u")
        }
    };
    // staging slot: unchanged by `half`, so the "slot c holds rows 4c..4c+3"
    // invariant the staging writes is preserved either way
    let b_stage_slot = |v: usize| -> String {
        if padcoal {
            "bslot".to_string()
        } else if coalesce {
            format!("bslot + {}u", v * BK)
        } else {
            format!("bslot + {v}u")
        }
    };
    let b_read_slot = |v: usize| -> String {
        if half {
            if v == 0 { "tx".to_string() } else { format!("tx + {}u", bn / 8) }
        } else {
            format!("tx * VGN + {v}u")
        }
    };
    let b_row = |v: usize, e: usize| -> String {
        if padcoal {
            format!("bslot * 4u + {e}u")
        } else if coalesce {
            format!("(bslot + {}u) * 4u + {e}u", v * BK)
        } else {
            format!("(bslot + {v}u) * 4u + {e}u")
        }
    };
    let load = |s: &mut String, kbase: &str, decl: bool| {
        if abl == 1 {
            // ablation: no global traffic at all, so the staging cost collapses
            for v in 0..vgm {
                s.push_str(&format!(" if (m0 + (agb + {v}u) * 4u < g.m) {{ la_{v}_0 = 1.0; }}\n"));
            }
            for v in 0..vgn {
                s.push_str(&format!(" if (n0 + (bgb + {v}u) * 4u < g.n) {{ lb_{v}_0 = 1.0; }}\n"));
            }
            return;
        }
        for v in 0..sga {
            for e in 0..4 {
                if decl {
                    s.push_str(&format!(" var la_{v}_{e} = 0.0;\n"));
                }
                // `padcoal` advances eight planes per round, not one slot.
                let ka = if padcoal { format!("({kbase} + akid + {}u)", v * 8) } else { format!("({kbase} + akid)") };
                let arow = a_row(v, e);
                s.push_str(&format!(
                    " if (m0 + {arow} < g.m && {ka} < g.k) {{ la_{v}_{e} = A[{}]; }}\n",
                    a_idx(&format!("(m0 + {arow})"), &ka)
                ));
            }
        }
        for v in 0..sgb {
            for e in 0..4 {
                if decl {
                    s.push_str(&format!(" var lb_{v}_{e} = 0.0;\n"));
                }
                let ka = if padcoal { format!("({kbase} + bkid + {}u)", v * 8) } else { format!("({kbase} + bkid)") };
                let brow = b_row(v, e);
                s.push_str(&format!(
                    " if (n0 + {brow} < g.n && {ka} < g.k) {{ lb_{v}_{e} = W[{}]; }}\n",
                    w_idx(&format!("(n0 + {brow})"), &ka)
                ));
            }
        }
    };
    let store = |s: &mut String, bufe: &str| {
        if abl == 3 { return; }
        // Under `padcoal` the group index is no longer in the slot (the slot is
        // `aslot` for every round) so it has to be in the plane, eight planes
        // apart -- otherwise both rounds of a thread write the same address and
        // the second silently wins. This is `f = (8*c_ + (p&7))*ST + s0` in v25.
        let apl = |v: usize| if padcoal { format!("(akid + {}u) * AST", v * 8) } else { "akid * AST".to_string() };
        let bpl = |v: usize| if padcoal { format!("(bkid + {}u) * BST", v * 8) } else { "bkid * BST".to_string() };
        for v in 0..sga {
            let ra = (0..4).map(|e| format!("la_{v}_{e}")).collect::<Vec<_>>().join(", ");
            s.push_str(&format!(" A4[{bufe} * {nak}u + {} + {slot}] = vec4<f32>({ra});\n", apl(v), slot = a_stage_slot(v)));
        }
        for v in 0..sgb {
            let rb = (0..4).map(|e| format!("lb_{v}_{e}")).collect::<Vec<_>>().join(", ");
            s.push_str(&format!(" B4[{bufe} * {nbk}u + {} + {slot}] = vec4<f32>({rb});\n", bpl(v), slot = b_stage_slot(v)));
        }
    };

    // `sga` already carries the per-thread group count, so the two maps differ only
    // in which of `pos / TPI` and `pos % TPI` selects the plane. Note the coalesced
    // form must *not* also scale by `sga`: its `v` groups are strided by `BK`
    // instead, and scaling both would leave half the slots unwritten.
    let posmap = if padcoal {
        // `v25`'s map: plane follows the lane, slot follows the warp. A and B
        // share the slot because `v25` uses one `s0` for `arow` and `brow`, and
        // the two tiles have the same shape here (both 128 rows wide).
        format!(
            " let pos = ty * 16u + tx;\n\
             let akid = pos % 8u;\nlet bkid = pos % 8u;\n\
             let aslot = 4u * (pos / 32u) + ((pos / 8u) % 4u);\nlet bslot = aslot;\n"
        )
    } else if coalesce {
        format!(
            " let pos = ty * 16u + tx;\n\
             let akid = pos % TPI;\nlet bkid = pos % TPI;\n\
             let aslot = pos / TPI;\nlet bslot = pos / TPI;\n"
        )
    } else {
        format!(
            " let pos = ty * 16u + tx;\n\
             let akid = pos / TPI;\nlet bkid = pos / TPI;\n\
             let aslot = (pos % TPI) * {sga}u;\nlet bslot = (pos % TPI) * {sgb}u;\n"
        )
    };
    s.push_str(&posmap);
    // For the linear instantiation the B tile is read by `tx`, so slot `v` of
    // thread `tx` is `v * TX + tx` rather than `tx * VGN + v`. A warp's first
    // eight lanes then hold eight *distinct* slots, and the `vec4` read covers
    // all 32 banks; the contiguous grouping hit four banks twice each. The
    // strided variants keep the packed layout because their `tx` range differs.
    // Software pipeline: the next tile's global loads are issued *before* the
    // current tile is consumed, so one global latency is hidden per iteration.
    // Costs 8 `vec4` = 32 registers held across the FMA block; dropping it was
    // measured and changed nothing, so the registers are not what limits this
    // kernel. See the note above on what does.
    s.push_str(" let k0b: u32 = 0u;\n");
    load(&mut s, "k0b", true);
    s.push_str(" var buf: u32 = 0u;\n");
    store(&mut s, "buf");
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= g.k) { break; }\n  let kn = k0 + BK;\n");
    if pipe {
        // MEASURED NEGATIVE, default off (`GEMM_PIPE`). This is the statement order
        // of `v22` in sgemm_hs.cu, which is +19% over `v8` *in CUDA*; here it is
        // **-6%**, three order-swapped rounds, no overlap in the distributions
        // (seq 380: 1.09/1.08/1.09 off vs 1.03/1.02/1.01 on).
        //
        // The two languages want the load in opposite places relative to the
        // barrier. `__syncthreads()` is cheap and an LDG stays in flight across
        // it, so CUDA wants the load immediately *after* the barrier where it is
        // issued as late as possible and still covered by the FMA block. WGSL's
        // `workgroupBarrier()` is a full acquire-release that drains, so the load
        // wants to be as far from it as possible -- which is what the default
        // order already does, at the top of the loop. **Do not port CUDA loop
        // orderings across without measuring; the barrier is not the same object
        // in the two toolchains.**
        store(&mut s, "buf");
        s.push_str("  workgroupBarrier();\n  if (kn < g.k) {\n");
        load(&mut s, "kn", false);
        s.push_str("  }\n");
    } else {
        load(&mut s, "kn", true);
    }
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    if abl == 2 {
        s.push_str(&format!("   let av_0_0 = A4[buf * {nak}u + q0 * AST + {}];\n", a_read_slot(0)));
    }
    // `lds_pipe` software-pipelines the shared reads one k-step ahead, so step
    // `u`'s FMAs cover step `u+1`'s LDS latency. The two barriers per k-tile put
    // all eight warps of a workgroup in lockstep, so without this they reach the
    // LDS together and stall together. Costs one extra k-step of fragments.
    let steps: Vec<usize> = if lds_pipe { (0..BK / 4).collect() } else { Vec::new() };
    if lds_pipe {
        for v in 0..vgm {
            s.push_str(&format!("   let av_0_{v} = A4[buf * {nak}u + (q0 + 0u) * AST + {slot}];\n", slot = a_read_slot(v)));
        }
        for v in 0..vgn {
            s.push_str(&format!("   let bv_0_{v} = B4[buf * {nbk}u + (q0 + 0u) * BST + {slot}];\n", slot = b_read_slot(v)));
        }
    }
    for u in steps.iter().copied() {
        if lds_pipe {
            let nu = u + 1;
            if nu < BK / 4 {
                for v in 0..vgm {
                    s.push_str(&format!("   let av_{nu}_{v} = A4[buf * {nak}u + (q0 + {nu}u) * AST + {slot}];\n", slot = a_read_slot(v)));
                }
                for v in 0..vgn {
                    s.push_str(&format!("   let bv_{nu}_{v} = B4[buf * {nbk}u + (q0 + {nu}u) * BST + {slot}];\n", slot = b_read_slot(v)));
                }
            }
        }
        for i in 0..tm {
            let a = format!("av_{u}_{}[{}]", i / 4, i % 4);
            for j in 0..tn {
                let b = format!("bv_{u}_{}[{}]", j / 4, j % 4);
                s.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
            }
        }
    }
    for u in 0..BK / 4 {
        if lds_pipe || abl == 2 { break; }
        for v in 0..vgm {
            s.push_str(&format!("   let av_{u}_{v} = A4[buf * {nak}u + (q0 + {u}u) * AST + {slot}];\n", slot = a_read_slot(v)));
        }
        for v in 0..vgn {
            s.push_str(&format!("   let bv_{u}_{v} = B4[buf * {nbk}u + (q0 + {u}u) * BST + {slot}];\n", slot = b_read_slot(v)));
        }
        for i in 0..tm {
            let a = format!("av_{u}_{}[{}]", i / 4, i % 4);
            for j in 0..tn {
                let b = format!("bv_{u}_{}[{}]", j / 4, j % 4);
                s.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
            }
        }
    }
    // step by BK/4, not 4: the inner loop covers `u` in 0..BK/4, so a hardcoded
    // step of 4 reads planes {0..3}, {4..7}.. and silently skips 2,3 and 6,7 when
    // BK is 8.
    s.push_str(&format!("   q0 = q0 + {}u;\n  }}\n", BK / 4));
    if pipe {
        s.push_str("  workgroupBarrier();\n  k0 = kn;\n }\n");
    } else if dbuf {
        // one barrier: the stores go to the buffer nobody is reading this iteration,
        // and the barrier publishes them for the next one
        s.push_str("  if (kn < g.k) {\n");
        store(&mut s, "(buf ^ 1u)");
        s.push_str("  }\n  workgroupBarrier();\n  buf = buf ^ 1u;\n  k0 = kn;\n }\n");
    } else {
        s.push_str("  workgroupBarrier();\n  if (kn < g.k) {\n");
        store(&mut s, "buf");
        s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");
    }
    for i in 0..tm {
        // Mirrors `a_slot`: packed, thread `ty` owns rows `ty*tm .. +tm-1`;
        // interleaved, group `i/4` of `ty` owns rows `4*(i/4*TY + ty) .. + 3`.
        s.push_str(&format!("let row{i} = m0 + {};\n", hs_row(i)));
    }
    for j in 0..tn {
        if strided {
            // optional bias epilogue; keeps binding 2 live so the auto layout still
            // has all five bindings
            s.push_str(&format!(
                "let col{j} = n0 + {};\nvar bias{j} = 0.0;\nif (g.flags != 0u && col{j} < g.n) {{ bias{j} = Bias[col{j}]; }}\n",
                hs_col(j)
            ));
        } else {
            // `col` mirrors `b_slot`: group `j/4` of thread `tx` starts at column
            // `4 * (j/4 * TX + tx)`.
            s.push_str(&format!(
                "let col{j} = n0 + {};\nvar bias{j} = 0.0;\nif (g.has_bias != 0u && col{j} < g.n) {{ bias{j} = Bias[col{j}]; }}\n",
                hs_col(j)
            ));
        }
    }
    for i in 0..tm {
        s.push_str(&format!("if (row{i} < g.m) {{\n"));
        for j in 0..tn {
            // `col{j}` already carries `n0` (and, when `half`, the split mapping),
            // and it is the same expression the guard above uses. Building a second
            // index string here is how the `n0` prefix went missing once: the guard
            // passed for every n-tile while all of them stored to the same place.
            let cn = format!("col{j}");
            let add = format!(" + bias{j}");
            // Guard on `col{j}`, not on `n0 + tx*tn + j`: with the interleaved B
            // slot the two name different columns, and a partial tile (the head's
            // `out_proj` has n = 8) would then be written out of bounds or skipped.
            s.push_str(&format!(
                "  if (col{j} < g.n) {{ C[{}] = c{i}{j}{add}; }}\n",
                c_idx(&format!("row{i}"), &cn)
            ));
        }
        s.push_str("}\n");
    }
    s.push_str("}\n");
    s
}

/// Linear layers: `C[m,n] = A[m,k] @ W[n,k]^T (+ bias[n])` with `A` and `W` both
/// `[row, k]`, `k` contiguous.
///
/// 128x128 output tile, 128 threads, **16x8** register tile per thread, `BK = 16`,
/// shared tile k-major (`S[k][row/4] = vec4(A[4c..4c+3][k])`) so the read side is
/// one `vec4` per operand per k-step. Two things differ from the older 256-thread
/// 8x8 kernel:
///
/// * **Staging is a stride-k gather, not a vector load.** Each thread reads four
///   *rows* at one k with four scalar loads (`la_{v}_{e} = A[(m0 + slot*4 + e) *
///   a_rs + abase + (kbase + kid)]`) and packs them into one `vec4`. A `vec4` load
///   would fetch four consecutive k of a single row, which is the wrong shape for
///   the transposed shared tile, so there is nothing to vectorise here. (An earlier
///   doc comment claimed a coalesced 4x4 transposing load; that design is not in the
///   code and the comment was wrong.)
/// * **8x8 register tile, split into two halves.** See `gemm_wgsl_bk`: the packed
///   mapping gives the B-side read a 4-way bank conflict, and the half-split
///   mapping costs nothing because it only moves the read-side slot and the
///   epilogue, never the staging.
///
/// `k` must be a multiple of `BK` — every linear layer in this model is
/// (`512`, `2048`, `192`, `576`) — so the kernel carries no per-element k guard.
fn gemm_wgsl() -> String {
    let abl: u8 = std::env::var("GEMM_ABLATE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // GEMM_BM=64 switches the output tile to 64x128 with a 4x8 register tile. Its
    // purpose is wave quantization, not per-block throughput: halving BM doubles
    // the number of m-tiles, so the grid reaches 2*gx*gy128 tiles and the ragged
    // last wave is a smaller fraction of the work. On sm_61 (`v11` in
    // sgemm_hs.cu) that is +24..28% for seq 385..576 and -13..19% for seq <= 384
    // and 641..768, because 4*gy128 is a multiple of the resident-slot count
    // exactly when gy128 % 3 == 0. It is a probe, not the default.
    let tm = match std::env::var("GEMM_BM").ok().as_deref() {
        Some("64") => 4usize,
        _ => 8usize,
    };
    gemm_wgsl_t(tm, 8, false, abl, std::env::var_os("GEMM_LDSPI").is_some())
}

/// The 64×128 instantiation of the *same* kernel, as its own pipeline, so a single
/// dispatch can pick its row tile. Identical to `GEMM_BM=64` in every respect
/// except that it does not re-tile the calls that do not want it — which is the
/// whole point, see [`gemm_wave_bm`].
fn gemm_wgsl_wide() -> String {
    let abl: u8 = std::env::var("GEMM_ABLATE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    gemm_wgsl_t(4, 8, false, abl, std::env::var_os("GEMM_LDSPI").is_some())
}

/// Resident workgroup slots of the linear GEMM: `__launch_bounds__(256, 2)` on
/// sm_61 is 2 blocks/SM, and a 1050 Ti has 6 SMs. It is a property of the machine
/// and the register count, not of the kernel, so it is an override rather than a
/// constant.
fn gemm_slots() -> u32 {
    static S: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("GEMM_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(12)
    })
}

/// Row tile for **one** linear dispatch, chosen by wave quantization rather than
/// globally.
///
/// **Measured: this buys nothing. Keep it off and do not re-run it.** The model
/// below predicted 1.13~1.33x on `o` and `fc2` at `gy == 4`, and the CUDA
/// testbed measured +26% for the same shape at `seq = 541` (`v8` 0.95 vs `v11`
/// 1.20 TFLOP/s). End to end it is a small consistent *loss*: interleaved,
/// order-swapped `low_latency` runs give 6.24/6.20, 6.18/6.15, 6.20/6.16 —
/// -0.6%, -0.5%, -0.6%, tight enough that it is not the run-to-run band. The
/// four-mode `frame_check` is 112.1 / 6.2 / 4.2 / 2.1 against
/// 110.7~115.2 / 6.2 / 4.2 / 2.1. The output is bit-identical (every `maxdiff`
/// and `flips` matches the default run exactly), so the 64x128 instantiation is
/// correct; it is just not faster.
///
/// The model is arithmetically right and *still* wrong as a performance
/// predictor, and the reason generalises past this kernel: a dispatch does not
/// run in lockstep waves. Blocks are scheduled dynamically, so the ragged last
/// wave is backfilled as soon as any block retires, and the
/// `ceil(tiles / slots)` penalty it assumes never fully materialises. Every
/// earlier use of that model in this file — including the "4*gy128 is a multiple
/// of the resident-slot count exactly when gy128 % 3 == 0" note on `GEMM_BM` —
/// has to be read with that in mind. It describes the slot arithmetic, not the
/// clock. It also means `cuBLAS` dropping to 1.24 TFLOP/s at `seq = 389` versus
/// 2.22 at `seq = 380` (10 tiles more, 2 tiles per wave) is probably *not*
/// wave quantization and still has no explanation.
///
/// What the model is left good for: picking a row tile that does not compute
/// rows nobody asked for. `m <= 64` at any `n`, and `m <= 128` at `n = 512`,
/// fit in one row block either way, and halving the row tile halves the work at
/// identical wave count. That is the `seq = 8 -> 36 ms` floor of the tower
/// curve. It measured as noise too, on the same runs.
///
/// `eff(64) / eff(128) = 0.855` is measured, not assumed: at `seq = 380`
/// (`gy = 3`, where both shapes are wave-balanced) `v8` runs 1.17 and `v11`
/// 1.00 TFLOP/s in `sgemm_hs.cu`.
pub fn gemm_wave_bm(m: u32, n: u32) -> u32 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("GEMM_WAVETILE").is_some()) {
        return GEMM_BM;
    }
    let gx = n.div_ceil(GEMM_BN).max(1);
    let slots = gemm_slots();
    gemm_wave_pick(m, n, slots)
}

/// The wave cost of one linear dispatch at a given row tile, as `gemm_wave_bm`
/// uses it. Split out so the arithmetic can be tested without a GPU or an env
/// var — the whole decision is `cost(64) < cost(128)`, and a silent change of
/// `GEMM_BM` or `GEMM_BN` would move the crossover without moving a test.
pub fn gemm_wave_cost(m: u32, n: u32, bm: u32, slots: u32) -> f64 {
    let gx = n.div_ceil(GEMM_BN).max(1);
    let gy = m.div_ceil(bm).max(1);
    let waves = (gx * gy).div_ceil(slots).max(1);
    waves as f64 * bm as f64
}

/// 64x128 is a 4x8 register tile: 12 shared floats per 32 FMA instead of 16 per
/// 64, i.e. 0.855x the per-FMA throughput. Measured, not assumed: at `seq = 380`
/// (`gy = 3`, where both shapes hit 100% wave fill) `sgemm_hs.cu` runs v8 at
/// 1.17 and v11 at 1.00 TFLOP/s.
pub const GEMM_M64_EFF: f64 = 0.855;

pub fn gemm_wave_pick(m: u32, n: u32, slots: u32) -> u32 {
    if gemm_wave_cost(m, n, 64, slots) / GEMM_M64_EFF < gemm_wave_cost(m, n, GEMM_BM, slots) {
        64
    } else {
        GEMM_BM
    }
}

/// Attention `QK^T`: same kernel, strided instantiation. `q`/`k`/`v` are
/// `[seq, heads*hd]` row-major, so QK^T is a `K == head_dim` GEMM whose operands
/// and result all carry a non-`k` row stride, and the head index rides in
/// `workgroup_id.z`.
fn gemm_strided_wgsl() -> String {
    gemm_wgsl_t(8, 8, true, 0, false)
}

const LN_WGSL: &str = r#"
struct Cfg { cols: u32, eps: f32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bs: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Cfg;

const BS: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn ln(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let cols = cfg.cols;
    var s = 0.0;
    for (var i = lid.x; i < cols; i += BS) { s += X[row * cols + i]; }
    red[lid.x] = s;
    workgroupBarrier();
    for (var k = BS >> 1u; k > 0u; k >>= 1u) {
        if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(cols);
    workgroupBarrier();
    var v = 0.0;
    for (var i = lid.x; i < cols; i += BS) {
        let d = X[row * cols + i] - mean;
        v += d * d;
    }
    red[lid.x] = v;
    workgroupBarrier();
    for (var k = BS >> 1u; k > 0u; k >>= 1u) {
        if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(cols) + cfg.eps);
    for (var i = lid.x; i < cols; i += BS) {
        Y[row * cols + i] = (X[row * cols + i] - mean) * inv * Wt[i] + Bs[i];
    }
}
"#;

const UNARY_WGSL: &str = r#"
struct Cfg { n: u32, mode: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;

fn erf_as(z: f32) -> f32 {
    let ax = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return select(-y, y, z >= 0.0);
}

@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let x = X[i];
    if (cfg.mode == 0u) {
        // gelu erf: 0.5 * x * (1 + erf(x / sqrt(2)))
        X[i] = 0.5 * x * (1.0 + erf_as(x * 0.7071067811865476));
    } else {
        X[i] = max(x, 0.0);
    }
}
"#;

const ADD_WGSL: &str = r#"
struct Cfg { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> C: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    C[i] = C[i] + B[i];
}
"#;

const ROPE_WGSL: &str = r#"
struct Cfg { seq: u32, heads: u32, hd: u32, col: u32, row_stride: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Cos: array<f32>;
@group(0) @binding(2) var<storage, read> Sin: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;

/// One thread per `(row, head, pair)`, so the `half` loop of the original
/// (one thread per `(row, head)`, striding by `hidden`) is gone: consecutive lanes
/// now read consecutive elements and every global access coalesces.
@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.seq * cfg.heads * (cfg.hd / 2u)) { return; }
    let hd = cfg.hd;
    let half = hd / 2u;
    let i = idx % half;
    let t = (idx / half) % cfg.seq;
    let head = idx / (half * cfg.seq);
    // `cfg.col` picks which column block of the packed buffer this pass rotates
    // (0 = Q, `hidden` = K) and `cfg.row_stride` is that buffer's row length,
    // `3*hidden`. Treating QKV as three separate `[seq, hidden]` matrices makes
    // both of these the wrong number by a factor of three.
    let base = cfg.col + t * cfg.row_stride + head * hd;
    let cbase = t * hd;
    let x1 = X[base + i];
    let x2 = X[base + half + i];
    X[base + i] = x1 * Cos[cbase + i] - x2 * Sin[cbase + i];
    X[base + half + i] = x2 * Cos[cbase + half + i] + x1 * Sin[cbase + half + i];
}
"#;

const ATTN_WGSL: &str = r#"
struct Cfg { seq: u32, heads: u32, hd: u32, valid: u32 }
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Cfg;

const WG: u32 = 64u;
var<workgroup> qsh: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
var<workgroup> scores: array<f32, 1024>;

@compute @workgroup_size(64)
fn attn(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = wgid.x;
    let head = wgid.y;
    let t = lid.x;
    let hd = cfg.hd;
    let hidden = cfg.heads * hd;
    let valid = cfg.valid;
    qsh[t] = Q[qi * hidden + head * hd + t];
    workgroupBarrier();

    let scale = inverseSqrt(f32(hd));
    for (var j = t; j < valid; j += WG) {
        var acc = 0.0;
        let kb = j * hidden + head * hd;
        for (var d = 0u; d < hd; d += 4u) {
            acc += qsh[d] * K[kb + d] + qsh[d + 1u] * K[kb + d + 1u]
                + qsh[d + 2u] * K[kb + d + 2u] + qsh[d + 3u] * K[kb + d + 3u];
        }
        scores[j] = acc * scale;
    }
    workgroupBarrier();

    var mx = -1.0e30;
    for (var j = t; j < valid; j += WG) { mx = max(mx, scores[j]); }
    red[t] = mx;
    workgroupBarrier();
    for (var s = WG >> 1u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] = max(red[t], red[t + s]); }
        workgroupBarrier();
    }
    let gmax = red[0];
    workgroupBarrier();

    var sm = 0.0;
    for (var j = t; j < valid; j += WG) {
        let e = exp(scores[j] - gmax);
        scores[j] = e;
        sm += e;
    }
    red[t] = sm;
    workgroupBarrier();
    for (var s = WG >> 1u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] += red[t + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    workgroupBarrier();
    for (var j = t; j < valid; j += WG) { scores[j] *= inv; }
    workgroupBarrier();

    for (var d = t; d < hd; d += WG) {
        var acc = 0.0;
        for (var j = 0u; j < valid; j++) {
            acc += scores[j] * V[j * hidden + head * hd + d];
        }
        Out[qi * hidden + head * hd + d] = acc;
    }
}
"#;

const ATTN_SM_WGSL: &str = r#"
struct Cfg { rows: u32, cols: u32, valid: u32, scale: f32 }
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;

// 8 rows per workgroup, 32 lanes per row. The reduction is 5 barrier steps and it
// is amortised over 8 independent rows, so the barrier cost per row drops ~13x
// versus one 256-lane workgroup per row.
const RX: u32 = 32u;
const RY: u32 = 8u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(32, 8)
fn softmax_rows(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // `live` guards only the global accesses — never an early return, because the
    // reductions below are workgroup barriers and those must be reached uniformly.
    let r = wgid.x * RY + lid.y;
    let live = r < cfg.rows;
    let row = select(0u, r, live);
    let lane = lid.x;
    let valid = cfg.valid;
    let base = row * cfg.cols;
    var mx = -1.0e30;
    for (var i = lane; i < valid; i += RX) {
        if (live) { mx = max(mx, S[base + i]); }
    }
    red[lid.y * RX + lane] = mx;
    workgroupBarrier();
    for (var k = RX >> 1u; k > 0u; k >>= 1u) {
        if (lane < k) { red[lid.y * RX + lane] = max(red[lid.y * RX + lane], red[lid.y * RX + lane + k]); }
        workgroupBarrier();
    }
    let gmax = red[lid.y * RX];
    workgroupBarrier();
    var sm = 0.0;
    for (var i = lane; i < valid; i += RX) {
        if (live) {
            let e = exp((S[base + i] - gmax) * cfg.scale);
            S[base + i] = e;
            sm += e;
        }
    }
    // masked keys contribute exactly zero to PV
    for (var i = valid + lane; i < cfg.cols; i += RX) {
        if (live) { S[base + i] = 0.0; }
    }
    red[lid.y * RX + lane] = sm;
    workgroupBarrier();
    for (var k = RX >> 1u; k > 0u; k >>= 1u) {
        if (lane < k) { red[lid.y * RX + lane] += red[lid.y * RX + lane + k]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[lid.y * RX];
    for (var i = lane; i < valid; i += RX) {
        if (live) { S[base + i] *= inv; }
    }
}
"#;

/// im2col for the head's kernel-3 sub-pixel conv: `A[t, ic*3 + j] = X[t-1+j, ic]`,
/// laid out to match `conv.weight` stored as `[hh*up, hh, 3]` so the conv becomes a
/// plain `[frames, hh*3, hh*up]` GEMM. The naive per-output-channel kernel this
/// replaces re-read the input once per output channel — about 2.4 GB of loads per
/// window — and was the single most expensive thing in the classification head.
const COL3_WGSL: &str = r#"
struct Cfg { frames: u32, hh: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn col3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.frames * cfg.hh * 3u) { return; }
    let t = idx / (cfg.hh * 3u);
    let r = idx % (cfg.hh * 3u);
    let ic = r / 3u;
    let j = r % 3u;
    let src = i32(t) + i32(j) - 1;
    if (src < 0 || src >= i32(cfg.frames)) {
        A[idx] = 0.0;
    } else {
        A[idx] = X[u32(src) * cfg.hh + ic];
    }
}
"#;

const CONV_WGSL: &str = r#"
struct Cfg { frames: u32, hh: u32, out_c: u32, _p: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Cfg;

@compute @workgroup_size(8, 8)
fn conv(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = wgid.x * 8u + lid.x;
    let oc = wgid.y * 8u + lid.y;
    if (t >= cfg.frames || oc >= cfg.out_c) { return; }
    let hh = cfg.hh;
    var acc = Bias[oc];
    for (var ic = 0u; ic < hh; ic++) {
        let wbase = (oc * hh + ic) * 3u;
        let left = select(0.0, X[(t - 1u) * hh + ic], t > 0u);
        let mid = X[t * hh + ic];
        let right = select(0.0, X[(t + 1u) * hh + ic], t + 1u < cfg.frames);
        acc += W[wbase] * left + W[wbase + 1u] * mid + W[wbase + 2u] * right;
    }
    Y[t * cfg.out_c + oc] = acc;
}
"#;

#[cfg(test)]
mod wave_tile_tests {
    use super::{gemm_wave_cost, gemm_wave_pick, GEMM_BM};

    /// 12 resident slots on a 6-SM sm_61 with `__launch_bounds__(256, 2)`.
    const SLOTS: u32 = 12;

    /// `n == 512` is the only shape whose tile count is `4 * gy`, and `4 * gy` is
    /// a multiple of 12 exactly when `gy % 3 == 0`. So the 64-row tile must win at
    /// `gy == 4` (seq 385..512) and must *not* win at `gy == 3` (seq 257..384),
    /// which is where `ultra_low_latency` sits once its cache prefix is full.
    #[test]
    fn n512_crossover_is_at_gy_four() {
        assert_eq!(gemm_wave_pick(343, 512, SLOTS), GEMM_BM, "gy=3 must stay 128-row");
        assert_eq!(gemm_wave_pick(380, 512, SLOTS), GEMM_BM, "gy=3 must stay 128-row");
        assert_eq!(gemm_wave_pick(505, 512, SLOTS), 64, "gy=4 must switch to 64-row");
        assert_eq!(gemm_wave_pick(541, 512, SLOTS), 64, "gy=5 rounds up, still 64");
    }

    /// At every `seq` the streaming modes actually run once the cache prefix is
    /// full, `n == 1536` (qkv) and `n == 2048` (fc1) come out ahead on 128-row
    /// tiles, so re-tiling them can only lose the 4x8 tile's shared-traffic ratio.
    /// This is the assertion that keeps the change off qkv and fc1 in production
    /// -- together the largest share of a tower.
    ///
    /// The boundary is not a clean cutoff and must not be asserted as one: at
    /// `seq = 257, n = 1536` the 64-row tile *does* win, because
    /// `ceil(257/64) = 5` row blocks (320 rows) beat `ceil(257/128) = 3`
    /// (384 rows) on both padding and waves. Only the shapes that production
    /// runs are pinned here.
    #[test]
    fn wide_n_never_re_tiles_at_production_seqs() {
        // The seq values the four modes actually run once the cache prefix is
        // full (offline 380/505/684, low 496..541, very 520..536, ultra 340..375).
        for seq in [340u32, 343, 380, 464, 496, 505, 524, 541, 684] {
            for n in [1536u32, 2048] {
                assert_eq!(
                    gemm_wave_pick(seq, n, SLOTS),
                    GEMM_BM,
                    "seq={seq} n={n} must stay 128-row"
                );
            }
        }
    }

    /// The other reason to prefer the 64-row tile is not about waves at all: when
    /// the whole `m` fits in one row block *and* the tile count stays within the
    /// resident slots, both instantiations run one wave but the 128-row one
    /// computes up to twice the rows it needs. This is the `seq = 8 -> 36 ms`
    /// floor of the tower curve, ~33 ms of it all wasted rows. The boundary is
    /// `m <= 64` at `n = 1536` (where halving BM doubles the tile count from 12
    /// to 24 and costs a second wave) and `m <= 128` at `n = 512` (where it goes
    /// 4 -> 8 and still fits).
    #[test]
    fn one_row_block_switches_only_while_it_still_fits() {
        for seq in [8u32, 24, 64] {
            for n in [512u32, 1536, 2048] {
                assert_eq!(gemm_wave_pick(seq, n, SLOTS), 64, "seq={seq} n={n}");
            }
        }
        // n=512 still fits at 128 rows; n=1536 does not.
        assert_eq!(gemm_wave_pick(128, 512, SLOTS), 64);
        assert_eq!(gemm_wave_pick(128, 1536, SLOTS), GEMM_BM);
    }

    /// The predicted ratio at the crossover is the thing CUDA measured
    /// independently: v8 0.95 vs v11 1.20 TFLOP/s at seq=541, i.e. 1.26x. If the
    /// model ever predicts something far from that, the model is what is wrong.
    #[test]
    fn predicted_gain_at_crossover_matches_cublas_side_measurement() {
        let wide = gemm_wave_cost(505, 512, GEMM_BM, SLOTS);
        let narrow = gemm_wave_cost(505, 512, 64, SLOTS) / 0.855;
        let predicted = wide / narrow;
        assert!(
            (1.05..=1.35).contains(&predicted),
            "predicted {predicted:.3}x at seq=505 n=512, CUDA measured 1.26x"
        );
    }

    /// A machine with more resident slots has fewer ragged waves to fix, so the
    /// switch should stop happening. This is the knob to turn on a card where
    /// `GEMM_SLOTS` is not 12 -- it must visibly change the answer, otherwise the
    /// override is not wired to anything.
    #[test]
    fn slot_count_actually_moves_the_decision() {
        // 12 slots: 16 tiles need 2 waves, so the switch is a pure wave win.
        assert_eq!(gemm_wave_pick(505, 512, SLOTS), 64);
        // 4 slots: 16 tiles need 4 waves and 32 tiles need 8 -- the 64-row tile
        // doubles the waves it saves rows in, and the 0.855 ratio decides.
        assert_eq!(gemm_wave_pick(505, 512, 4), GEMM_BM, "4 slots: waves dominate");
    }
}
