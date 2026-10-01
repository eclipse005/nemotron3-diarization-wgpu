//! Stage-5 validation: do the wgpu kernels reproduce the CPU encoder and head?
//!
//! 1. Print the adapter.
//! 2. Run the GPU tower on the first offline window of `diarization_example`
//!    and diff against the CPU tower (and against `baseline/hidden/...final_ln.npy`).
//! 3. Run the GPU head on the reference hidden states and diff against the CPU head.
//!
//! Usage:  cargo run --release --bin gpu_check [../baseline] [file-stem]

use std::path::PathBuf;
use std::time::Instant;

use nemotron3_diarization_wgpu::{read_f32, GpuEngine, Model, Weights};

fn maxdiff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../baseline".into()));
    let stem = std::env::args().nth(2).unwrap_or_else(|| "diarization_example".into());
    let ckpt = PathBuf::from(
        std::env::args().nth(3).unwrap_or_else(|| "../models/Nemotron-3-Diarization".into()),
    );

    let w = Weights::load(&ckpt)?;
    let cpu = Model::load(&ckpt)?;
    let mut gpu = GpuEngine::load(&cpu.cfg, &w)?;
    println!("gpu: {}", gpu.describe());
    if std::env::var("EMBED_CHECK").is_ok() {
        // offline embeds one window at a time instead of the whole recording (that
        // is what removed the ~10.9 min length cap), so the per-window path has to
        // be bit-identical to the whole-file one. Windows overlap by the right
        // context, which is why several of these are not aligned to the window size.
        for (frames, win) in [(2049usize, 380usize), (4001, 380), (9761, 380), (20011, 380), (9761, 341), (9761, 1024)] {
            let (worst, exact) = gpu.check_embed_range(frames, win);
            println!(
                "  embed {frames:6} frames, window {win:5}: maxdiff {worst:.3e}  {}",
                if exact { "bit-identical" } else { "FAIL" }
            );
        }
        return Ok(());
    }
    if std::env::var("CONCUR_CHECK").is_ok() {
        for seq in [24u32, 128, 380] {
            let (solo, pair, chain) = gpu.probe_dispatch_concurrency(seq);
            println!(
                "  seq={seq:4}  solo {solo:7.3} ms | pair(disjoint) {pair:7.3} ms ({:.2}x solo) | chain(dependent) {chain:7.3} ms ({:.2}x solo)",
                pair / solo, chain / solo
            );
        }
        return Ok(());
    }
    if std::env::var("ATTN_CHAIN_CHECK").is_ok() {
        for seq in [64u32, 380] {
            let d = gpu.check_attn_chain(seq, &w);
            println!("  attn-chain seq={seq}: maxdiff {d:.3e}  {}", if d < 1e-3 { "ok" } else { "FAIL" });
        }
        return Ok(());
    }
    if std::env::var("QKV_CHECK").is_ok() {
        for seq in [17u32, 64, 380] {
            let d = gpu.check_qkv(seq, &w);
            println!("  qkv seq={seq}: maxdiff {d:.3e}  {}", if d < 1e-3 { "ok" } else { "FAIL" });
        }
        return Ok(());
    }
    if std::env::var("ROPE_CHECK").is_ok() {
        for seq in [380u32, 505, 684, 1024] {
            let (d, unwritten) = gpu.check_rope(seq);
            println!("  rope seq={seq}: maxdiff {d:.3e} unwritten {unwritten:.0}  {}", if d < 1e-5 { "ok" } else { "FAIL" });
        }
        return Ok(());
    }
    if std::env::var("ATTN_GEMM_CHECK").is_ok() {
        // The two attention GEMMs are separate strided instantiations with their
        // own operand/result strides; GEMM_CHECK never touches them.
        let h = cpu.cfg.audio_config.hidden_size as u32;
        let nh = cpu.cfg.audio_config.num_attention_heads as u32;
        let hd = h / nh;
        for seq in [8u32, 17, 64, 130, 380, 684] {
            let dq = gpu.check_gemm_qk(seq, nh, hd, h);
            let dp = gpu.check_gemm_pv(seq, nh, hd, h);
            println!(
                "  attn-gemm seq={seq}: qk maxdiff {dq:.3e} {}  pv maxdiff {dp:.3e} {}",
                if dq < 1e-3 { "ok" } else { "FAIL" },
                if dp < 1e-3 { "ok" } else { "FAIL" }
            );
        }
        return Ok(());
    }
    if std::env::var("SM_CHECK").is_ok() {
        for (rows, cols, valid) in [(3040u32, 380u32, 380u32), (5472, 684, 684), (5472, 684, 683), (17, 64, 40), (8, 8, 5)] {
            let d = gpu.check_attn_softmax(rows, cols, valid);
            println!(
                "  softmax {rows}x{cols} valid={valid}: maxdiff {d:.3e}  {}",
                if d < 1e-5 { "ok" } else { "FAIL" }
            );
        }
        return Ok(());
    }
    if std::env::var("GEMM_CHECK").is_ok() {
        // `GEMM_SHAPE=m,n,k` narrows this to one case, which is how a kernel
        // edit gets bisected down to a single tile/k combination.
        // The `n` values deliberately include a partial tile (100) and a very
        // narrow one (8, the head's `out_proj`): a column-mapping bug that only
        // shows up off the tile boundary passes every full-tile shape.
        let mut shapes: Vec<(u32, u32, u32)> = vec![
            (128, 128, 512),
            (380, 512, 512),
            (380, 2048, 512),
            (684, 512, 512),
            (505, 512, 512),
            (380, 100, 512),
            (380, 8, 512),
            (17, 24, 512),
        ];
        if let Ok(v) = std::env::var("GEMM_SHAPE") {
            let p: Vec<u32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            assert_eq!(p.len(), 3, "GEMM_SHAPE wants m,n,k");
            shapes = vec![(p[0], p[1], p[2])];
        }
        for (m, n, k) in shapes {
            let d = gpu.check_gemm(m, n, k);
            println!("  gemm {m}x{n}x{k}: maxdiff {d:.3e}  {}", if d < 1e-3 { "ok" } else { "FAIL" });
        }
        return Ok(());
    }
    if std::env::var("GEMM_BENCH").is_ok() {
        gpu.warm_gpu();
        for seq in [64u32, 128, 256, 380, 512, 684] {
            let _ = gpu.bench_gemm(seq, 5);
            let (ms, tflops) = gpu.bench_gemm(seq, 40);
            println!("  gemm {seq}x512x512 x40  {ms:.1} ms  {tflops:.2} TFLOP/s");
        }
        return Ok(());
    }
    if std::env::var("SUBMIT_COST").is_ok() {
        gpu.warm_gpu();
        println!("what                         per-submit ms");
        for (name, ms) in gpu.probe_submit_cost() {
            println!("{name:<28} {ms:8.3}");
        }
        return Ok(());
    }
    if std::env::var("TOWER_CURVE").is_ok() {
        // Where does a tower's time actually go, as a function of seq? The
        // streaming modes walk seq from 4/8/13 up to ~541, and every number
        // measured so far was at seq 380-684, i.e. only the fat end. This
        // prints full-tower time next to linear-GEMM-only time and attention
        // time so the non-GEMM residue (softmax, elementwise, per-dispatch
        // overhead, and the idle SMs when gy*gx < 6) is visible directly.
        use nemotron3_diarization_wgpu::{S_ADD, S_ALL, S_ATTN, S_FC1, S_FC2, S_GELU, S_LN, S_O, S_QKV, S_ROPE, S_SM};
        let seqs: Vec<u32> = std::env::var("STAGE_SEQ")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![4, 8, 13, 24, 40, 64, 96, 128, 192, 256, 320, 380, 464, 541, 684]);
        let iters: u32 = std::env::var("STAGE_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
        let lin = S_QKV | S_O | S_FC1 | S_FC2;
        gpu.warm_gpu();
        println!("seq    tower    linear   attn   other   lin TFLOP/s  whole-tower TFLOP/s   ms/dispatch");
        for seq in seqs {
            let _ = gpu.bench_stages(seq, seq, S_ALL, 5);
            let full = gpu.bench_stages(seq, seq, S_ALL, iters) / iters as f64;
            let lin_t = gpu.bench_stages(seq, seq, lin, iters) / iters as f64;
            let attn_t = gpu.bench_stages(seq, seq, S_ATTN, iters) / iters as f64;
            let other = (full - lin_t - attn_t).max(0.0);
            // 2 layers-worth of the four linear GEMMs, per tower
            let flop = 2.0 * seq as f64 * (1536.0 * 512.0 + 512.0 * 512.0 + 2048.0 * 512.0 + 512.0 * 2048.0) * 31.0;
            let per_tower = full * 1000.0; // ms
            // ~10 dispatches per layer x 31 layers is the tower's dispatch count
            let dispatches = 31.0 * 12.0;
            println!(
                "{seq:<5} {full:7.3}ms {lin_t:7.3}ms {attn_t:6.3}ms {other:6.3}ms {:>9.3} {:>17.3} {:>13.4}",
                flop / (lin_t * 1e9),
                flop / (full * 1e9),
                per_tower / dispatches
            );
        }
        let _ = (S_LN, S_GELU, S_ADD, S_ROPE, S_SM);
        return Ok(());
    }
    if std::env::var("ATTN_BENCH").is_ok() {
        use nemotron3_diarization_wgpu::{
            S_ADD, S_ALL, S_ATTN, S_FC1, S_FC2, S_GELU, S_LN, S_O, S_PV, S_QK, S_QKV, S_ROPE, S_SM,
        };
        let iters: u32 = std::env::var("STAGE_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        // Default is the fat end, but the streaming modes actually run at the seq
        // values round 18 found to matter most -- 385..512 is 36% of
        // `ultra_low_latency`'s steps and is where the wave-quantization cliff
        // sits -- so `ATTN_SEQ=392,512` measures where the work is.
        let attn_seqs: Vec<u32> = std::env::var("ATTN_SEQ")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![380, 505, 684]);
        for seq in attn_seqs {
            println!("\nattention, seq {seq}, {iters} layers each");
            for (name, bit) in [
                ("qk gemm", S_QK),
                ("softmax", S_SM),
                ("pv gemm", S_PV),
            ] {
                let _ = gpu.bench_stages(seq, seq, bit, 2);
                let t = gpu.bench_stages(seq, seq, bit, iters);
                println!("  {name:<10} {t:>8.1} ms  ({:>6.1} us/tower, {:>5.1} us/layer)", t / iters as f64 * 1e3, t / iters as f64 / 31.0 * 1e3);
            }
            let _ = gpu.bench_stages(seq, seq, S_ALL, 2);
            let full = gpu.bench_stages(seq, seq, S_ALL, iters);
            let no_attn = gpu.bench_stages(seq, seq, S_ALL & !S_ATTN, iters);
            let attn = full - no_attn;
            // `attn` is `full - no_attn`, and both are totals over `iters` *towers*,
            // each of which runs all 31 layers -- so the flop count needs the x31.
            // Without it the printed figure is 31x too low (it read 0.014
            // TFLOP/s at seq 392 where the real value is 0.42), which is the same
            // class of error as round 9's un-divided tower total.
            let fl = 2.0 * 2.0 * seq as f64 * seq as f64 * 8.0 * 64.0 * 31.0 * iters as f64;
            println!(
                "  {:<10} {attn:>8.1} ms  ({:.1} us/tower)  {:.1} GFLOP -> {:.3} TFLOP/s",
                "attention",
                attn / iters as f64 * 1e3,
                fl / 1e9,
                fl / (attn * 1e9)
            );
            // rest of the tower, for reference
            println!("  {:<10} {:>8.1} ms", "no attn", no_attn);
            for (name, bit) in [
                ("qkv gemm", S_QKV),
                ("o gemm", S_O),
                ("fc1 gemm", S_FC1),
                ("fc2 gemm", S_FC2),
                ("layer_norm", S_LN),
                ("rope", S_ROPE),
                ("gelu", S_GELU),
                ("add", S_ADD),
            ] {
                let t = gpu.bench_stages(seq, seq, bit, iters);
                println!("  {name:<10} {t:>8.1} ms");
            }
        }
        return Ok(());
    }
    if std::env::var("STAGE_BENCH").is_ok() {
        use nemotron3_diarization_wgpu::{S_ALL, S_ATTN, S_ADD, S_FC1, S_FC2, S_GELU, S_LN, S_O, S_QKV, S_ROPE};
        let iters: u32 = std::env::var("STAGE_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        for seq in [380u32, 684] {
            println!("\ntower stage breakdown, seq {seq}, {iters} towers, 31 layers each");
            let _ = gpu.bench_stages(seq, seq, S_ALL, 2);
            let full = gpu.bench_stages(seq, seq, S_ALL, iters);
            println!("  {:<22} {:>9.1} ms", "everything", full);
            let mut prev = full;
            for (name, bit) in [
                ("attn", S_ATTN),
                ("qkv gemm", S_QKV),
                ("o gemm", S_O),
                ("fc1 gemm", S_FC1),
                ("fc2 gemm", S_FC2),
                ("layer_norm", S_LN),
                ("rope", S_ROPE),
                ("gelu", S_GELU),
                ("residual add", S_ADD),
            ] {
                let t = gpu.bench_stages(seq, seq, S_ALL & !bit, iters);
                println!(
                    "  {:<22} {:>9.1} ms   (without it: {:>8.1} ms, cost {:>7.1} ms)",
                    name,
                    t,
                    t,
                    prev - t
                );
                prev = t;
            }
            // attention arithmetic, for a TFLOP/s read
            let nh = 8.0f64;
            let hd = 64.0f64;
            let fl = 2.0 * 2.0 * seq as f64 * seq as f64 * nh * hd * 31.0 * iters as f64;
            let t_attn = full - gpu.bench_stages(seq, seq, S_ALL & !S_ATTN, iters);
            println!("  attention = {fl:.1} MFLOP -> {:.3} TFLOP/s", fl / (t_attn * 1e9));
        }
        return Ok(());
    }

    let samples: Vec<f32> = read_f32(&root.join("waveforms").join(format!("{stem}.npy")))?
        .data
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let fe = &cpu.proc.feature_extractor;

    let t0 = Instant::now();
    let mut trace = Vec::new();
    cpu.run_offline_capturing(&samples, Some(&mut trace))?;
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let w0 = &trace[0];
    println!(
        "cpu first window: {} frames, valid {}, scored {}, {cpu_ms:.0} ms for the whole file",
        w0.window_frames, w0.valid_frames, w0.scored_frames
    );

    let filters = nemotron3_diarization_wgpu::mel_filters(fe);
    let mut mel = nemotron3_diarization_wgpu::log_mel(
        &samples,
        fe,
        nemotron3_diarization_wgpu::Padding::Centered,
        &filters,
    );
    let valid_mel = nemotron3_diarization_wgpu::frame_count(
        samples.len(),
        fe,
        nemotron3_diarization_wgpu::Padding::Centered,
    );
    mel.resize((valid_mel + 1) * fe.feature_size, 0.0);
    let all_embeds = cpu.tower.embed(&mel, valid_mel + 1);
    let hidden = cpu.hidden_size();
    let win = w0.window_frames;
    let window = &all_embeds[..win * hidden];

    let t1 = Instant::now();
    let gpu_enc = gpu.forward_embeds(window, w0.valid_frames)?;
    let gpu_enc_ms = t1.elapsed().as_secs_f64() * 1e3;
    let cpu_enc = &w0.encoder_out;
    let d_enc = maxdiff(&gpu_enc, cpu_enc);
    println!(
        "encoder  GPU vs CPU  maxdiff {d_enc:.3e}  ({gpu_enc_ms:.1} ms for {} frames)",
        win
    );

    let final_ln = read_f32(&root.join("hidden").join(format!("{stem}__offline__final_ln.npy")))?;
    let d_ref = maxdiff(&gpu_enc, &final_ln.data[..win * hidden].iter().map(|v| *v as f32).collect::<Vec<_>>());
    println!("encoder  GPU vs baseline final_ln  maxdiff {d_ref:.3e}");

    let t2 = Instant::now();
    let gpu_head = gpu.head_forward(&gpu_enc, win)?;
    let gpu_head_ms = t2.elapsed().as_secs_f64() * 1e3;
    let cpu_head = cpu.head.forward(&w0.encoder_out, win);
    let d_head = maxdiff(&gpu_head, &cpu_head);
    println!(
        "head     GPU vs CPU  maxdiff {d_head:.3e}  ({gpu_head_ms:.1} ms)"
    );

    println!("\nfull offline on GPU…");
    let gpu_model = Model::load_gpu(&ckpt)?;
    if let Some(d) = gpu_model.gpu_describe() {
        println!("  {d}");
    }
    let t3 = Instant::now();
    let out = gpu_model.run_offline(&samples)?;
    let gpu_off_s = t3.elapsed().as_secs_f64();
    let frames_ref = read_f32(&root.join("frames").join(format!("{stem}__offline.npy")))?;
    let (mut max_abs, mut sum_sq, mut flips) = (0.0f64, 0.0f64, 0usize);
    let ns = cpu.num_speakers();
    let n = out.num_frames.min(frames_ref.rows());
    for t in 0..n {
        for s in 0..ns {
            let got = out.logits[t * ns + s] as f64;
            let want = frames_ref.data[t * frames_ref.cols() + s];
            let d = (got - want).abs();
            sum_sq += d * d;
            max_abs = max_abs.max(d);
            let sg = 1.0 / (1.0 + (-got as f32).exp());
            let sw = 1.0 / (1.0 + (-want as f32).exp());
            if (sg > 0.5) != (sw > 0.5) {
                flips += 1;
            }
        }
    }
    let rms = (sum_sq / (n * ns) as f64).sqrt();
    println!(
        "  {n} frames in {gpu_off_s:.2}s  maxdiff {max_abs:.3e} rms {rms:.3e}  flips {flips}/{}",
        n * ns
    );

    let ok = d_enc < 2e-2 && d_head < 2e-2 && max_abs < 2e-2 && flips == 0;
    if ok {
        println!("\nwgpu encoder, head, and offline logits match the reference.");
        Ok(())
    } else {
        eprintln!("\nwgpu does NOT match (encoder {d_enc:.3e}, head {d_head:.3e}, logits {max_abs:.3e}, flips {flips})");
        std::process::exit(1);
    }
}
