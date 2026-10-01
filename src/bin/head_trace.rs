//! Stage-by-stage diff of the classification head.
//!
//! `head_check` says the head is wrong; this says *where*. It reads a saved encoder
//! output plus the reference's own intermediate tensors and prints the max diff of
//! each stage, so a bug in the sub-pixel conv cannot hide behind a bug in `out_proj`.
//!
//! The reference tensors come from a PyTorch dump:
//!
//! ```python
//! x = torch.from_numpy(np.load("in.npy"))[None]
//! up = model.model.upsampler(model.model.proj(x))
//! r1 = model.classifier.act_fn(up); d = model.classifier.dense(r1)
//! r2 = model.classifier.act_fn(d); lg = model.classifier.out_proj(r2)
//! ```
//!
//! Usage:  head_trace <prefix>   (reads <prefix>_{in,proj,relu1,dense,relu2,logits}.npy)
//!                          [model-dir]
use std::path::PathBuf;
use nemotron3_diarization_wgpu::{read_f32, Model};

fn main() -> anyhow::Result<()> {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "/tmp/head".into());
    let ckpt = PathBuf::from(
        std::env::args().nth(2).unwrap_or_else(|| "../models/Nemotron-3-Diarization".into()),
    );
    let model = Model::load(&ckpt)?;
    let rd = |name: &str| -> anyhow::Result<Vec<f64>> {
        Ok(read_f32(&PathBuf::from(format!("{prefix}_{name}.npy")))?.data)
    };
    let x: Vec<f32> = rd("in")?.into_iter().map(|v| v as f32).collect();
    let n = x.len() / 512;
    let t = model.head.trace(&x, n);
    let show = |name: &str, got: &[f32], want: &[f64], rows: usize, cols: usize| {
        let d: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b).abs()).fold(0.0, f64::max);
        println!("{name:<12} maxdiff {d:.3e}");
        println!("   ours[0..4] {:?}", &got[..4]);
        println!("   ref [0..4] {:?}", &want[..4]);
        let _ = (rows, cols);
    };
    let (rp, ru, rl) = (rd("proj")?, rd("upsampled")?, rd("logits")?);
    show("proj", &t.projected, &rp, n, 192);
    show("upsampled", &t.upsampled, &ru, n*8, 192);
    let (rd1, rdd, rd2) = (rd("relu1")?, rd("dense")?, rd("relu2")?);
    show("relu1", &t.r1, &rd1, n*8, 192);
    show("dense", &t.dense, &rdd, n*8, 192);
    show("relu2", &t.r2, &rd2, n*8, 192);
    show("logits", &t.logits, &rl, n*8, 8);
    let range = |v: &[f32]| (v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("ours dense range {:?}", range(&t.dense));
    println!("ref  dense range {:?}", (rdd.iter().cloned().fold(f64::INFINITY, f64::min), rdd.iter().cloned().fold(f64::NEG_INFINITY, f64::max)));
    Ok(())
}
