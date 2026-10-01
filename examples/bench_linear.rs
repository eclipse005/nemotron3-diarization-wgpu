use rayon::prelude::*;
use std::time::Instant;

fn tile4(x: &[f32], rows: usize, cols: usize, w: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    y.par_chunks_mut(out).enumerate().for_each(|(r, yrow)| {
        let xrow = &x[r * cols..(r + 1) * cols];
        let mut o = 0;
        while o + 4 <= out {
            let w0 = &w[o*cols..o*cols+cols];
            let w1 = &w[(o+1)*cols..(o+1)*cols+cols];
            let w2 = &w[(o+2)*cols..(o+2)*cols+cols];
            let w3 = &w[(o+3)*cols..(o+3)*cols+cols];
            let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for i in 0..cols {
                let v = xrow[i];
                a0 += v*w0[i]; a1 += v*w1[i]; a2 += v*w2[i]; a3 += v*w3[i];
            }
            yrow[o]=a0; yrow[o+1]=a1; yrow[o+2]=a2; yrow[o+3]=a3;
            o += 4;
        }
        while o < out {
            let wrow = &w[o*cols..(o+1)*cols];
            let mut a = 0.0f32;
            for i in 0..cols { a += xrow[i]*wrow[i]; }
            yrow[o] = a; o += 1;
        }
    });
    y
}

fn tile8(x: &[f32], rows: usize, cols: usize, w: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    y.par_chunks_mut(out).enumerate().for_each(|(r, yrow)| {
        let xrow = &x[r * cols..(r + 1) * cols];
        let mut o = 0;
        while o + 8 <= out {
            let (mut a0,mut a1,mut a2,mut a3) = (0.0f32,0.0f32,0.0f32,0.0f32);
            let (mut b0,mut b1,mut b2,mut b3) = (0.0f32,0.0f32,0.0f32,0.0f32);
            for i in 0..cols {
                let v = xrow[i];
                a0 += v*w[o*cols+i];       a1 += v*w[(o+1)*cols+i];
                a2 += v*w[(o+2)*cols+i];   a3 += v*w[(o+3)*cols+i];
                b0 += v*w[(o+4)*cols+i];   b1 += v*w[(o+5)*cols+i];
                b2 += v*w[(o+6)*cols+i];   b3 += v*w[(o+7)*cols+i];
            }
            yrow[o]=a0; yrow[o+1]=a1; yrow[o+2]=a2; yrow[o+3]=a3;
            yrow[o+4]=b0; yrow[o+5]=b1; yrow[o+6]=b2; yrow[o+7]=b3;
            o += 8;
        }
        while o < out {
            let wrow = &w[o*cols..(o+1)*cols];
            let mut a = 0.0f32;
            for i in 0..cols { a += xrow[i]*wrow[i]; }
            yrow[o] = a; o += 1;
        }
    });
    y
}

fn main() {
    for (rows, cols, out) in [(380usize, 512usize, 512usize), (277, 512, 2048), (13, 512, 512), (277, 512, 512)] {
        let x: Vec<f32> = (0..rows*cols).map(|i| (i%7) as f32*0.01).collect();
        let w: Vec<f32> = (0..out*cols).map(|i| (i%5) as f32*0.01).collect();
        let n = 10;
        let a = tile4(&x, rows, cols, &w, out);
        println!("--- {rows}x{cols}x{out}");
        for (name, f) in [("tile4", &tile4 as &dyn Fn(&[f32],usize,usize,&[f32],usize)->Vec<f32>), ("tile8", &tile8)] {
            let t = Instant::now();
            for _ in 0..n { std::hint::black_box(f(&x, rows, cols, &w, out)); }
            let e = t.elapsed().as_secs_f64();
            let r = f(&x, rows, cols, &w, out);
            let d = (0..a.len()).map(|i| (a[i]-r[i]).abs()).fold(0f32, f32::max);
            println!("  {name:<6} {:.3}s  {:.2} GMAC/s  diff {d:e}", e, (rows*cols*out*n) as f64/e/1e9);
        }
    }
}
