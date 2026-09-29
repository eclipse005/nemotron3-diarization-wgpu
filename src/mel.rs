//! The audio front-end: preemphasis → STFT → power → slaney mel → natural log.
//!
//! This is a bit-for-bit port of
//! `transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py`.
//! It is the part of the pipeline most likely to diverge, because the reference mixes
//! three conventions that are easy to get individually right and jointly wrong:
//!
//! 1. **preemphasis is applied to the raw waveform**, before the STFT — not to the
//!    mel output. `y[0] = x[0]`, `y[i] = x[i] - 0.97 * x[i-1]`, and everything past
//!    the true audio length is zeroed.
//! 2. **the mel filterbank is librosa's**, with `norm="slaney"` and the slaney mel
//!    scale. The HF `mel_filter_bank` helper defaults to HTK and float64, and the
//!    reference explicitly avoids it (there is a commented-out block in the source
//!    saying they switched to librosa *because* the two disagree numerically).
//! 3. **natural log with a `2^-24` guard** — not log10, not dB. The floor is
//!    `log(2^-24) ≈ -16.63`.
//!
//! The window is `hann_window(win_length, periodic=False)`, and `torch.stft` centres a
//! window shorter than `n_fft` inside the FFT frame — so the effective 512-sample
//! window is 56 zeros, then the 400-tap hann, then 56 zeros.
//!
//! Frame counts (this is how the processor decides whether a chunk is well formed):
//!
//! | `center` | valid frames |
//! |---|---|
//! | `true`  (offline, first streaming chunk) | `floor(L / hop)` |
//! | `false` (later streaming chunks)          | `floor((L - n_fft) / hop) + 1` |

use rustfft::{num_complex::Complex, FftPlanner};

use crate::config::FeatureExtractorConfig;

/// `2^-24`, the guard added before the logarithm. `log(2^-24) ≈ -16.63` is the
/// floor of every feature value, which is a quick way to spot a wrong constant.
pub const LOG_ZERO_GUARD_VALUE: f32 = 5.960_464_5e-8;

/// `torch.hann_window(n, periodic=False)` — the symmetric window, **not** the
/// periodic one. The denominator is `n - 1`, not `n`: torch's `periodic=False` is
/// `0.5 * (1 - cos(2*pi*i / (n - 1)))`, while the periodic form divides by `n`.
/// Getting this wrong shifts the whole taper by half a sample and shows up as a
/// ~1e-2 error spread across every mel bin.
fn hann_periodic_false(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / denom).cos()))
        .collect()
}

/// librosa's slaney mel scale: the log segment's step per 27 mels above 1 kHz.
fn logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// librosa's `hz_to_mel(..., htk=False)` — the slaney scale.
fn hz_to_mel(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    if f < MIN_LOG_HZ {
        f / F_SP
    } else {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / logstep()
    }
}

/// librosa's `mel_to_hz(..., htk=False)`.
fn mel_to_hz(m: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    if m < MIN_LOG_MEL {
        F_SP * m
    } else {
        MIN_LOG_HZ * (logstep() * (m - MIN_LOG_MEL)).exp()
    }
}

/// librosa's `filters.mel(sr, n_fft, n_mels, fmin, fmax, norm="slaney")`, returned
/// row-major as `(n_mels, n_fft / 2 + 1)`.
pub fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // `np.fft.rfftfreq(n_fft, 1 / sr)` — bin k sits at k * sr / n_fft.
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    // `n_mels + 2` anchor points, evenly spaced in mel, converted back to Hz.
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let t = i as f64 / (n_mels + 1) as f64;
            mel_to_hz(min_mel + t * (max_mel - min_mel))
        })
        .collect();

    // slaney normalisation is done in Hz, not in mel:
    // `enorm = 2 / (mel_f[2:] - mel_f[:-2])`
    let enorm: Vec<f64> = (0..n_mels)
        .map(|i| 2.0 / (mel_f[i + 2] - mel_f[i]))
        .collect();

    let mut weights = vec![0.0f32; n_mels * n_freqs];
    for i in 0..n_mels {
        let lower_den = mel_f[i + 1] - mel_f[i];
        let upper_den = mel_f[i + 2] - mel_f[i + 1];
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[i] - freq) / lower_den;
            let upper = (mel_f[i + 2] - freq) / upper_den;
            let v = lower.min(upper).max(0.0);
            weights[i * n_freqs + k] = (v * enorm[i]) as f32;
        }
    }
    weights
}

/// Apply preemphasis in place. `x[0]` is left alone, `x[1..]` becomes
/// `x[i] - preemphasis * x[i-1]`, and nothing past `len` is touched.
pub fn apply_preemphasis(samples: &mut [f32], preemphasis: f32) {
    for i in (1..samples.len()).rev() {
        samples[i] -= preemphasis * samples[i - 1];
    }
}

/// How the STFT window is arranged inside the FFT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// `torch.stft(center=True)`: `n_fft / 2` zeros on each side.
    Centered,
    /// `torch.stft(center=False)`: no padding at all.
    Uncentered,
}

/// Valid mel frames produced for `n_samples` under the given padding.
pub fn frame_count(n_samples: usize, fe: &FeatureExtractorConfig, padding: Padding) -> usize {
    match padding {
        Padding::Centered => n_samples / fe.hop_length,
        Padding::Uncentered => {
            if n_samples < fe.n_fft {
                0
            } else {
                (n_samples - fe.n_fft) / fe.hop_length + 1
            }
        }
    }
}

/// Compute log-mel features, returning `(n_frames, n_mels)`, row-major.
///
/// `features_lengths` — the number of *valid* frames — is the return value of
/// [`frame_count`]; the reference masks anything past it, and so must the caller.
pub fn log_mel(
    samples: &[f32],
    fe: &FeatureExtractorConfig,
    padding: Padding,
    filters: &[f32],
) -> Vec<f32> {
    let n_fft = fe.n_fft;
    let hop = fe.hop_length;
    let n_mels = fe.feature_size;
    let n_freqs = n_fft / 2 + 1;

    let n_frames = frame_count(samples.len(), fe, padding);
    if n_frames == 0 {
        return Vec::new();
    }

    // Order matters: the reference preemphasises the waveform *first*, on a buffer
    // whose tail past the true length is exactly zero, and only then does `torch.stft`
    // add the centre padding. Padding first would make the right-hand zero run decay as
    // `-0.97 * x[i-1]` instead of staying 0, which perturbs the last frames of a chunk.
    let mut emphasised = samples.to_vec();
    if fe.preemphasis != 0.0 {
        apply_preemphasis(&mut emphasised, fe.preemphasis);
    }

    let padded: Vec<f32> = match padding {
        Padding::Centered => {
            let mut p = Vec::with_capacity(emphasised.len() + n_fft);
            p.resize(n_fft / 2, 0.0);
            p.extend_from_slice(&emphasised);
            p.resize(p.len() + n_fft / 2, 0.0);
            p
        }
        Padding::Uncentered => emphasised,
    };

    // `torch.stft` pads a `win_length` window into an `n_fft` frame by centring it.
    let win = hann_periodic_false(fe.win_length);
    let left = (n_fft.saturating_sub(fe.win_length)) / 2;
    let mut frame_window = vec![0.0f32; n_fft];
    frame_window[left..left + fe.win_length].copy_from_slice(&win);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    let mut out = vec![0.0f32; n_frames * n_mels];
    for t in 0..n_frames {
        let start = t * hop;
        for j in 0..n_fft {
            buf[j] = Complex::new(padded[start + j] * frame_window[j], 0.0);
        }
        fft.process(&mut buf);
        // power spectrum
        for k in 0..n_freqs {
            let (re, im) = (buf[k].re, buf[k].im);
            let power = re * re + im * im;
            for m in 0..n_mels {
                out[t * n_mels + m] += filters[m * n_freqs + k] * power;
            }
        }
    }

    for v in out.iter_mut() {
        *v = (*v + LOG_ZERO_GUARD_VALUE).ln();
    }
    out
}

/// Convenience: build the filterbank for a feature-extractor config.
pub fn mel_filters(fe: &FeatureExtractorConfig) -> Vec<f32> {
    slaney_mel_filterbank(
        fe.feature_size,
        fe.n_fft,
        fe.sampling_rate,
        0.0,
        fe.sampling_rate as f64 / 2.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_fe() -> FeatureExtractorConfig {
        FeatureExtractorConfig {
            feature_size: 128,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            sampling_rate: 16000,
            preemphasis: 0.97,
        }
    }

    #[test]
    fn log_guard_is_the_expected_floor() {
        assert!(
            (LOG_ZERO_GUARD_VALUE.ln() - (-16.635_532)).abs() < 1e-4,
            "log(2^-24) should be about -16.6355, got {}",
            LOG_ZERO_GUARD_VALUE.ln()
        );
    }

    #[test]
    fn periodic_false_hann_is_symmetric_and_zero_at_both_ends() {
        let w = hann_periodic_false(400);
        // `periodic=False` divides by n-1, so *both* endpoints are exactly zero.
        // The periodic form (denominator n) leaves w[399] at ~6e-5 — catching that
        // is the whole point of this test.
        assert!(w[0].abs() < 1e-6, "w[0] should be ~0, got {}", w[0]);
        assert!(
            w[399].abs() < 1e-6,
            "w[n-1] must be ~0 too, got {} — is the denominator n or n-1?",
            w[399]
        );
        for i in 0..200 {
            assert!((w[i] - w[399 - i]).abs() < 1e-6);
        }
        // peak at the centre
        assert!(w[199] > 0.999 && w[199] < 1.001, "peak {} at i=199", w[199]);
    }

    #[test]
    fn mel_filterbank_shape_and_coverage() {
        let f = slaney_mel_filterbank(128, 512, 16000, 0.0, 8000.0);
        assert_eq!(f.len(), 128 * 257);
        // every row must have some weight
        for m in 0..128 {
            assert!(f[m * 257..(m + 1) * 257].iter().any(|&v| v > 0.0));
        }
        // The 128th triangular filter spans ~7616..8000 Hz, i.e. bins 244..256.
        // Its value *at* bin 256 is exactly zero by construction — the upper edge of a
        // triangle — so check the peak sits just below Nyquist instead.
        let top = &f[127 * 257..128 * 257];
        let peak_bin = top
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!((244..=256).contains(&peak_bin), "top mel peaks at {peak_bin}");
        assert!(top[256].abs() < 1e-12, "bin 256 is the filter edge, must be 0");
        // ...and the lowest filter sits at DC. Bin 0 is exactly the triangle's lower
        // edge, so it is 0 too; bin 1 (31.25 Hz) is inside the ~0..47 Hz first filter.
        assert!(f[1] > 0.0, "mel 0 must cover DC (bin 1)");
    }

    #[test]
    fn frame_counts_match_the_processor() {
        let fe = test_fe();
        // low_latency first chunk: 104 mel frames centred
        assert_eq!(frame_count(16_680, &fe, Padding::Centered), 104);
        // later chunk: 17040 samples uncentred must give exactly 104
        assert_eq!(frame_count(17_040, &fe, Padding::Uncentered), 104);
        // offline whole-file rule
        assert_eq!(frame_count(16_000, &fe, Padding::Centered), 100);
    }

    #[test]
    fn features_are_finite_and_bounded_below() {
        let fe = test_fe();
        let filters = mel_filters(&fe);
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let f = log_mel(&samples, &fe, Padding::Centered, &filters);
        assert_eq!(f.len(), 100 * 128);
        assert!(f.iter().all(|v| v.is_finite()));
        // log(2^-24) is the floor for a zero-energy band
        assert!(f.iter().all(|&v| v >= LOG_ZERO_GUARD_VALUE.ln() - 1e-3));
    }
}
