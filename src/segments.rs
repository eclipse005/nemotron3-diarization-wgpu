//! Turning per-frame logits into speech segments.
//!
//! Port of `Nemotron3DiarizationProcessor.extract_speaker_dict`. The transform is
//! deliberately trivial — a threshold, then run-length changes per speaker — which
//! makes it a good end-to-end check: if the logits match, the segments must match
//! exactly, and if they do not, the *first* differing segment localises the frame.

/// One speaker's turn, in seconds, with the speaker index in arrival order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub speaker: usize,
}

impl Segment {
    /// The reference emits `{Start, End, Speaker}` with `round(x, 2)`.
    ///
    /// `serde_json` prints an `f32` as its exact `f64` expansion (`0.349999994…`);
    /// Python's `json.dumps(round(x, 2))` prints two decimals. Re-round through
    /// `f64` so the two serialisations compare equal.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "Start": hundredths(self.start),
            "End": hundredths(self.end),
            "Speaker": self.speaker,
        })
    }
}

fn hundredths(v: f32) -> f64 {
    (v as f64 * 100.0).round() / 100.0
}

/// Seconds per 10 ms frame — `hop_length / sampling_rate`.
pub fn frame_duration(hop_length: usize, sampling_rate: u32) -> f32 {
    hop_length as f32 / sampling_rate as f32
}

/// `torch.sigmoid` in fp32.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Python's `round(x, 2)`: half-to-even, not half-away-from-zero.
///
/// `f32::round` rounds `.5` up, so `round(0.125, 2)` would give `0.13` here and
/// `0.12` in the reference. With a 10 ms frame duration no tie is reachable, but the
/// two-decimal outputs are compared verbatim, so the rule is implemented rather than
/// assumed.
fn round2(v: f32) -> f32 {
    let scaled = (v as f64) * 100.0;
    let r = scaled.round();
    let r = if (scaled - scaled.trunc()).abs() == 0.5 {
        // tie: pick the even neighbour
        let t = scaled.trunc();
        if (t as i64) % 2 == 0 { t } else { t + 1.0 }
    } else {
        r
    };
    (r / 100.0) as f32
}

/// `extract_speaker_dict`: threshold the sigmoid, diff each speaker's activity, and
/// pair the `+1` and `-1` transitions into segments.
///
/// `mask` is the per-frame attention mask at the **10 ms** rate; `None` scores every
/// frame. Overlapping speech yields overlapping segments — the model does not resolve
/// turn-taking, and the reference does not either.
pub fn extract_speaker_dict(
    logits: &[f32],
    num_frames: usize,
    num_speakers: usize,
    duration: f32,
    mask: Option<&[bool]>,
    threshold: f32,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    for s in 0..num_speakers {
        let mut start: Option<usize> = None;
        // a run of activity is closed by the trailing zero the reference appends, so
        // a speaker active in the very last frame still produces a segment
        for t in 0..=num_frames {
            let active = t < num_frames
                && sigmoid(logits[t * num_speakers + s]) > threshold
                && mask.map_or(true, |m| m.get(t).copied().unwrap_or(false));
            match (active, start) {
                (true, None) => start = Some(t),
                (false, Some(s0)) => {
                    segments.push(Segment {
                        start: round2(s0 as f32 * duration),
                        end: round2(t as f32 * duration),
                        speaker: s,
                    });
                    start = None;
                }
                _ => {}
            }
        }
    }
    // `segments.sort(key=lambda seg: (seg["Start"], seg["Speaker"]))`
    segments.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.speaker.cmp(&b.speaker))
    });
    segments
}

/// Fraction of frames in which at least one speaker is active, and per-speaker counts —
/// the two summary numbers the Python baseline records next to its segments.
pub fn activity_stats(
    logits: &[f32],
    num_frames: usize,
    num_speakers: usize,
    mask: Option<&[bool]>,
    threshold: f32,
) -> (f32, Vec<usize>) {
    let mut per_speaker = vec![0usize; num_speakers];
    let mut any = 0usize;
    let mut total = 0usize;
    for t in 0..num_frames {
        if !mask.map_or(true, |m| m.get(t).copied().unwrap_or(false)) {
            continue;
        }
        total += 1;
        let mut hot = false;
        for s in 0..num_speakers {
            if sigmoid(logits[t * num_speakers + s]) > threshold {
                per_speaker[s] += 1;
                hot = true;
            }
        }
        if hot {
            any += 1;
        }
    }
    let pct = if total == 0 { 0.0 } else { any as f32 * 100.0 / total as f32 };
    (pct, per_speaker)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: usize = 2;

    fn logit(active: bool) -> f32 {
        if active { 2.0 } else { -2.0 }
    }

    #[test]
    fn runs_become_segments_in_arrival_order() {
        // frame:  0  1  2  3  4  5
        // spk 0:   .  .  X  X  .  .
        // spk 1:   .  X  .  .  X  X
        let acts = [[false, false, true, true, false, false], [false, true, false, false, true, true]];
        let mut logits = Vec::new();
        for t in 0..6 {
            for s in 0..NS {
                logits.push(logit(acts[s][t]));
            }
        }
        let segs = extract_speaker_dict(&logits, 6, NS, 0.01, None, 0.5);
        assert_eq!(
            segs,
            vec![
                Segment { start: 0.01, end: 0.02, speaker: 1 },
                Segment { start: 0.02, end: 0.04, speaker: 0 },
                Segment { start: 0.04, end: 0.06, speaker: 1 },
            ]
        );
    }

    /// A speaker still talking in the final frame must still get a segment: the
    /// reference appends a zero column before diffing, which closes the run.
    #[test]
    fn a_run_open_at_the_end_is_still_closed() {
        // frame 0 silent, frame 1 speaker 0 talking and never stopping
        let logits = [logit(false), logit(false), logit(true), logit(false)];
        let segs = extract_speaker_dict(&logits, 2, NS, 0.01, None, 0.5);
        assert_eq!(segs, vec![Segment { start: 0.01, end: 0.02, speaker: 0 }]);
    }

    #[test]
    fn the_mask_removes_frames_from_both_sides() {
        let logits: Vec<f32> = (0..6).map(|t| logit(t % 2 == 0)).collect::<Vec<_>>();
        let segs = extract_speaker_dict(
            &logits,
            6,
            1,
            0.01,
            Some(&[false, true, true, true, true, false]),
            0.5,
        );
        // frames 0 and 5 are masked out; the runs at 2 and at 4 both survive
        assert_eq!(
            segs,
            vec![
                Segment { start: 0.02, end: 0.03, speaker: 0 },
                Segment { start: 0.04, end: 0.05, speaker: 0 },
            ]
        );
    }

    #[test]
    fn round2_matches_pythons_half_to_even() {
        assert_eq!(round2(0.125), 0.12);
        assert_eq!(round2(0.135), 0.14);
        assert_eq!(round2(1.005), 1.0);
        assert_eq!(round2(28.1), 28.1);
    }

    #[test]
    fn stats_count_frames_not_segments() {
        let logits: Vec<f32> = (0..4).flat_map(|t| [logit(t < 2), logit(t == 3)]).collect();
        let (pct, per) = activity_stats(&logits, 4, NS, None, 0.5);
        assert_eq!(per, vec![2, 1]);
        assert_eq!(pct, 75.0);
    }
}
