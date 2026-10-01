#!/usr/bin/env python3
"""Dump the reference speaker cache after every offline chunk.

The Rust `head_check` compares against these: `/tmp/s<i>_{cache_e,cache_p,fifo_e}.npy`
plus `/tmp/s0_probs.npy` for the pooled step probabilities. Run it whenever the cache
diverges, to tell a scoring bug from a selection bug.

The first `_compress` call also writes `/tmp/c_{probs_in,scores,scores_boosted,
scores_strong,scores_final,flat,topk_sorted}.npy` for `compress_check`.
"""
import numpy as np, torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationSpeakerCache,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
m = AutoModelForAudioFrameClassification.from_pretrained(str(ROOT / "models" / "Nemotron-3-Diarization")).eval()
p = AutoProcessor.from_pretrained(str(ROOT / "models" / "Nemotron-3-Diarization"))
a = np.load(ROOT / "baseline" / "waveforms" / "diarization_example.npy")
inp = p(a, sampling_rate=16000)
feats, am = inp["input_features"], inp["attention_mask"]
emb = m.model.audio_tower.embedder(feats)
em = am[:, ::8].bool()
cache = Nemotron3DiarizationSpeakerCache(
    m.config.streaming_config,
    fifo_length=m.config.fifo_length,
    speaker_cache_update_period=m.config.speaker_cache_update_period,
)
_orig_compress = Nemotron3DiarizationSpeakerCache._compress
_dumped_compress = False


def _compress_and_dump(self, embeds, probs, silence_embeds):
    global _dumped_compress
    import math
    from torch import nn

    scores = self._get_frame_scores(probs)
    scores_raw = scores.clone()
    scores[:, self.speaker_cache_length :] += self.latest_frames_score_boost
    scores_boosted = scores.clone()
    scores = self._boost_scores(scores, self.num_strong_boosted_frames, boost=-2.0 * math.log(0.5))
    scores_strong = scores.clone()
    scores = self._boost_scores(scores, self.num_weak_boosted_frames, boost=-math.log(0.5))
    scores_final = scores.clone()
    scores = nn.functional.pad(scores, (0, 0, 0, self.num_silence_frames), value=float("inf"))
    batch_size, num_frames, num_speakers = probs.shape
    num_scored_frames = num_frames + self.num_silence_frames
    flat_scores = scores.transpose(1, 2).reshape(batch_size, -1)
    topk_scores, topk_indices = torch.topk(flat_scores, self.speaker_cache_length, dim=1, sorted=False)
    sentinel = num_scored_frames * num_speakers
    topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), sentinel)
    topk_sorted, _ = torch.sort(topk_indices, dim=1)
    if not _dumped_compress:
        np.save("/tmp/c_probs_in.npy", probs[0].detach().cpu().numpy())
        np.save("/tmp/c_scores.npy", scores_raw[0].detach().cpu().numpy())
        np.save("/tmp/c_scores_boosted.npy", scores_boosted[0].detach().cpu().numpy())
        np.save("/tmp/c_scores_strong.npy", scores_strong[0].detach().cpu().numpy())
        np.save("/tmp/c_scores_final.npy", scores_final[0].detach().cpu().numpy())
        np.save("/tmp/c_flat.npy", flat_scores[0].detach().cpu().numpy())
        np.save("/tmp/c_topk_sorted.npy", topk_sorted[0].detach().cpu().numpy().astype(np.float32))
        print("dumped compress intermediates", probs.shape, "->", self.speaker_cache_length)
        _dumped_compress = True
    return _orig_compress(self, embeds, probs, silence_embeds)


Nemotron3DiarizationSpeakerCache._compress = _compress_and_dump

n_ce = emb.shape[1]
with torch.inference_mode():
    for i, start in enumerate(range(0, n_ce, m.config.chunk_length)):
        end = min(start + m.config.chunk_length, n_ce)
        ce = emb[:, start : min(end + m.config.chunk_right_context, n_ce)]
        cached = cache.get_embeds(ce)
        win = torch.cat([cached, ce], dim=1)
        sm = torch.cat([em.new_ones(1, cached.shape[1]), em[:, start : start + ce.shape[1]]], dim=1)
        out = m.model(inputs_embeds=win, attention_mask=sm, position_ids=torch.arange(win.shape[1])[None])
        lg = m.classifier(out.last_hidden_state)
        if i == 0:
            np.save("/tmp/s0_probs.npy", cache._pool_probs(lg, sm)[0].numpy())
        cache.update(win, lg, m.silence_embeds, end - start, mask=sm)
        np.save(f"/tmp/s{i}_cache_e.npy", cache.embeds[0, : cache.num_cache_frames].numpy())
        np.save(f"/tmp/s{i}_cache_p.npy", cache.probs[0, : cache.num_cache_frames].numpy())
        np.save(f"/tmp/s{i}_fifo_e.npy", cache.fifo[0, : cache.num_fifo_frames].numpy())
        print(i, "cache", cache.num_cache_frames, "fifo", cache.num_fifo_frames, "compressed", cache.is_compressed)
