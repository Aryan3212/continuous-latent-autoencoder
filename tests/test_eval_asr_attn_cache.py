from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval.eval_asr_attn import (
    AttnDecoderHead,
    _CachedFeatureDataset,
    _cache_clae_features,
    _collate_cached_features,
    _reset_probe_seed,
)


class AttentionAsrCacheTest(unittest.TestCase):
    def test_clae_cache_is_variable_length_reusable_and_source_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "audio.wav"
            source.write_bytes(b"stable source identity")
            row = {
                "audio_filepath": str(source),
                "duration": 1.0,
                "text": "test",
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            cache_dir = root / "cache"
            lm = SimpleNamespace(
                cfg=SimpleNamespace(data=SimpleNamespace(sample_rate=16000))
            )

            def fake_features(*args, **kwargs):
                self.assertEqual(kwargs["start_index"], 0)
                yield (
                    torch.tensor(
                        [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]]
                    ),
                    torch.tensor([2]),
                    [row],
                )

            with patch(
                "eval.eval_asr_attn.iter_frame_features",
                side_effect=fake_features,
            ):
                cache, _, _ = _cache_clae_features(
                    str(manifest),
                    text_key="text",
                    max_samples=0,
                    segment_seconds=1.0,
                    max_utt_seconds=1.0,
                    lm=lm,
                    chunk_seconds=None,
                    source="encoder",
                    mel_hop=320,
                    extraction_batch_size=4,
                    num_workers=0,
                    extractor_identity={"checkpoint": "test"},
                    cache_dir=cache_dir,
                    split_name="train",
                    n_filtered=0,
                    n_unknown_duration=0,
                )

            frames, text = _CachedFeatureDataset(cache)[0]
            self.assertEqual(text, "test")
            self.assertEqual(tuple(frames.shape), (2, 2))
            self.assertTrue(torch.equal(
                frames,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ))

            with patch(
                "eval.eval_asr_attn.iter_frame_features",
                side_effect=AssertionError("complete cache should not re-extract"),
            ):
                reused, _, _ = _cache_clae_features(
                    str(manifest),
                    text_key="text",
                    max_samples=0,
                    segment_seconds=1.0,
                    max_utt_seconds=1.0,
                    lm=lm,
                    chunk_seconds=None,
                    source="encoder",
                    mel_hop=320,
                    extraction_batch_size=4,
                    num_workers=0,
                    extractor_identity={"checkpoint": "test"},
                    cache_dir=cache_dir,
                    split_name="train",
                    n_filtered=0,
                    n_unknown_duration=0,
                )
            self.assertEqual(len(reused.records), 1)

            source.write_bytes(b"changed source identity")
            with self.assertRaisesRegex(RuntimeError, "Source audio changed"):
                _cache_clae_features(
                    str(manifest),
                    text_key="text",
                    max_samples=0,
                    segment_seconds=1.0,
                    max_utt_seconds=1.0,
                    lm=lm,
                    chunk_seconds=None,
                    source="encoder",
                    mel_hop=320,
                    extraction_batch_size=4,
                    num_workers=0,
                    extractor_identity={"checkpoint": "test"},
                    cache_dir=cache_dir,
                    split_name="train",
                    n_filtered=0,
                    n_unknown_duration=0,
                )

    def test_collate_pads_only_current_batch(self) -> None:
        padded, lengths, texts = _collate_cached_features([
            (torch.ones(2, 3), "short"),
            (torch.ones(5, 3), "long"),
        ])
        self.assertEqual(tuple(padded.shape), (2, 5, 3))
        self.assertEqual(lengths.tolist(), [2, 5])
        self.assertEqual(texts, ["short", "long"])

    def test_probe_head_seed_is_independent_of_prior_rng_use(self) -> None:
        _reset_probe_seed(17)
        first = AttnDecoderHead(
            feat_dim=4,
            vocab_size=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_ff=16,
        )
        first_state = {
            key: value.detach().clone()
            for key, value in first.state_dict().items()
        }

        _ = torch.rand(1000)
        _reset_probe_seed(17)
        second = AttnDecoderHead(
            feat_dim=4,
            vocab_size=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_ff=16,
        )
        self.assertTrue(all(
            torch.equal(value, second.state_dict()[key])
            for key, value in first_state.items()
        ))


if __name__ == "__main__":
    unittest.main()
