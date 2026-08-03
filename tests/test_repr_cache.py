from __future__ import annotations

import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eval import repr_bench


class RepresentationCacheTest(unittest.TestCase):
    def test_cache_hit_uses_current_speaker_labels(self) -> None:
        waveform = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32)
        first = [repr_bench.Utterance(id="utt", speaker="old", wav=waveform)]
        relabelled = [
            repr_bench.Utterance(id="utt", speaker="corrected", wav=waveform)
        ]
        embedder = SimpleNamespace(
            fn=lambda _wav: np.asarray([[1.0, 2.0]], dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            repr_bench,
            "EMB_DIR",
            pathlib.Path(temp_dir),
        ), patch.object(repr_bench, "build_embedder", return_value=embedder):
            repr_bench.extract("wavlm", first, pool="mean")

            with patch.object(
                repr_bench,
                "build_embedder",
                side_effect=AssertionError("cache hit should not re-extract"),
            ):
                cached = repr_bench.extract("wavlm", relabelled, pool="mean")

        self.assertEqual(cached["speakers"].tolist(), ["corrected"])


if __name__ == "__main__":
    unittest.main()
