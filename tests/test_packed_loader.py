"""Focused invariants for the packed streaming loader (not run on coding hosts)."""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import train
from data_loading import (
    PackedShard,
    PackedShardError,
    PackedTarDataset,
    _packed_buffer_requires_eviction,
    _packed_safe_relative_path,
    packed_epoch_assignment,
)
from schema import DataCfg


def _shards(counts: list[int]) -> list[PackedShard]:
    return [
        PackedShard(pathlib.Path(f"/tmp/train-{index:06d}.tar"), f"shards/train-{index:06d}.tar", count)
        for index, count in enumerate(counts)
    ]


class PackedLoaderTests(unittest.TestCase):
    def test_scheduler_state_restores_only_for_matching_schedule_inputs(self) -> None:
        saved = train._ScheduleInputs(lr=1.0e-3, warmup_steps=5_000, total_steps=100_000, min_lr_ratio=0.0)
        self.assertTrue(train._schedule_inputs_match(saved, saved))
        self.assertFalse(
            train._schedule_inputs_match(
                saved,
                train._ScheduleInputs(
                    lr=5.0e-4,
                    warmup_steps=5_000,
                    total_steps=100_000,
                    min_lr_ratio=0.0,
                ),
            )
        )

    def test_epoch_assignment_is_unique_and_batch_aligned(self) -> None:
        shards = _shards([120, 100, 90, 80, 70, 60])
        groups, quota = packed_epoch_assignment(
            shards, seed=42, epoch=3, total_consumers=3, batch_size=10
        )
        self.assertEqual(quota % 10, 0)
        self.assertGreaterEqual(quota, 10)
        flattened = [shard.relative_path for group in groups for shard in group]
        self.assertEqual(sorted(flattened), sorted(shard.relative_path for shard in shards))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_epoch_assignment_rejects_too_few_shards_or_a_partial_batch(self) -> None:
        with self.assertRaisesRegex(PackedShardError, "at least one shard"):
            packed_epoch_assignment(_shards([100]), seed=0, epoch=0, total_consumers=2, batch_size=10)
        with self.assertRaisesRegex(PackedShardError, "cannot form one batch"):
            packed_epoch_assignment(_shards([9, 100]), seed=0, epoch=0, total_consumers=2, batch_size=10)

    def test_member_path_and_buffer_budget_safety_invariants(self) -> None:
        with self.assertRaisesRegex(PackedShardError, "unsafe"):
            _packed_safe_relative_path("../escape.tar", "shard")
        with self.assertRaisesRegex(PackedShardError, "unexpected"):
            PackedTarDataset._safe_member_name("samples/unsafe.flac", ".flac")
        self.assertFalse(
            _packed_buffer_requires_eviction(item_count=1, buffered_bytes=513, byte_budget=512)
        )
        self.assertTrue(
            _packed_buffer_requires_eviction(item_count=2, buffered_bytes=513, byte_budget=512)
        )

    def test_tar_backend_requires_a_descriptor_but_files_stays_compatible(self) -> None:
        self.assertEqual(DataCfg(train_manifest="staging/manifests/train.jsonl").backend, "files")
        with self.assertRaisesRegex(ValueError, "shard_manifest"):
            DataCfg(train_manifest="staging/manifests/train.jsonl", backend="tar")

    def test_legacy_checkpoint_has_scheduler_and_data_epoch_fallbacks(self) -> None:
        class Loadable:
            def load_state_dict(self, _: object, strict: bool = True) -> None:
                return None

        class Scaler:
            def is_enabled(self) -> bool:
                return False

            def load_state_dict(self, _: object) -> None:
                raise AssertionError("disabled legacy scaler must not load state")

        with mock.patch.object(
            train.torch,
            "load",
            return_value={"step": 40_000, "model": {}, "optimizer": {}},
        ):
            state = train._restore_checkpoint(
                "legacy.pt",
                model=Loadable(),
                optimizer=Loadable(),
                scaler=Scaler(),
                disc=None,
                optimizer_d=None,
                scaler_d=None,
            )
        self.assertEqual(state, (40_000, None, None, None))
