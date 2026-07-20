"""Pure contract checks for the packed-audio producer helpers.

These are deliberately stdlib-only test cases; they do not decode audio or
touch datasets.  The module under test still requires the normal project audio
dependencies when tests are run on the remote machine.
"""

import unittest

from scripts.prepare_audio_shards import (
    PCM16_STORAGE_PEAK,
    PackingError,
    _amplitude_restore_gain,
    _current_preprocessing_contract,
    _legacy_preprocessing_contract,
    _optional_amplitude_metadata,
    _optional_restore_gain,
    _same_amplitude_scaling,
)


class AmplitudeScalingHelpersTest(unittest.TestCase):
    def test_no_gain_below_storage_headroom(self) -> None:
        self.assertEqual(_amplitude_restore_gain(PCM16_STORAGE_PEAK), 1.0)

    def test_overrange_peak_has_reversible_headroom(self) -> None:
        peak = 1.0140847
        gain = _amplitude_restore_gain(peak)
        self.assertGreater(gain, 1.0)
        self.assertLess(peak / gain, PCM16_STORAGE_PEAK)

    def test_legacy_gain_is_optional(self) -> None:
        self.assertEqual(_optional_restore_gain(None, "gain"), 1.0)

    def test_optional_amplitude_metadata_is_atomic_and_reversible(self) -> None:
        self.assertEqual(_optional_amplitude_metadata({}, "metadata"), (1.0, None, None))
        self.assertEqual(
            _optional_amplitude_metadata(
                {"amplitude_restore_gain": 1.25, "canonical_peak": 1.1, "storage_peak": 0.88},
                "metadata",
            ),
            (1.25, 1.1, 0.88),
        )
        with self.assertRaisesRegex(PackingError, "present together"):
            _optional_amplitude_metadata({"amplitude_restore_gain": 1.25}, "metadata")
        with self.assertRaisesRegex(PackingError, "finite numeric"):
            _optional_amplitude_metadata(
                {"amplitude_restore_gain": None, "canonical_peak": 1.1, "storage_peak": 0.88},
                "metadata",
            )
        with self.assertRaisesRegex(PackingError, "inconsistent"):
            _optional_amplitude_metadata(
                {"amplitude_restore_gain": 2.0, "canonical_peak": 1.1, "storage_peak": 0.88},
                "metadata",
            )

    def test_legacy_and_current_contracts_are_distinct(self) -> None:
        self.assertNotEqual(_legacy_preprocessing_contract(), _current_preprocessing_contract())

    def test_scaling_stat_comparison_checks_all_aggregate_fields(self) -> None:
        expected = {
            "scaled_sample_count": 2,
            "max_restore_gain": 1.25,
            "max_canonical_peak": 1.2,
        }
        self.assertTrue(_same_amplitude_scaling(expected, dict(expected)))
        self.assertFalse(
            _same_amplitude_scaling(expected, {**expected, "scaled_sample_count": 1})
        )


if __name__ == "__main__":
    unittest.main()
