import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "make_qualitative_comparison.py"


class TestQualitativeComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError(
                "Expected public qualitative script does not exist yet: "
                f"{SCRIPT}"
            )

        spec = importlib.util.spec_from_file_location(
            "make_qualitative_comparison", SCRIPT
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_dataset_specs_match_release_models(self):
        specs = self.mod.DATASET_SPECS

        self.assertEqual(specs["potsdam"]["num_classes"], 6)
        self.assertEqual(tuple(specs["potsdam"]["delta_hidden_dims"]), ())

        self.assertEqual(specs["vaihingen"]["num_classes"], 6)
        self.assertEqual(
            tuple(specs["vaihingen"]["delta_hidden_dims"]),
            (128, 128),
        )

        self.assertEqual(specs["loveda"]["num_classes"], 7)
        self.assertEqual(
            tuple(specs["loveda"]["delta_hidden_dims"]),
            (128, 64),
        )

    def test_extract_logits_supports_public_model_output(self):
        logits = torch.randn(1, 6, 16, 16)
        out = {"seg_logits": logits}

        got = self.mod.extract_logits(out)

        self.assertIs(got, logits)

    def test_error_map_marks_only_wrong_valid_pixels(self):
        pred = np.array(
            [
                [0, 1],
                [2, 1],
            ],
            dtype=np.int64,
        )
        gt = np.array(
            [
                [0, 0],
                [2, 255],
            ],
            dtype=np.int64,
        )

        rgb = self.mod.make_error_map(pred, gt, ignore_index=255)

        self.assertEqual(rgb.shape, (2, 2, 3))

        # Correct valid pixels -> white.
        np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(rgb[1, 0], [255, 255, 255])

        # Wrong valid pixel -> red.
        np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])

        # Ignore pixel -> gray.
        np.testing.assert_array_equal(rgb[1, 1], [220, 220, 220])

    def test_public_script_has_no_autodl_absolute_paths(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("/root/autodl-tmp", src)

    def test_cli_exposes_release_visualization_arguments(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)

        help_text = proc.stdout
        for flag in [
            "--dataset",
            "--config",
            "--checkpoint",
            "--split",
            "--num-cases",
            "--output-dir",
            "--baseline-checkpoint",
            "--loss-topk-checkpoint",
        ]:
            self.assertIn(flag, help_text)


if __name__ == "__main__":
    unittest.main()
