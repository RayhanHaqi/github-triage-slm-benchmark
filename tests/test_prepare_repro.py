"""Cheap regression tests: config regexes/parsers + frozen split reproduction.

Run: python -m unittest discover -s tests
The reproduction test needs the frozen reference data; it skips cleanly when
/home/tilakoid/github-triage-data is absent.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specialist import cli  # noqa: E402
from specialist.model import parse_prediction, strict_prediction  # noqa: E402
from specialist.prepare import prepare  # noqa: E402

FROZEN_DIR = Path("/home/tilakoid/github-triage-data")
CONFIG_PATH = ROOT / "configs" / "vscode-bug-feature.yaml"
LABELS = ["bug", "feature-request"]


class ParserTest(unittest.TestCase):
    def test_strict_prediction(self):
        self.assertEqual(strict_prediction(" bug \n", LABELS), "bug")
        self.assertEqual(strict_prediction("feature-request", LABELS), "feature-request")
        self.assertIsNone(strict_prediction("This is a bug.", LABELS))

    def test_semantic_prediction(self):
        self.assertEqual(parse_prediction("This is a bug.", LABELS), "bug")
        self.assertEqual(parse_prediction("Feature Request", LABELS), "feature-request")
        self.assertIsNone(parse_prediction("feature_request", LABELS))  # frozen parser: [\s-]+ only
        self.assertIsNone(parse_prediction("bug or feature-request", LABELS))
        self.assertIsNone(parse_prediction("nonsense", LABELS))


@unittest.skipUnless(FROZEN_DIR.is_dir(), "frozen reference data not available")
class FrozenReproductionTest(unittest.TestCase):
    def test_prepare_reproduces_frozen_splits_byte_for_byte(self):
        config = cli.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            raw_dir = workspace / "raw"
            raw_dir.mkdir()
            for class_name, spec in config["sources"].items():
                shutil.copyfile(spec["file"], raw_dir / f"{class_name}.json")

            prepare(config, workspace)

            for name in ("train", "val", "test"):
                self.assertEqual(
                    (workspace / f"{name}.jsonl").read_bytes(),
                    (FROZEN_DIR / f"{name}.jsonl").read_bytes(),
                    f"{name}.jsonl differs from frozen reference",
                )

            stats = json.loads((workspace / "dataset_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(
                stats["split"]["sizes"],
                {"train": 1593, "val": 200, "test": 200},
            )


if __name__ == "__main__":
    unittest.main()
