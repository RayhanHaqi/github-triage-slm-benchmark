"""Cheap regression tests: parser/config regexes, pinned prepare, legacy repro.

Run: python -m unittest discover -s tests
Pinned dataset tests mock the network. The legacy reproduction test needs the
frozen sibling data; it skips cleanly when that directory is absent.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specialist import cli  # noqa: E402
from specialist import dataset as dataset_mod  # noqa: E402
from specialist.gather import gather  # noqa: E402
from specialist.model import parse_prediction, strict_prediction  # noqa: E402
from specialist.prepare import prepare  # noqa: E402

LEGACY_RAW_DIR = Path("/home/tilakoid/github-triage-data")
CONFIG_PATH = ROOT / "configs" / "vscode-bug-feature.yaml"
LABELS = ["bug", "feature-request"]

# Tiny synthetic split bodies used to exercise the pinned fetch path offline.
PINNED_BODIES = {
    "train": b'{"label": "bug"}\n{"label": "feature-request"}\n',
    "val": b'{"label": "feature-request"}\n',
    "test": b'{"label": "bug"}\n',
}


def _specs_for(bodies: dict[str, bytes]) -> dict:
    return {
        name: {
            "remote": dataset_mod.local_name(name),
            "size": len(body),
            "rows": sum(1 for line in body.splitlines() if line.strip()),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in bodies.items()
    }


def _serve(specs: dict, bodies: dict[str, bytes], calls: list):
    def fake_urlopen(url, timeout=60.0):
        calls.append(url)
        for name, spec in specs.items():
            if url.endswith(spec["remote"]):
                return io.BytesIO(bodies[name])
        raise AssertionError(f"unexpected URL: {url}")

    return fake_urlopen


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


class PinnedPrepareTest(unittest.TestCase):
    def test_prepare_fetches_verifies_and_reuses_the_cache(self):
        config = cli.load_config(CONFIG_PATH)
        specs = _specs_for(PINNED_BODIES)
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with mock.patch.object(dataset_mod, "DATASET_SPLITS", specs), \
                    mock.patch.object(dataset_mod, "_urlopen",
                                      side_effect=_serve(specs, PINNED_BODIES, calls)):
                stats = prepare(config, workspace)
            self.assertEqual(len(calls), 3)
            for name, body in PINNED_BODIES.items():
                self.assertEqual((workspace / f"{name}.jsonl").read_bytes(), body)
            self.assertEqual(
                stats["split"]["sizes"],
                {"train": 2, "val": 1, "test": 1},
            )
            self.assertEqual(
                json.loads((workspace / "dataset_stats.json").read_text(encoding="utf-8"))["labels"],
                LABELS,
            )

            # Second run: every split verifies locally, no network call.
            calls.clear()
            with mock.patch.object(dataset_mod, "DATASET_SPLITS", specs), \
                    mock.patch.object(dataset_mod, "_urlopen",
                                      side_effect=AssertionError("cache hit must not fetch")):
                prepare(config, workspace)
            self.assertEqual(calls, [])

    def test_standalone_phases_fail_closed_without_verified_splits(self):
        config = cli.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for call in (
                lambda: cli.cmd_baseline(config, workspace),
                lambda: cli.cmd_train(config, workspace),
                lambda: cli.cmd_evaluate(config, workspace, "base"),
            ):
                with self.assertRaises(dataset_mod.DatasetError) as ctx:
                    call()
                self.assertIn("missing frozen split", str(ctx.exception))


class GatherTest(unittest.TestCase):
    def test_gather_refuses_pinned_config(self):
        config = cli.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("specialist.gather.gh_issue_list",
                            side_effect=AssertionError("must not gather")):
                with self.assertRaises(RuntimeError) as ctx:
                    gather(config, tmp)
        self.assertIn("pinned prepared dataset", str(ctx.exception))
        self.assertIn(dataset_mod.DATASET_REPO, str(ctx.exception))

    def test_legacy_gather_copies_raw_files_byte_for_byte(self):
        config = cli.load_config(CONFIG_PATH)
        config.pop("dataset", None)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for label in LABELS:
                source = tmp_path / f"{label}.json"
                source.write_bytes(b'[{"number": 1}]\n')
                config["sources"][label]["file"] = str(source)
            workspace = tmp_path / "workspace"
            report = gather(config, workspace)
            for label in LABELS:
                self.assertEqual(
                    (workspace / "raw" / f"{label}.json").read_bytes(),
                    b'[{"number": 1}]\n',
                )
                self.assertEqual(report[label]["mode"], "copied")


class RunPhasesTest(unittest.TestCase):
    def test_pinned_run_skips_gather(self):
        config = cli.load_config(CONFIG_PATH)
        self.assertEqual(cli.phases_for_run(config), ("prepare", "baseline", "train"))

    def test_cmd_run_skips_gather_for_pinned_config(self):
        config = cli.load_config(CONFIG_PATH)
        phases: list[str] = []
        comparison = {
            "baseline": {"strict_accuracy": 0.5, "semantic_accuracy": 0.6},
            "finetuned": {"strict_accuracy": 0.7, "semantic_accuracy": 0.8},
            "accuracy": {"delta_pp": {"strict": 20.0, "semantic": 20.0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            with mock.patch.object(cli, "run_phase",
                                   side_effect=lambda phase, *args, **kwargs: phases.append(phase)), \
                    mock.patch.object(cli, "build_comparison", return_value=comparison):
                cli.cmd_run(CONFIG_PATH, run_dir, config)
        self.assertEqual(phases, ["prepare", "baseline", "train", "evaluate"])

    def test_legacy_run_keeps_gather(self):
        config = cli.load_config(CONFIG_PATH)
        config.pop("dataset", None)
        self.assertEqual(
            cli.phases_for_run(config), ("gather", "prepare", "baseline", "train")
        )


class ActivePathTest(unittest.TestCase):
    def test_no_active_absolute_machine_paths(self):
        active = [
            CONFIG_PATH,
            *sorted((ROOT / "configs" / "qlora-large").glob("*.yaml")),
            *sorted((ROOT / "src" / "specialist").glob("*.py")),
            ROOT / "scripts" / "run_qlora_large.py",
        ]
        for path in active:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("/home/", text)
                self.assertNotIn("github-triage-data", text)


@unittest.skipUnless(LEGACY_RAW_DIR.is_dir(), "frozen reference data not available")
class LegacyReproductionTest(unittest.TestCase):
    def test_legacy_prepare_reproduces_frozen_splits_byte_for_byte(self):
        # Legacy config: the pinned dataset block is removed and raw file paths
        # restored, exactly the pre-pin configuration.
        config = cli.load_config(CONFIG_PATH)
        config.pop("dataset", None)
        config["sources"]["bug"]["file"] = str(LEGACY_RAW_DIR / "bugs.json")
        config["sources"]["feature-request"]["file"] = str(LEGACY_RAW_DIR / "features.json")

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
                    (LEGACY_RAW_DIR / f"{name}.jsonl").read_bytes(),
                    f"{name}.jsonl differs from frozen reference",
                )

            stats = json.loads((workspace / "dataset_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(
                stats["split"]["sizes"],
                {"train": 1593, "val": 200, "test": 200},
            )


if __name__ == "__main__":
    unittest.main()
