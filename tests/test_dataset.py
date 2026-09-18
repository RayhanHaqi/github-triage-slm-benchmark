"""Pinned dataset trust-root tests. No network: `_urlopen` is always mocked.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specialist import dataset as dataset_mod  # noqa: E402


def body_for(rows: int = 2) -> bytes:
    return b"".join(
        b'{"issue_number": %d, "label": "bug"}\n' % index for index in range(rows)
    )


def spec_for(body: bytes, *, rows: int = 2, remote: str = "train.jsonl") -> dict:
    return {
        "remote": remote,
        "size": len(body),
        "rows": rows,
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def serve(body: bytes):
    return mock.patch.object(
        dataset_mod, "_urlopen", side_effect=lambda url, timeout=60.0: io.BytesIO(body)
    )


class TrustRootTest(unittest.TestCase):
    def test_identity_is_pinned_and_complete(self):
        identity = dataset_mod.dataset_identity()
        self.assertEqual(identity["repo"], dataset_mod.DATASET_REPO)
        self.assertEqual(identity["revision"], dataset_mod.DATASET_REVISION)
        self.assertRegex(dataset_mod.DATASET_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(set(identity["splits"]), {"train", "val", "test"})
        for split, spec in dataset_mod.DATASET_SPLITS.items():
            self.assertEqual(len(spec["sha256"]), 64)
            self.assertGreater(spec["size"], 0)
            self.assertEqual(identity["splits"][split]["sha256"], spec["sha256"])
            self.assertEqual(identity["splits"][split]["local"], f"{split}.jsonl")

    def test_remote_validation_filename_mapping(self):
        self.assertTrue(dataset_mod.remote_url("train").startswith("https://"))
        self.assertTrue(dataset_mod.remote_url("train").endswith("/train.jsonl"))
        self.assertTrue(dataset_mod.remote_url("val").endswith("/validation.jsonl"))
        self.assertTrue(dataset_mod.remote_url("test").endswith("/test.jsonl"))
        self.assertIn(
            f"/{dataset_mod.DATASET_REPO}/resolve/{dataset_mod.DATASET_REVISION}/",
            dataset_mod.remote_url("val"),
        )
        with self.assertRaises(dataset_mod.DatasetError):
            dataset_mod.remote_url("other")

    def test_config_validation(self):
        self.assertIsNone(dataset_mod.dataset_config({}))
        self.assertEqual(dataset_mod.config_problems({}), [])
        valid = {
            "dataset": {
                "repo": dataset_mod.DATASET_REPO,
                "revision": dataset_mod.DATASET_REVISION,
            }
        }
        self.assertEqual(dataset_mod.config_problems(valid), [])
        self.assertEqual(dataset_mod.dataset_config(valid)["revision"], dataset_mod.DATASET_REVISION)

        bad_blocks = (
            {"repo": "other/dataset", "revision": dataset_mod.DATASET_REVISION},
            {"repo": dataset_mod.DATASET_REPO, "revision": "main"},
            {"repo": dataset_mod.DATASET_REPO, "revision": "HEAD"},
            {"repo": dataset_mod.DATASET_REPO, "revision": "0" * 39},
            {"repo": dataset_mod.DATASET_REPO, "revision": "0" * 40},
        )
        for block in bad_blocks:
            with self.subTest(block=block):
                with self.assertRaises(dataset_mod.DatasetError):
                    dataset_mod.dataset_config({"dataset": block})


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def _temp_files(self) -> list[Path]:
        return list(self.directory.glob(".*.part"))

    def test_valid_fetch_then_verified_cache_without_network(self):
        body = body_for()
        spec = spec_for(body)
        with serve(body) as opener:
            entry = dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertEqual(entry["rows"], 2)
        self.assertEqual((self.directory / "train.jsonl").read_bytes(), body)
        self.assertEqual(opener.call_count, 1)
        self.assertIn(dataset_mod.DATASET_REVISION, opener.call_args[0][0])

        with mock.patch.object(
            dataset_mod, "_urlopen", side_effect=AssertionError("cache hit must not use network")
        ):
            entry = dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertEqual(entry["sha256"], spec["sha256"])
        self.assertFalse(self._temp_files())

    def test_oversized_stream_is_refused(self):
        body = body_for(3)
        spec = spec_for(body, rows=3)
        spec["size"] = len(body) - 1  # ceiling below the served bytes
        with serve(body):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("exceeded the expected size", str(ctx.exception))
        self.assertFalse((self.directory / "train.jsonl").exists())
        self.assertFalse(self._temp_files())

    def test_short_stream_size_mismatch_is_refused(self):
        body = body_for()
        spec = spec_for(body)
        spec["size"] = len(body) + 10
        with serve(body):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("size mismatch", str(ctx.exception))
        self.assertFalse(self._temp_files())

    def test_hash_mismatch_is_refused(self):
        body = body_for()
        spec = spec_for(body)
        spec["sha256"] = "0" * 64
        with serve(body):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("sha256 mismatch", str(ctx.exception))
        self.assertFalse((self.directory / "train.jsonl").exists())
        self.assertFalse(self._temp_files())

    def test_row_count_mismatch_is_refused(self):
        body = body_for()
        spec = spec_for(body, rows=3)
        with serve(body):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("row count mismatch", str(ctx.exception))
        self.assertFalse(self._temp_files())

    def test_http_failure_is_refused_and_temp_cleaned(self):
        spec = spec_for(body_for())
        error = urllib.error.HTTPError(dataset_mod.remote_url("train"), 404, "Not Found", None, None)
        with mock.patch.object(dataset_mod, "_urlopen", side_effect=error):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("404", str(ctx.exception))
        self.assertFalse((self.directory / "train.jsonl").exists())
        self.assertFalse(self._temp_files())

    def test_non_https_final_redirect_url_refused_before_writing(self):
        spec = spec_for(body_for())

        class RedirectedResponse:
            status = 200

            def geturl(self):
                return "http://cdn.example.invalid/train.jsonl"

            def read(self, size):
                raise AssertionError("must not read bytes after a non-HTTPS final URL")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(dataset_mod, "_urlopen", return_value=RedirectedResponse()):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertIn("non-HTTPS redirect target", str(ctx.exception))
        self.assertFalse((self.directory / "train.jsonl").exists())
        self.assertFalse(self._temp_files())

    def test_predictable_temp_symlink_cannot_truncate_outside_victim(self):
        body = body_for()
        spec = spec_for(body)
        dest_dir = self.directory / "dest"
        outside_dir = self.directory / "outside"
        dest_dir.mkdir()
        outside_dir.mkdir()
        victim = outside_dir / "victim.bin"
        victim.write_bytes(b"do not truncate")
        predictable = dest_dir / f".train.jsonl.{os.getpid()}.part"
        try:
            predictable.symlink_to(victim)
        except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform
            self.skipTest(f"symlinks unsupported on this platform: {exc}")

        with serve(body):
            dataset_mod.fetch_split(dest_dir, "train", {"train": spec})

        self.assertEqual(victim.read_bytes(), b"do not truncate")
        self.assertTrue(predictable.is_symlink())
        self.assertEqual((dest_dir / "train.jsonl").read_bytes(), body)
        leftovers = [path for path in dest_dir.glob(".*.part") if path != predictable]
        self.assertEqual(leftovers, [])

    def test_non_https_url_refused(self):
        with self.assertRaises(dataset_mod.DatasetError):
            dataset_mod._urlopen("http://example.invalid/data.jsonl")

    def test_valid_destination_is_preserved_without_network(self):
        body = body_for()
        spec = spec_for(body)
        (self.directory / "train.jsonl").write_bytes(body)
        with mock.patch.object(
            dataset_mod, "_urlopen", side_effect=AssertionError("verified cache must not fetch")
        ):
            entry = dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertEqual(entry["sha256"], spec["sha256"])
        self.assertEqual((self.directory / "train.jsonl").read_bytes(), body)

    def test_failed_fetch_preserves_corrupt_destination(self):
        spec = spec_for(body_for())
        corrupt = b"not the dataset\n"
        (self.directory / "train.jsonl").write_bytes(corrupt)
        with serve(b"different bytes\n"):
            with self.assertRaises(dataset_mod.DatasetError):
                dataset_mod.fetch_split(self.directory, "train", {"train": spec})
        self.assertEqual((self.directory / "train.jsonl").read_bytes(), corrupt)
        self.assertFalse(self._temp_files())

    def test_ensure_dataset_verify_only_refuses_missing_and_changed(self):
        body = body_for()
        spec = {"train": spec_for(body)}
        with self.assertRaises(dataset_mod.DatasetError) as ctx:
            dataset_mod.ensure_dataset(self.directory, splits=spec, fetch=False)
        self.assertIn("missing frozen split", str(ctx.exception))
        self.assertFalse((self.directory / "train.jsonl").exists())

        with serve(body):
            report = dataset_mod.ensure_dataset(self.directory, splits=spec, fetch=True)
        self.assertEqual(report["train"]["rows"], 2)

        (self.directory / "train.jsonl").write_bytes(body + b"\n")
        with mock.patch.object(
            dataset_mod, "_urlopen", side_effect=AssertionError("verify-only must not fetch")
        ):
            with self.assertRaises(dataset_mod.DatasetError) as ctx:
                dataset_mod.ensure_dataset(self.directory, splits=spec, fetch=False)
        self.assertIn("size mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
