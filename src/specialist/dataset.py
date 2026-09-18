"""Pinned public dataset for the active VSCode bug/feature-request benchmark.

Single local trust root for the prepared Hugging Face dataset: repository,
immutable full revision, remote/local split filenames, exact byte sizes, row
counts and SHA-256 digests all live here. Callers either verify a local cache
or fetch through ``https://huggingface.co/datasets/<repo>/resolve/<revision>/``;
there is no ``main``/HEAD/metadata fallback and no generic provider abstraction.

Stdlib only. Downloads stream into a temp file in the destination directory,
enforce the expected byte size, hash and non-empty row count, and only then
atomically replace the destination. A verified cache hit performs no network
access. Any failure preserves an existing destination and removes the temp file.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DATASET_REPO = "Tilakoid/vscode-bug-feature-triage"
DATASET_REVISION = "15c7d77e083d0cd30ae84cc5de6add1dce6cf950"
DATASET_URL_ROOT = "https://huggingface.co/datasets"

# split -> remote filename, exact size in bytes, row count, sha256. Local names
# are `<split>.jsonl` (val maps from validation.jsonl), owned by local_name().
DATASET_SPLITS: dict[str, dict] = {
    "train": {
        "remote": "train.jsonl",
        "size": 5697146,
        "rows": 1593,
        "sha256": "3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b",
    },
    "val": {
        "remote": "validation.jsonl",
        "size": 716214,
        "rows": 200,
        "sha256": "7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348",
    },
    "test": {
        "remote": "test.jsonl",
        "size": 640586,
        "rows": 200,
        "sha256": "9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025",
    },
}

_CHUNK_SIZE = 1 << 20
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_URL_TIMEOUT = 60.0


class DatasetError(RuntimeError):
    """Pinned dataset configuration, fetch or verification failure."""


def local_name(split: str) -> str:
    """Local filename for a split (train.jsonl, val.jsonl, test.jsonl)."""
    return f"{split}.jsonl"


def dataset_identity() -> dict:
    """Stable manifest identity: repo, exact revision and per-split digests."""
    return {
        "repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "splits": {
            split: {
                "remote": spec.get("remote", local_name(split)),
                "local": local_name(split),
                "size": spec.get("size"),
                "rows": spec["rows"],
                "sha256": spec["sha256"],
            }
            for split, spec in DATASET_SPLITS.items()
        },
    }


def config_problems(config: dict) -> list[str]:
    """Problems with a `dataset:` block; empty for legacy configs without one."""
    block = config.get("dataset")
    if block is None:
        return []
    if not isinstance(block, dict):
        return ["must be a mapping with repo and revision"]
    problems = []
    if block.get("repo") != DATASET_REPO:
        problems.append(f"repo: expected {DATASET_REPO!r}, found {block.get('repo')!r}")
    revision = block.get("revision")
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        problems.append(
            f"revision: expected a full 40-hex revision, found {revision!r}"
        )
    elif revision != DATASET_REVISION:
        problems.append(
            f"revision: expected {DATASET_REVISION!r}, found {revision!r}"
        )
    return problems


def dataset_config(config: dict) -> dict | None:
    """Normalized pinned dataset block, or None for a legacy raw config.

    Raises DatasetError when a `dataset:` block exists but is not the exact
    pinned repo/revision.
    """
    if config.get("dataset") is None:
        return None
    problems = config_problems(config)
    if problems:
        raise DatasetError("invalid pinned dataset config:\n  - " + "\n  - ".join(problems))
    return {"repo": DATASET_REPO, "revision": DATASET_REVISION}


def remote_url(split: str, splits: dict | None = None) -> str:
    """The only allowed fetch URL for one pinned split (HTTPS, exact revision)."""
    specs = DATASET_SPLITS if splits is None else splits
    if split not in specs:
        raise DatasetError(f"unknown dataset split: {split!r}")
    if not _REPO.fullmatch(DATASET_REPO):
        raise DatasetError(f"invalid pinned dataset repo: {DATASET_REPO!r}")
    if not _SHA40.fullmatch(DATASET_REVISION):
        raise DatasetError(f"pinned dataset revision is not a full 40-hex sha: {DATASET_REVISION!r}")
    remote = specs[split].get("remote") or local_name(split)
    return f"{DATASET_URL_ROOT}/{DATASET_REPO}/resolve/{DATASET_REVISION}/{remote}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def _split_report(directory: str | Path, split: str, specs: dict) -> tuple[dict, list[str]]:
    """Inspect one local split against its expected digest; never touches network."""
    spec = specs[split]
    path = Path(directory) / local_name(split)
    entry: dict = {"path": str(path), "exists": path.is_file()}
    problems: list[str] = []
    if not entry["exists"]:
        return entry, [f"missing frozen split: {path}"]
    size = path.stat().st_size
    sha256 = _sha256_file(path)
    rows = _count_rows(path)
    entry.update({"size": size, "sha256": sha256, "rows": rows})
    expected_size = spec.get("size")
    if expected_size is not None and size != expected_size:
        problems.append(f"{split} size mismatch: {size} != {expected_size}")
    if sha256 != spec["sha256"]:
        problems.append(f"{split} sha256 mismatch: {sha256} != {spec['sha256']}")
    if rows != spec["rows"]:
        problems.append(f"{split} row count mismatch: {rows} != {spec['rows']}")
    return entry, problems


def verify_split(directory: str | Path, split: str, splits: dict | None = None) -> dict:
    """Verified entry for one local split; raises DatasetError on any mismatch."""
    specs = DATASET_SPLITS if splits is None else splits
    if split not in specs:
        raise DatasetError(f"unknown dataset split: {split!r}")
    entry, problems = _split_report(directory, split, specs)
    if problems:
        raise DatasetError(f"{split} split verification failed: " + "; ".join(problems))
    return entry


def verify_dataset(
    directory: str | Path, splits: dict | None = None
) -> tuple[dict, list[str]]:
    """(report, problems) for every expected split; problems never raise here."""
    specs = DATASET_SPLITS if splits is None else splits
    report: dict = {}
    problems: list[str] = []
    for split in specs:
        entry, split_problems = _split_report(directory, split, specs)
        report[split] = entry
        problems += split_problems
    return report, problems


def _urlopen(url: str, timeout: float = _URL_TIMEOUT):
    if not url.startswith(DATASET_URL_ROOT + "/"):
        raise DatasetError(f"refusing a non-Hugging-Face URL: {url}")
    return urllib.request.urlopen(url, timeout=timeout)


def fetch_split(directory: str | Path, split: str, splits: dict | None = None) -> dict:
    """Return a verified local split, fetching it only when the cache is invalid.

    Streams into an exclusively created temp file in `directory` (same
    filesystem, closed before `os.replace`), enforces the expected byte size,
    sha256 and non-empty row count, then atomically replaces the destination.
    """
    specs = DATASET_SPLITS if splits is None else splits
    if split not in specs:
        raise DatasetError(f"unknown dataset split: {split!r}")
    spec = specs[split]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    entry, problems = _split_report(directory, split, specs)
    if not problems:
        return entry

    url = remote_url(split, specs)
    dest = directory / local_name(split)
    # Exclusive stdlib temp creation: a predictable name (or a symlink planted
    # at one) must never be opened, so an outside file cannot be truncated.
    temp_file = tempfile.NamedTemporaryFile(
        dir=directory, prefix=f".{dest.name}.", suffix=".part", delete=False
    )
    temp = Path(temp_file.name)
    expected_size = spec.get("size")
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            response = _urlopen(url)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"failed to fetch {url}: {type(exc).__name__}: {exc}") from exc
        with response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise DatasetError(f"{url} returned HTTP {status}")
            # Redirects are followed by urllib; the final URL must stay HTTPS
            # before a single byte is written. In-memory test doubles without
            # geturl keep working.
            geturl = getattr(response, "geturl", None)
            if geturl is not None:
                final_url = geturl()
                if not isinstance(final_url, str) or not final_url.startswith("https://"):
                    raise DatasetError(
                        f"refusing a non-HTTPS redirect target for {url}: {final_url!r}"
                    )
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if expected_size is not None and size > expected_size:
                    raise DatasetError(
                        f"{url} exceeded the expected size {expected_size} bytes"
                    )
                digest.update(chunk)
                temp_file.write(chunk)
        temp_file.close()
        if expected_size is not None and size != expected_size:
            raise DatasetError(f"{url} size mismatch: {size} != {expected_size}")
        actual_sha = digest.hexdigest()
        if actual_sha != spec["sha256"]:
            raise DatasetError(
                f"{url} sha256 mismatch: {actual_sha} != {spec['sha256']}"
            )
        rows = _count_rows(temp)
        if rows != spec["rows"]:
            raise DatasetError(f"{url} row count mismatch: {rows} != {spec['rows']}")
        os.replace(temp, dest)
    except BaseException:
        try:
            temp_file.close()
        except OSError:
            pass
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return verify_split(directory, split, specs)


def ensure_dataset(
    directory: str | Path, *, splits: dict | None = None, fetch: bool = False
) -> dict:
    """Verified report for every split; fetch missing/invalid splits when asked.

    With `fetch=False` (verify only, e.g. a resume or a standalone phase) a
    missing or changed split raises DatasetError and is never repaired.
    """
    specs = DATASET_SPLITS if splits is None else splits
    if fetch:
        return {split: fetch_split(directory, split, specs) for split in specs}
    report, problems = verify_dataset(directory, specs)
    if problems:
        raise DatasetError(
            "frozen dataset verification failed:\n  - " + "\n  - ".join(problems)
        )
    return report


def require_configured_dataset(
    config: dict, directory: str | Path, *, fetch: bool = False
) -> dict | None:
    """Ensure the pinned dataset for a config; None for legacy raw configs."""
    if dataset_config(config) is None:
        return None
    return ensure_dataset(directory, fetch=fetch)
