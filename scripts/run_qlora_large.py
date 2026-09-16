#!/usr/bin/env python3
"""Sequential runner for the fixed four-model QLoRA-large benchmark track.

Deliberately narrow: one approved model list, one fixed order, one output
namespace (`benchmarks/qlora-large/<run-id>/`); never touches the completed
BF16 track (`benchmarks/20260915T112519Z`) or the root `benchmarks/LATEST`.

Per model, in order:
  1. preflight  -- tokenizer/processor-only selection of 8 train / 2 val / 1
     test rows whose rendered supervised prompt is >= 2048 tokens for the exact
     pinned revision (resolved SHA required), then a tiny run in a *fresh*
     `preflight/attempt-N/{first,repeat}` workspace through the production
     train path (batch/GA unchanged, eval batch 1, 3 epochs of one optimizer
     step each, eval/save/best-checkpoint reload), one quantized baseline
     generation and one selected-adapter generation, all validated. Peak VRAM
     decides go / repeat once / stop; on go the attempt scratch is deleted.
  2. baseline   -- production `specialist.cli baseline` on the frozen 200-row test split.
  3. train      -- production `specialist.cli train` (3 epochs, best checkpoint).
  4. adapter_eval -- production `specialist.cli evaluate` on the same 200 rows.
  5. comparison -- existing `specialist.cli.build_comparison`.
  6. acceptance -- strict artifact/quantization/revision/hash/VRAM validation.
  7. cleanup    -- GPU-idle checked, known-safe descendants only, recorded.

Recovery: `--resume <run-dir>` continues a partial run, re-validates git
commit/config digests/environment/frozen data/GPU identity before every model,
and never reruns a successful phase (baseline included). `--retry-failed`
(resume only) allows at most one retry of a phase whose failure matches the
positive transient allowlist (HF/network 5xx/timeout/connection, AF_UNIX path,
explicit external interruption); everything else is a hard stop. A train retry
resumes only from a complete checkpoint (trainer_state + adapter + optimizer +
scheduler + rng state), otherwise the partial train output is deleted and the
run restarts the phase identically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from specialist.cli import build_comparison, load_config, quantization_view  # noqa: E402
from specialist.model import sha256_file  # noqa: E402

TRACK = "qlora-large"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
RUN_ROOT = BENCHMARKS_DIR / TRACK

# Never written to or deleted anywhere in this runner.
PROTECTED_PATHS = (
    BENCHMARKS_DIR / "20260915T112519Z",  # completed BF16 benchmark
    BENCHMARKS_DIR / "20260915T100825Z",  # aborted earlier attempt
    BENCHMARKS_DIR / "LATEST",  # root pointer belongs to the BF16 track
)

# Untracked paths tolerated during a run are computed per run (exact active run
# directory); new runs and dry-run allow none.

FROZEN_DATA_DIR = Path("/home/tilakoid/github-triage-data")
FROZEN_SPLITS = {
    "train": {
        "sha256": "3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b",
        "rows": 1593,
    },
    "val": {
        "sha256": "7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348",
        "rows": 200,
    },
    "test": {
        "sha256": "9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025",
        "rows": 200,
    },
}

MIN_FREE_GIB = 80.0
SHORT_TMP_ROOT = Path("/tmp/opencode/b")

PINNED_VERSIONS = {
    "torch": "2.11.0",
    "transformers": "5.5.0",
    "peft": "0.20.0",
    "datasets": "4.3.0",
    "trl": "0.24.0",
    "unsloth": "2026.9.4",
    "unsloth_zoo": "2026.9.3",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
    "tqdm": "4.70.1",
}

PREFLIGHT_NEEDS = {"train": 8, "val": 2, "test": 1}
MIN_SUPERVISED_TOKENS = 2048

VRAM_GO_GIB = 13.50
VRAM_REPEAT_MAX_GIB = 14.25
VRAM_HEADROOM_MIN_GIB = 1.0
VRAM_REPEAT_TOLERANCE_GIB = 0.25

PHASES = ("preflight", "baseline", "train", "adapter_eval", "comparison", "acceptance", "cleanup")

SYSTEM_PROMPT = (
    "Classify the GitHub issue into exactly one category: bug or feature-request. "
    "Return only the category name."
)
SOURCE_FILES = {"bug": "bugs.json", "feature-request": "features.json"}
SOURCE_REPO = "microsoft/vscode"

# Track-critical expectations shared by all four configs; per-model fields live
# on ModelSpec. Anything here is checked against source and snapshot configs
# before any GPU/network work.
TRACK_EXPECTATIONS = {
    "model": {
        "system_prompt": SYSTEM_PROMPT,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "use_exact_model_name": True,
        "trust_remote_code": False,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0,
        "bias": "none",
        "gradient_checkpointing": "unsloth",
        "random_state": 42,
    },
    "training": {
        "output_subdir": "trainer",
        "adapter_subdir": "adapter",
        "per_device_eval_batch_size": 1,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": True,
        "fp16": False,
        "optim": "adamw_8bit",
        "learning_rate": 2.0e-4,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "report_to": "none",
    },
    "evaluation": {
        "split": "test",
        "max_user_tokens": 1800,
        "max_new_tokens": 12,
        "do_sample": False,
        "pad_eos": True,
    },
    "benchmark": {"method": "QLoRA NF4"},
}
EFFECTIVE_BATCH = 8

# Positive retry allowlist: transient HF/network conditions, the known AF_UNIX
# short-path failure and explicit external interruption. Anything else,
# including every unknown/in-process error, is a hard stop.
TRANSIENT_FAILURE_PATTERNS = (
    re.compile(r"service unavailable|bad gateway|gateway time-?out|internal server error|too many requests"),
    re.compile(r"http(?:s)?(?: error| status)?\s*[:=]?\s*(?:429|500|502|503|504)\b"),
    re.compile(r"timed? out|timeout|read timed out|connect(?:ion)? time-?out"),
    re.compile(r"connection (?:reset|aborted|refused|closed|error)"),
    re.compile(r"temporary failure in name resolution|name or service not known|nodename nor servname"),
    re.compile(r"remote end closed connection|incomplete read|protocol error"),
    re.compile(r"af_unix path too long|unix socket path.*too long"),
    re.compile(r"externally interrupted|interrupted by signal|received sigint|received sigterm|keyboardinterrupt"),
)
TRANSIENT_EXIT_CODES = (130, 143)

# Conservative tokens: any trainable parameter whose name contains one of these
# means a vision-like tower/projector is being trained (not just echoed flags).
VISION_LIKE_PARAMETER_TOKENS = (
    "vision",
    "visual",
    "image",
    "pixel",
    "patch",
    "multi_modal",
    "multimodal",
    "mm_projector",
    "projector",
)

# Phase-owned outputs that must be cleared before a fresh retry of that phase.
PHASE_PARTIAL_FILES = {
    "baseline": ("baseline_metrics.json", "baseline_predictions.csv"),
    "adapter_eval": ("finetuned_metrics.json", "finetuned_predictions.csv"),
    "comparison": ("comparison.json",),
    "acceptance": ("acceptance.json",),
    "cleanup": ("cleanup.json",),
}

CHECKPOINT_REQUIRED_FILES = (
    "trainer_state.json",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
CHECKPOINT_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass(frozen=True)
class ModelSpec:
    index: int
    slug: str
    config: str
    repo: str
    revision: str
    kind: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    display_name: str
    lora_targets: object  # tuple of module names, or "all-linear"
    vision_collator: bool
    chat_template_kwargs: dict


# Fixed order and exact identity, cross-checked against the configs before any
# download or GPU work.
EXPECTED_MODELS = (
    ModelSpec(
        1, "01-qwen3-8b", "configs/qlora-large/01-qwen3-8b.yaml",
        "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
        "language", 2, 4, "Qwen3-8B",
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        False, {"enable_thinking": False},
    ),
    ModelSpec(
        2, "02-ministral-3-8b-instruct", "configs/qlora-large/02-ministral-3-8b-instruct.yaml",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "f6fae9795746f63c9be8344932f01275f3c63734",
        "vision", 2, 4, "Ministral-3-8B-Instruct-2512-BF16",
        "all-linear", True, {},
    ),
    ModelSpec(
        3, "03-qwen3.5-9b", "configs/qlora-large/03-qwen3.5-9b.yaml",
        "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "vision", 2, 4, "Qwen3.5-9B",
        "all-linear", False, {"enable_thinking": False},
    ),
    ModelSpec(
        4, "04-ministral-3-14b-instruct", "configs/qlora-large/04-ministral-3-14b-instruct.yaml",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "3cea74c1ebaf5ce5f5a2553de470e2ceab825142",
        "vision", 1, 8, "Ministral-3-14B-Instruct-2512-BF16",
        "all-linear", True, {},
    ),
)

# (component label, metrics slot, metric key) for the VRAM gate. Every value is
# required to be present, numeric, finite and > 0.
PEAK_FIELDS = (
    ("train_setup_reserved_gib", "train", "setup_peak_reserved_gib"),
    ("train_setup_allocated_gib", "train", "setup_peak_allocated_gib"),
    ("train_training_reserved_gib", "train", "train_peak_reserved_gib"),
    ("train_training_allocated_gib", "train", "train_peak_allocated_gib"),
    ("baseline_load_reserved_gib", "baseline", "load_peak_reserved_gb"),
    ("baseline_load_allocated_gib", "baseline", "load_peak_allocated_gb"),
    ("baseline_generation_reserved_gib", "baseline", "peak_reserved_gb"),
    ("baseline_generation_allocated_gib", "baseline", "peak_allocated_gb"),
    ("adapter_load_reserved_gib", "adapter", "load_peak_reserved_gb"),
    ("adapter_load_allocated_gib", "adapter", "load_peak_allocated_gb"),
    ("adapter_generation_reserved_gib", "adapter", "peak_reserved_gb"),
    ("adapter_generation_allocated_gib", "adapter", "peak_allocated_gb"),
)


class RunnerError(RuntimeError):
    """Fatal runner problem (validation, identity, recovery policy)."""


class PhaseFailure(RunnerError):
    def __init__(self, phase: str, message: str, *, retryable: bool = False, log: str | None = None):
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.retryable = bool(retryable)
        self.log = log


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_lines(path: str | Path) -> int:
    total = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def count_csv_data_rows(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def read_tail(path: str | Path, limit: int = 4000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_within(path: str | Path, root: str | Path) -> bool:
    p, r = _resolved(path), _resolved(root)
    return p == r or r in p.parents


def protected_intersection(path: str | Path, protected=None) -> bool:
    """True when `path` is, contains, or is contained by a protected path."""
    protected = PROTECTED_PATHS if protected is None else protected
    p = _resolved(path)
    for candidate in protected:
        c = _resolved(candidate)
        if p == c or c in p.parents or p in c.parents:
            return True
    return False


def existing_ancestor(path: str | Path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def new_run_id(root: str | Path) -> str:
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate, counter = stamp, 1
    while (root / candidate).exists():
        candidate = f"{stamp}-{counter}"
        counter += 1
    return candidate


def spec_by_slug(slug: str) -> ModelSpec:
    for spec in EXPECTED_MODELS:
        if spec.slug == slug:
            return spec
    raise RunnerError(f"unknown model slug: {slug!r}; fixed track is {[s.slug for s in EXPECTED_MODELS]}")


# --------------------------------------------------------------------------- #
# probes (all external effects; patched in tests)
# --------------------------------------------------------------------------- #

def probe_versions() -> dict:
    from importlib import metadata

    versions = {"python": sys.version.split()[0]}
    for name in PINNED_VERSIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_pinned_versions(versions: dict) -> list[str]:
    problems = []
    for name, expected in PINNED_VERSIONS.items():
        actual = versions.get(name)
        # Local build tags (e.g. torch 2.11.0+cu128) keep the pinned core version.
        core = actual.split("+", 1)[0] if isinstance(actual, str) else actual
        if core != expected:
            problems.append(f"version mismatch: {name}=={actual!r}, expected {expected!r}")
    return problems


def probe_git_head(project_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def probe_git_status(project_root: str | Path) -> list[str]:
    """Full untracked listing (`-uall`): no collapsed directories hiding files."""
    result = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def git_tree_problems(
    porcelain_lines: list[str], allowed_untracked: tuple[str, ...] = ()
) -> list[str]:
    """Tracked changes are always dirty; untracked paths must be explicitly allowed.

    Callers pass the exact active run prefix (`benchmarks/qlora-large/<run-id>/`)
    during a run/resume; new runs and dry-run allow no untracked paths at all.
    """
    problems = []
    for line in porcelain_lines:
        line = line.rstrip()
        if not line:
            continue
        status = line[:2]
        raw_path = line[3:].strip().strip('"') if len(line) > 3 else ""
        if status == "??":
            if allowed_untracked and raw_path.startswith(tuple(allowed_untracked)):
                continue
            problems.append(f"untracked path outside the active run: {raw_path}")
        else:
            problems.append(f"tracked change: {line}")
    return problems


def probe_disk_free_gib(path: str | Path) -> float:
    return shutil.disk_usage(existing_ancestor(path)).free / 1024**3


def probe_gpu_identity() -> dict:
    """Same-interpreter torch view of the compute device; errors propagate."""
    import torch

    if not torch.cuda.is_available():
        raise RunnerError("torch reports no usable CUDA device")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    if total_gib <= 0:
        raise RunnerError(f"torch reports non-positive device memory: {props.total_memory!r}")
    return {
        "device_index": 0,
        "name": str(props.name),
        "total_gib": float(total_gib),
        "visible_devices": int(torch.cuda.device_count()),
    }


def parse_gpu_compute_processes(output: str) -> list[str]:
    """CSV rows (`pid, name, memory`); empty output means idle, malformed fails closed."""
    processes = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "no running processes" in line.lower():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or not fields[0].isdigit():
            raise RunnerError(f"malformed nvidia-smi compute-process output: {line!r}")
        processes.append(line)
    return processes


def probe_gpu_compute_processes() -> list[str]:
    """nvidia-smi stays the foreign-process probe (identity/capacity come from torch).

    Fails closed: OSError, nonzero exit status or malformed output raise; an
    empty successful listing means the GPU is idle.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise RunnerError(f"nvidia-smi process probe failed: {type(exc).__name__}: {exc}") from exc
    if result.returncode != 0:
        raise RunnerError(
            f"nvidia-smi process probe failed with exit code {result.returncode}: "
            f"{result.stderr.strip()[:200]!r}"
        )
    return parse_gpu_compute_processes(result.stdout)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def validate_config(config: dict, spec: ModelSpec, data_dir: str | Path = FROZEN_DATA_DIR) -> list[str]:
    """Every track-critical field of source/snapshot configs, per fixed spec."""
    problems: list[str] = []
    model_cfg = config.get("model") or {}
    lora_cfg = config.get("lora") or {}
    train_cfg = config.get("training") or {}
    eval_cfg = config.get("evaluation") or {}
    bench_cfg = config.get("benchmark") or {}
    sources = config.get("sources") or {}

    def expect(container: dict, key: str, expected, label: str, *, default=None):
        actual = container.get(key, default)
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, found {actual!r}")

    expect(model_cfg, "base_model", spec.repo, f"{spec.slug} model.base_model")
    expect(model_cfg, "revision", spec.revision, f"{spec.slug} model.revision")
    expect(model_cfg, "kind", spec.kind, f"{spec.slug} model.kind", default="language")
    for key, expected in TRACK_EXPECTATIONS["model"].items():
        expect(model_cfg, key, expected, f"{spec.slug} model.{key}")
    for key, expected in TRACK_EXPECTATIONS["lora"].items():
        if key == "target_modules":
            continue
        expect(lora_cfg, key, expected, f"{spec.slug} lora.{key}")
    if isinstance(spec.lora_targets, str):
        expect(lora_cfg, "target_modules", spec.lora_targets, f"{spec.slug} lora.target_modules")
    else:
        expected_targets = list(spec.lora_targets)
        actual_targets = lora_cfg.get("target_modules")
        if actual_targets != expected_targets:
            problems.append(
                f"{spec.slug} lora.target_modules: expected {expected_targets!r}, "
                f"found {actual_targets!r}"
            )
    for key, expected in TRACK_EXPECTATIONS["training"].items():
        expect(train_cfg, key, expected, f"{spec.slug} training.{key}")
    expect(train_cfg, "per_device_train_batch_size", spec.per_device_train_batch_size,
           f"{spec.slug} training.per_device_train_batch_size")
    expect(train_cfg, "gradient_accumulation_steps", spec.gradient_accumulation_steps,
           f"{spec.slug} training.gradient_accumulation_steps")
    if spec.per_device_train_batch_size * spec.gradient_accumulation_steps != EFFECTIVE_BATCH:
        problems.append(f"{spec.slug} effective batch is not {EFFECTIVE_BATCH}")
    expect(train_cfg, "vision_collator", spec.vision_collator, f"{spec.slug} training.vision_collator")
    if spec.kind == "vision":
        for key in ("finetune_vision_layers", "finetune_language_layers",
                    "finetune_attention_modules", "finetune_mlp_modules"):
            expected = False if key == "finetune_vision_layers" else True
            expect(lora_cfg, key, expected, f"{spec.slug} lora.{key}")
    for key, expected in TRACK_EXPECTATIONS["evaluation"].items():
        expect(eval_cfg, key, expected, f"{spec.slug} evaluation.{key}")
    expected_template = dict(spec.chat_template_kwargs)
    actual_template = eval_cfg.get("chat_template_kwargs")
    if actual_template != expected_template:
        problems.append(
            f"{spec.slug} evaluation.chat_template_kwargs: expected {expected_template!r}, "
            f"found {actual_template!r}"
        )
    expect(bench_cfg, "checkpoint", spec.repo, f"{spec.slug} benchmark.checkpoint")
    expect(bench_cfg, "official_checkpoint", spec.repo, f"{spec.slug} benchmark.official_checkpoint")
    expect(bench_cfg, "revision", spec.revision, f"{spec.slug} benchmark.revision")
    expect(bench_cfg, "pinned_revision", spec.revision, f"{spec.slug} benchmark.pinned_revision")
    expect(bench_cfg, "display_name", spec.display_name, f"{spec.slug} benchmark.display_name")
    expect(bench_cfg, "method", TRACK_EXPECTATIONS["benchmark"]["method"],
           f"{spec.slug} benchmark.method")

    if list(sources) != list(SOURCE_FILES):
        problems.append(f"{spec.slug} sources: expected {list(SOURCE_FILES)}, found {list(sources)}")
    for label, filename in SOURCE_FILES.items():
        entry = sources.get(label) or {}
        expect(entry, "repo", SOURCE_REPO, f"{spec.slug} sources.{label}.repo")
        expect(entry, "github_label", label, f"{spec.slug} sources.{label}.github_label")
        file_value = entry.get("file")
        if not file_value:
            problems.append(f"{spec.slug} sources.{label}.file: missing (frozen dataset required)")
            continue
        path = Path(str(file_value)).expanduser()
        if path.name != filename or path.resolve().parent != Path(data_dir).resolve():
            problems.append(
                f"{spec.slug} sources.{label}.file: expected {filename} under "
                f"{Path(data_dir)}, found {path}"
            )
    return problems


def frozen_data_report(data_dir: str | Path, splits: dict | None = None) -> tuple[dict, list[str]]:
    splits = FROZEN_SPLITS if splits is None else splits
    report: dict = {}
    problems: list[str] = []
    for name, expected in splits.items():
        path = Path(data_dir) / f"{name}.jsonl"
        entry = {"path": str(path), "exists": path.is_file()}
        if not entry["exists"]:
            problems.append(f"missing frozen split: {path}")
            report[name] = entry
            continue
        entry["sha256"] = sha256_file(path)
        entry["rows"] = count_lines(path)
        if entry["sha256"] != expected["sha256"]:
            problems.append(f"{name} sha256 mismatch: {entry['sha256']} != {expected['sha256']}")
        if entry["rows"] != expected["rows"]:
            problems.append(f"{name} row count mismatch: {entry['rows']} != {expected['rows']}")
        report[name] = entry
    return report, problems


def copy_frozen_into_workspace(
    data_dir: str | Path, workspace: str | Path, splits: dict | None = None
) -> dict:
    """Byte-copy the frozen splits into a workspace and verify each copy."""
    splits = FROZEN_SPLITS if splits is None else splits
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    copied: dict = {}
    for name, expected in splits.items():
        src = Path(data_dir) / f"{name}.jsonl"
        dst = workspace / f"{name}.jsonl"
        if not src.is_file():
            raise RunnerError(f"frozen split missing: {src}")
        if not dst.is_file() or sha256_file(dst) != expected["sha256"]:
            shutil.copy2(src, dst)
        actual = sha256_file(dst)
        if actual != expected["sha256"]:
            raise RunnerError(
                f"copied {name} sha256 mismatch: {actual} != {expected['sha256']}"
            )
        copied[name] = {"sha256": actual, "rows": count_lines(dst)}
    return copied


def validate_namespace(root: str | Path, project_root: str | Path) -> list[str]:
    root = _resolved(root)
    problems = []
    if protected_intersection(root):
        problems.append(f"output root intersects a protected path: {root}")
    if root == _resolved(project_root):
        problems.append("output root must not be the project root")
    if root.exists() and not root.is_dir():
        problems.append(f"output root is not a directory: {root}")
    return problems


def write_track_latest(root: str | Path, run_id: str) -> Path:
    """Atomically write `<root>/LATEST` after a completed track; never the root pointer."""
    target = Path(root) / "LATEST"
    if protected_intersection(target):
        raise RunnerError(f"refusing to write protected LATEST: {target}")
    tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(run_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return target


# --------------------------------------------------------------------------- #
# VRAM gate, failure policy and peaks
# --------------------------------------------------------------------------- #

def vram_decision(
    peak_gib: float,
    total_gib: float | None,
    *,
    offload: bool = False,
    oom: bool = False,
    repeat_gib: float | None = None,
) -> tuple[str, str]:
    """Return ("go"|"repeat"|"stop", reason) for the peak reserved VRAM."""
    if oom:
        return "stop", "CUDA OOM during preflight is a hard stop"
    if offload:
        return "stop", "CPU/disk/meta offload or non-CUDA placement is a hard stop"
    if total_gib is None:
        return "stop", "PyTorch-visible GPU total is unknown; refusing to judge VRAM"
    if (total_gib - peak_gib) < VRAM_HEADROOM_MIN_GIB:
        return "stop", f"headroom {total_gib - peak_gib:.2f} GiB < {VRAM_HEADROOM_MIN_GIB} GiB"
    if peak_gib <= VRAM_GO_GIB:
        return "go", f"peak {peak_gib:.2f} GiB <= {VRAM_GO_GIB} GiB"
    if peak_gib <= VRAM_REPEAT_MAX_GIB:
        if repeat_gib is None:
            return "repeat", (
                f"peak {peak_gib:.2f} GiB in ({VRAM_GO_GIB}, {VRAM_REPEAT_MAX_GIB}] GiB; "
                "one fresh repeat required"
            )
        if repeat_gib > VRAM_REPEAT_MAX_GIB:
            return "stop", f"repeat peak {repeat_gib:.2f} GiB > {VRAM_REPEAT_MAX_GIB} GiB"
        if abs(repeat_gib - peak_gib) > VRAM_REPEAT_TOLERANCE_GIB:
            return "stop", (
                f"repeat peak {repeat_gib:.2f} GiB differs from first {peak_gib:.2f} GiB "
                f"by more than {VRAM_REPEAT_TOLERANCE_GIB} GiB"
            )
        if (total_gib - repeat_gib) < VRAM_HEADROOM_MIN_GIB:
            return "stop", f"repeat headroom {total_gib - repeat_gib:.2f} GiB < {VRAM_HEADROOM_MIN_GIB} GiB"
        return "go", f"repeat peak {repeat_gib:.2f} GiB within {VRAM_REPEAT_TOLERANCE_GIB} GiB of first"
    return "stop", f"peak {peak_gib:.2f} GiB > {VRAM_REPEAT_MAX_GIB} GiB"


def classify_failure(text: str | None, exit_code: int | None = None) -> tuple[bool, str]:
    """Positive transient allowlist; unknown failures are hard."""
    lowered = (text or "").lower()
    for pattern in TRANSIENT_FAILURE_PATTERNS:
        if pattern.search(lowered):
            return True, f"transient (matched {pattern.pattern!r})"
    if exit_code in TRANSIENT_EXIT_CODES:
        return True, f"external interruption (exit code {exit_code})"
    return False, "not in the transient allowlist"


def quantization_problems(metrics: dict, label: str = "") -> list[str]:
    prefix = f"{label}: " if label else ""
    quant = metrics.get("quantization") or {}
    problems: list[str] = []
    if not quant:
        return [f"{prefix}quantization metadata missing"]
    if not quant.get("effective_load_in_4bit"):
        problems.append(f"{prefix}effective 4-bit load not detected")
    if quant.get("quant_type") != ["nf4"]:
        problems.append(f"{prefix}quant_type != ['nf4']: {quant.get('quant_type')!r}")
    if quant.get("compute_dtype") != ["torch.bfloat16"]:
        problems.append(f"{prefix}compute_dtype != ['torch.bfloat16']: {quant.get('compute_dtype')!r}")
    if quant.get("double_quant") is not True:
        problems.append(f"{prefix}double_quant is not True: {quant.get('double_quant')!r}")
    offload = quant.get("offload") or {}
    offloaded = {key: value for key, value in offload.items() if value}
    if offloaded:
        problems.append(f"{prefix}offload present: {sorted(offloaded)}")
    if quant.get("non_cuda_parameter_count"):
        problems.append(f"{prefix}{quant['non_cuda_parameter_count']} parameters not on CUDA")
    if quant.get("bitsandbytes_8bit_module_count"):
        problems.append(f"{prefix}{quant['bitsandbytes_8bit_module_count']} bitsandbytes 8-bit modules")
    if not quant.get("quantized_parameter_devices"):
        problems.append(f"{prefix}no quantized parameter devices recorded")
    return problems


def any_offload(metrics_list: list[dict]) -> bool:
    for metrics in metrics_list:
        quant = metrics.get("quantization") or {}
        offload = quant.get("offload") or {}
        if any(offload.get(key) for key in ("cpu", "disk", "meta")):
            return True
        if quant.get("non_cuda_parameter_count"):
            return True
    return False


def peak_maxima(
    train_metrics: dict | None, baseline_metrics: dict | None, adapter_metrics: dict | None
) -> dict:
    """Max of setup/load and phase peaks; every required value must be valid."""
    slots = {"train": train_metrics or {}, "baseline": baseline_metrics or {}, "adapter": adapter_metrics or {}}
    components: dict = {}
    problems: list[str] = []
    for label, slot, key in PEAK_FIELDS:
        raw = slots[slot].get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            problems.append(f"missing/non-numeric peak {slot}.{key}: {raw!r}")
            components[label] = None
            continue
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            problems.append(f"invalid peak {slot}.{key}: {raw!r}")
            components[label] = None
            continue
        components[label] = value
    reserved = [value for key, value in components.items() if key.endswith("_reserved_gib")]
    allocated = [value for key, value in components.items() if key.endswith("_allocated_gib")]
    valid = all(value is not None for value in components.values())
    return {
        "components": components,
        "reserved_gib": max(components[key] for key in components if key.endswith("_reserved_gib")) if valid else None,
        "allocated_gib": max(components[key] for key in components if key.endswith("_allocated_gib")) if valid else None,
        "problems": problems,
    }


def identity_problems(metrics: dict, spec: ModelSpec, label: str) -> list[str]:
    problems = []
    if metrics.get("base_model") != spec.repo:
        problems.append(f"{label}: base_model {metrics.get('base_model')!r} != {spec.repo!r}")
    if metrics.get("model_kind") != spec.kind:
        problems.append(f"{label}: model_kind {metrics.get('model_kind')!r} != {spec.kind!r}")
    for field in ("requested_revision", "resolved_revision"):
        if metrics.get(field) != spec.revision:
            problems.append(f"{label}: {field} {metrics.get(field)!r} != {spec.revision!r}")
    return problems


def best_checkpoint_problems(train_metrics: dict, *, require_final: bool = True) -> list[str]:
    """Three epoch evals, finite losses, and best checkpoint/epoch/step consistency."""
    problems: list[str] = []
    entries = [
        entry for entry in train_metrics.get("log_history") or [] if "eval_loss" in entry
    ]
    epochs = [entry.get("epoch") for entry in entries]
    expected_epochs = [1.0, 2.0, 3.0]
    if [float(epoch) if isinstance(epoch, (int, float)) else epoch for epoch in epochs] != expected_epochs:
        problems.append(f"expected eval epochs {expected_epochs}, found {epochs}")
    for index, entry in enumerate(entries):
        loss = entry.get("eval_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            problems.append(f"eval loss {index + 1} not finite: {loss!r}")
    losses = [entry.get("eval_loss") for entry in entries]
    finite_losses = [loss for loss in losses if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(loss)]

    best_metric = train_metrics.get("best_metric")
    if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)) or not math.isfinite(best_metric):
        problems.append(f"best_metric not finite: {best_metric!r}")
    else:
        if finite_losses and not math.isclose(float(best_metric), min(finite_losses), rel_tol=1e-9, abs_tol=1e-9):
            problems.append(
                f"best_metric {best_metric!r} != min eval loss {min(finite_losses)!r}"
            )

    checkpoint = train_metrics.get("best_model_checkpoint")
    if not checkpoint:
        problems.append("best_model_checkpoint missing")
    elif not Path(str(checkpoint)).is_dir():
        problems.append(f"best_model_checkpoint not found on disk: {checkpoint}")

    best_epoch = train_metrics.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, (int, float)) or not math.isfinite(best_epoch):
        problems.append(f"best_epoch not recorded/finite: {best_epoch!r}")
    else:
        best_epoch = float(best_epoch)
        matching = [
            entry for entry in entries
            if isinstance(entry.get("epoch"), (int, float))
            and float(entry["epoch"]) == best_epoch
            and isinstance(entry.get("eval_loss"), (int, float))
            and isinstance(best_metric, (int, float))
            and math.isclose(float(entry["eval_loss"]), float(best_metric), rel_tol=1e-9, abs_tol=1e-9)
        ]
        if not matching:
            problems.append(
                f"best_epoch {best_epoch!r} does not match an eval entry with best_metric {best_metric!r}"
            )
    if checkpoint and Path(str(checkpoint)).is_dir():
        match = re.search(r"checkpoint-(\d+)\s*$", str(checkpoint))
        if not match:
            problems.append(f"cannot derive step from best_model_checkpoint: {checkpoint}")
        else:
            step = int(match.group(1))
            history_match = [
                entry for entry in entries
                if entry.get("step") is not None and int(entry["step"]) == step
                and isinstance(best_epoch, (int, float))
                and entry.get("epoch") is not None and float(entry["epoch"]) == float(best_epoch)
            ]
            if not history_match:
                problems.append(
                    f"best checkpoint step {step} has no matching log_history entry at epoch {best_epoch!r}"
                )
    if require_final:
        final_epoch = train_metrics.get("final_epoch")
        final_eval_epoch = train_metrics.get("final_eval_epoch")
        if not (isinstance(final_epoch, (int, float)) and float(final_epoch) == 3.0):
            problems.append(f"final_epoch is not 3.0: {final_epoch!r}")
        if not (isinstance(final_eval_epoch, (int, float)) and float(final_eval_epoch) == 3.0):
            problems.append(f"final_eval_epoch is not 3.0: {final_eval_epoch!r}")
    return problems


def trainable_params_problems(train_metrics: dict, spec: ModelSpec) -> list[str]:
    summary = train_metrics.get("trainable_parameters")
    problems: list[str] = []
    if not isinstance(summary, dict) or not summary:
        return ["trainable_parameters summary missing"]
    if summary.get("status") == "unavailable":
        if spec.kind == "vision":
            return [
                "trainable parameter inspection unavailable for a vision config: "
                f"{summary.get('reason')}"
            ]
        return []
    if "count" not in summary or "numel" not in summary:
        return ["trainable_parameters summary missing"]
    if int(summary.get("count") or 0) <= 0 or int(summary.get("numel") or 0) <= 0:
        problems.append(f"trainable_parameters is empty: {summary!r}")
    if spec.kind == "vision":
        vision_count = int(summary.get("vision_like_count") or 0)
        if vision_count > 0:
            problems.append(
                f"{vision_count} vision-like trainable parameters: "
                f"{summary.get('vision_like_names')}"
            )
    return problems


# --------------------------------------------------------------------------- #
# preflight proof
# --------------------------------------------------------------------------- #

def selection_problems(selection: dict, spec: ModelSpec) -> list[str]:
    """Exact identity/counts/lengths for a retained selection.json."""
    problems = []
    if selection.get("model") != spec.slug or selection.get("repo") != spec.repo:
        problems.append("selection metadata model/repo mismatch")
    if selection.get("requested_revision") != spec.revision:
        problems.append("selection metadata requested revision mismatch")
    if selection.get("resolved_revision") != spec.revision:
        problems.append(
            f"selection resolved revision {selection.get('resolved_revision')!r} != "
            f"pinned {spec.revision!r}"
        )
    threshold = selection.get("threshold", MIN_SUPERVISED_TOKENS)
    selected = selection.get("selected") or {}
    for split, need in PREFLIGHT_NEEDS.items():
        picks = selected.get(split) or []
        if len(picks) != need:
            problems.append(f"selection {split}: {len(picks)} rows != {need}")
        short = [pick for pick in picks if int(pick.get("length", 0)) < threshold]
        if short:
            problems.append(f"selection {split}: {len(short)} rows below {threshold} tokens")
    return problems


def preflight_run_problems(spec: ModelSpec, workspace: str | Path) -> list[str]:
    """Validation applied identically to the first and the repeat tiny runs."""
    workspace = Path(workspace)
    problems: list[str] = []
    metrics: dict = {}
    for label, filename in (
        ("baseline", "baseline_metrics.json"),
        ("adapter", "finetuned_metrics.json"),
        ("train", "train_metrics.json"),
    ):
        path = workspace / filename
        if not path.is_file():
            problems.append(f"{label}: missing {filename}")
            continue
        try:
            loaded = read_json(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{label}: unreadable {filename}: {exc}")
            continue
        metrics[label] = loaded

    for label, loaded in metrics.items():
        problems += identity_problems(loaded, spec, label)
        problems += quantization_problems(loaded, label)
    if len(metrics) == 3:
        views = [quantization_view(metrics[label]) for label in ("baseline", "adapter", "train")]
        if any(view != views[0] for view in views[1:]):
            problems.append("baseline/adapter/train effective quantization differ")

    baseline = metrics.get("baseline") or {}
    adapter = metrics.get("adapter") or {}
    train = metrics.get("train") or {}
    if baseline:
        if baseline.get("mode") != "base" or baseline.get("checkpoint") != "base":
            problems.append(
                f"baseline: expected mode/checkpoint 'base', found "
                f"{baseline.get('mode')!r}/{baseline.get('checkpoint')!r}"
            )
        if baseline.get("adapter") is not None:
            problems.append(f"baseline: unexpected adapter {baseline.get('adapter')!r}")
        if baseline.get("examples") != 1:
            problems.append(f"baseline: examples {baseline.get('examples')!r} != 1")
    if adapter:
        if adapter.get("mode") != "adapter":
            problems.append(f"adapter: mode {adapter.get('mode')!r} != 'adapter'")
        expected_adapter = (workspace / "adapter").resolve()
        if not str(adapter.get("adapter") or ""):
            problems.append("adapter: adapter path missing")
        elif _resolved(adapter["adapter"]) != expected_adapter:
            problems.append(
                f"adapter: path {adapter.get('adapter')!r} != {expected_adapter}"
            )
        if not expected_adapter.is_dir():
            problems.append(f"adapter: directory does not exist: {expected_adapter}")
        if adapter.get("examples") != 1:
            problems.append(f"adapter: examples {adapter.get('examples')!r} != 1")
    if train:
        problems += best_checkpoint_problems(train)
        problems += trainable_params_problems(train, spec)

    for label, filename in (
        ("baseline", "baseline_predictions.csv"),
        ("adapter", "finetuned_predictions.csv"),
    ):
        path = workspace / filename
        if not path.is_file():
            problems.append(f"{label}: missing {filename}")
            continue
        try:
            rows = count_csv_data_rows(path)
        except OSError as exc:
            problems.append(f"{label}: unreadable {filename}: {exc}")
            continue
        if rows != 1:
            problems.append(f"{label}: {rows} prediction rows != 1")
    return problems


def run_summary(metrics: dict, spec: ModelSpec) -> dict:
    train = metrics.get("train") or {}
    baseline = metrics.get("baseline") or {}
    adapter = metrics.get("adapter") or {}
    return {
        "peaks": peak_maxima(train, baseline, adapter),
        "offload": any_offload([train, baseline, adapter]),
        "quantization": {
            "train": train.get("quantization"),
            "baseline": baseline.get("quantization"),
            "adapter": adapter.get("quantization"),
        },
        "eval_losses": [
            entry.get("eval_loss")
            for entry in train.get("log_history") or []
            if "eval_loss" in entry
        ],
        "best_model_checkpoint": train.get("best_model_checkpoint"),
        "best_metric": train.get("best_metric"),
        "best_epoch": train.get("best_epoch"),
        "vision_trainable": {
            "vision_like_count": (train.get("trainable_parameters") or {}).get("vision_like_count"),
            "vision_like_names": (train.get("trainable_parameters") or {}).get("vision_like_names"),
        },
    }


def select_preflight_rows(
    rows_by_split: dict[str, list[dict]],
    length_fn,
    *,
    needs: dict | None = None,
    threshold: int | None = None,
) -> dict:
    """First-N deterministic selection of real rows whose rendered prompt is long enough."""
    needs = PREFLIGHT_NEEDS if needs is None else needs
    threshold = MIN_SUPERVISED_TOKENS if threshold is None else threshold
    selected: dict = {}
    missing: dict = {}
    scanned: dict = {}
    for split, need in needs.items():
        rows = rows_by_split.get(split) or []
        picks = []
        for index, row in enumerate(rows):
            length = int(length_fn(row, split))
            if length >= threshold:
                picks.append(
                    {
                        "index": index,
                        "issue_number": row.get("issue_number"),
                        "length": length,
                        "row": row,
                    }
                )
                if len(picks) == need:
                    break
        scanned[split] = len(rows)
        if len(picks) < need:
            missing[split] = need - len(picks)
        selected[split] = picks
    return {"selected": selected, "missing": missing, "scanned": scanned}


def rendered_supervised_length(source, row: dict, system_prompt: str, chat_kwargs: dict) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": row["label"]},
    ]
    rendered = source.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **chat_kwargs
    )
    # transformers returns a BatchEncoding (a Mapping, not a dict subclass).
    if hasattr(rendered, "input_ids"):
        ids = rendered.input_ids
    elif hasattr(rendered, "get"):
        ids = rendered.get("input_ids", rendered)
    else:
        ids = rendered
    if ids and isinstance(ids[0], (list, tuple)):  # single unbatched row
        ids = ids[0]
    return len(ids)


SNAPSHOT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def revision_from_object(source) -> str | None:
    """Exact resolved revision from tokenizer/processor metadata.

    Uses the shared `resolved_commit_hash` helper first, then the hub snapshot
    path embedded in tokenizer/processor init kwargs (transformers 5.5 does not
    expose `_commit_hash` on slow tokenizers). Returns None when unresolved.
    """
    from specialist.model import resolved_commit_hash

    holders = [source, getattr(source, "tokenizer", None), getattr(source, "image_processor", None)]
    resolved = resolved_commit_hash(*holders)
    if resolved:
        return resolved
    for holder in holders:
        if holder is None:
            continue
        values: list[str] = []
        kwargs = getattr(holder, "init_kwargs", None)
        if hasattr(kwargs, "values"):
            values += [value for value in kwargs.values() if isinstance(value, str)]
        for attribute in ("vocab_file", "merges_file", "tokenizer_file", "name_or_path"):
            value = getattr(holder, attribute, None)
            if isinstance(value, str):
                values.append(value)
        for value in values:
            parts = Path(value).parts
            if "snapshots" in parts:
                index = parts.index("snapshots")
                if index + 1 < len(parts) and SNAPSHOT_SHA_RE.fullmatch(parts[index + 1]):
                    return parts[index + 1]
    return None


# --------------------------------------------------------------------------- #
# checkpoints / artifacts
# --------------------------------------------------------------------------- #

def _file_problems(path: Path, label: str) -> list[str]:
    """Existence + non-empty check for a required file."""
    if not path.is_file():
        return [f"{label}: missing"]
    try:
        if path.stat().st_size <= 0:
            return [f"{label}: zero-byte file"]
    except OSError as exc:
        return [f"{label}: unreadable stat: {exc}"]
    return []


def safetensors_problems(path: Path, label: str) -> list[str]:
    """Structural safetensors check (header JSON + declared tensor byte ranges).

    Uses the installed library when available (real interrupted-write detection);
    otherwise validates the same header/offset structure directly. An empty or
    truncated file always fails.
    """
    problems: list[str] = []
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"{label}: unreadable: {exc}"]
    if size <= 8:
        return [f"{label}: truncated safetensors header"]
    try:
        with path.open("rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little")
            if header_length <= 0 or header_length > size - 8:
                return [f"{label}: invalid safetensors header length {header_length}"]
            header = json.loads(handle.read(header_length).decode("utf-8"))
            payload_end = 8 + header_length
            if not isinstance(header, dict) or "__metadata__" not in header or len(header) < 2:
                # a real weights file always declares at least one tensor
                if len([k for k in header if k != "__metadata__"]) < 1:
                    problems.append(f"{label}: no tensors declared in safetensors header")
            for name, spec in header.items():
                if name == "__metadata__" or not isinstance(spec, dict):
                    continue
                offsets = spec.get("data_offsets")
                if (
                    not isinstance(offsets, list) or len(offsets) != 2
                    or not all(isinstance(offset, int) for offset in offsets)
                    or offsets[0] < 0 or offsets[1] < offsets[0]
                ):
                    problems.append(f"{label}: invalid data_offsets for {name!r}")
                    continue
                if payload_end + offsets[1] > size:
                    problems.append(f"{label}: tensor {name!r} data exceeds file size (truncated)")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        problems.append(f"{label}: unreadable safetensors: {exc}")
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                problems.append(f"{label}: safetensors declares no tensors")
    except ImportError:
        pass  # structural check above is the fallback
    except Exception as exc:
        problems.append(f"{label}: safetensors unreadable: {type(exc).__name__}: {exc}")
    return problems


def torch_state_problems(path: Path, label: str) -> list[str]:
    """Load a torch state file on CPU with weights_only; fail closed on any error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is part of the ML env
        return [f"{label}: torch unavailable for validation: {exc}"]
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return [f"{label}: unreadable torch state: {type(exc).__name__}: {exc}"]
    return []


def rng_state_problems(path: Path, label: str) -> list[str]:
    """Validate a Trainer `rng_state.pth` including its NumPy RNG payload.

    Trust boundary: this file is produced by Hugging Face Trainer inside the
    invariant-checked active run (clean git, pinned environment, digest-verified
    data), never from external input, so it is the one checkpoint file loaded
    with `weights_only=False` — the NumPy state (`numpy.random.get_state()`)
    cannot be represented under `weights_only=True`. Corruption is still
    detected against the actual Trainer schema: `python` tuple, valid `numpy`
    tuple, at least one non-empty tensor under `cpu` (or the legacy `torch`
    key), and `cuda` as either one non-empty tensor (current Trainer) or a
    non-empty sequence of non-empty tensors (legacy).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is part of the ML env
        return [f"{label}: torch unavailable for validation: {exc}"]
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [f"{label}: unreadable RNG state: {type(exc).__name__}: {exc}"]
    if not isinstance(state, dict) or not state:
        return [f"{label}: RNG state is not a non-empty dict"]

    def nonempty_tensor(value) -> bool:
        numel = getattr(value, "numel", None)
        return numel is not None and int(numel()) > 0

    problems: list[str] = []
    python_state = state.get("python")
    if not (isinstance(python_state, tuple) and len(python_state) >= 2):
        problems.append(f"{label}: python RNG state is not a state tuple")
    numpy_state = state.get("numpy")
    if not (
        isinstance(numpy_state, tuple)
        and len(numpy_state) >= 2
        and isinstance(numpy_state[0], str)
        and hasattr(numpy_state[1], "shape")
    ):
        problems.append(f"{label}: numpy RNG state is not a numpy state tuple")
    tensor_keys = [name for name in ("cpu", "torch") if name in state]
    if not tensor_keys:
        problems.append(f"{label}: RNG state has no cpu/torch tensor")
    for name in tensor_keys:
        if not nonempty_tensor(state[name]):
            problems.append(f"{label}: {name} RNG state is not a non-empty tensor")
    if "cuda" in state:
        cuda_state = state["cuda"]
        if nonempty_tensor(cuda_state):
            pass
        elif isinstance(cuda_state, (list, tuple)):
            if not cuda_state:
                problems.append(f"{label}: cuda RNG state is an empty sequence")
            else:
                for index, entry in enumerate(cuda_state):
                    if not nonempty_tensor(entry):
                        problems.append(
                            f"{label}: cuda RNG state {index} is not a non-empty tensor"
                        )
        else:
            problems.append(f"{label}: cuda RNG state is neither a tensor nor a sequence")
    return problems


def complete_checkpoint_problems(checkpoint: str | Path) -> list[str]:
    """Complete = named step == trainer_state.global_step, every file non-empty,
    JSON parseable, safetensors/torch states actually loadable."""
    checkpoint = Path(checkpoint)
    problems: list[str] = []
    if not checkpoint.is_dir():
        return [f"checkpoint directory missing: {checkpoint}"]
    name_match = re.search(r"checkpoint-(\d+)$", checkpoint.name)
    if not name_match:
        problems.append(f"{checkpoint}: directory name is not checkpoint-<step>")
    state_path = checkpoint / "trainer_state.json"
    state: dict | None = None
    if not state_path.is_file():
        problems.append(f"{checkpoint.name}: trainer_state.json missing")
    else:
        try:
            state = read_json(state_path)
        except (OSError, ValueError) as exc:
            problems.append(f"{checkpoint.name}: unreadable trainer_state.json: {exc}")
        else:
            if not isinstance(state, dict) or not isinstance(state.get("global_step"), int):
                problems.append(f"{checkpoint.name}: trainer_state.global_step not an int")
            elif name_match and state["global_step"] != int(name_match.group(1)):
                problems.append(
                    f"{checkpoint.name}: global_step {state['global_step']} != directory "
                    f"step {name_match.group(1)}"
                )
    for name in CHECKPOINT_REQUIRED_FILES[1:]:
        problems += _file_problems(checkpoint / name, f"{checkpoint.name}: {name}")
    adapter_config_path = checkpoint / "adapter_config.json"
    if not _file_problems(adapter_config_path, f"{checkpoint.name}: adapter_config.json"):
        try:
            adapter_config = read_json(adapter_config_path)
        except (OSError, ValueError) as exc:
            problems.append(f"{checkpoint.name}: unreadable adapter_config.json: {exc}")
        else:
            if not isinstance(adapter_config, dict):
                problems.append(f"{checkpoint.name}: adapter_config.json is not a JSON object")

    weight = next(
        (checkpoint / name for name in CHECKPOINT_WEIGHT_FILES if (checkpoint / name).is_file()),
        None,
    )
    if weight is None:
        problems.append(f"{checkpoint.name}: no adapter weight file ({CHECKPOINT_WEIGHT_FILES})")
    elif weight.suffix == ".safetensors":
        problems += safetensors_problems(weight, f"{checkpoint.name}: {weight.name}")
    else:
        problems += _file_problems(weight, f"{checkpoint.name}: {weight.name}")

    for name in ("optimizer.pt", "scheduler.pt"):
        path = checkpoint / name
        file_problems = _file_problems(path, f"{checkpoint.name}: {name}")
        problems += file_problems
        if not file_problems:
            problems += torch_state_problems(path, f"{checkpoint.name}: {name}")
    rng_path = checkpoint / "rng_state.pth"
    rng_file_problems = _file_problems(rng_path, f"{checkpoint.name}: rng_state.pth")
    problems += rng_file_problems
    if not rng_file_problems:
        problems += rng_state_problems(rng_path, f"{checkpoint.name}: rng_state.pth")
    return problems


def adapter_artifact_problems(adapter_dir: str | Path) -> list[str]:
    adapter_dir = Path(adapter_dir)
    problems: list[str] = []
    if not adapter_dir.is_dir():
        return [f"adapter directory missing: {adapter_dir}"]
    if not (adapter_dir / "adapter_config.json").is_file():
        problems.append(f"{adapter_dir.name}: adapter_config.json missing")
    if not any((adapter_dir / name).is_file() for name in ADAPTER_WEIGHT_FILES):
        problems.append(f"{adapter_dir.name}: no adapter weight file ({ADAPTER_WEIGHT_FILES})")
    return problems


def _read_metrics(path: Path, label: str) -> tuple[dict | None, list[str]]:
    if not path.is_file():
        return None, [f"{label}: missing {path.name}"]
    try:
        metrics = read_json(path)
    except (OSError, ValueError) as exc:
        return None, [f"{label}: unreadable {path.name}: {exc}"]
    if not isinstance(metrics, dict):
        return None, [f"{label}: {path.name} is not a JSON object"]
    return metrics, []


def artifact_problems(phase: str, context: dict, frozen: dict | None = None) -> list[str]:
    frozen = FROZEN_SPLITS if frozen is None else frozen
    workspace = context["workspace"]
    model_dir = context["model_dir"]
    test_rows = frozen["test"]["rows"]
    problems: list[str] = []
    if phase == "preflight":
        path = model_dir / "preflight" / "preflight.json"
        if not path.is_file():
            return [f"preflight: missing {path.name}"]
        try:
            summary = read_json(path)
        except (OSError, ValueError) as exc:
            return [f"preflight: unreadable {path.name}: {exc}"]
        if summary.get("decision") != "go":
            problems.append(f"preflight: decision {summary.get('decision')!r} != 'go'")
        selection_path = model_dir / "preflight" / "selection.json"
        if not selection_path.is_file():
            problems.append("preflight: missing selection.json")
        else:
            try:
                selection = read_json(selection_path)
            except (OSError, ValueError) as exc:
                problems.append(f"preflight: unreadable selection.json: {exc}")
            else:
                problems += selection_problems(selection, context["spec"])
        return problems
    if phase == "baseline":
        metrics, problems = _read_metrics(workspace / "baseline_metrics.json", "baseline")
        if metrics:
            if metrics.get("mode") != "base":
                problems.append(f"baseline: mode {metrics.get('mode')!r} != 'base'")
            if metrics.get("examples") != test_rows:
                problems.append(f"baseline: examples {metrics.get('examples')!r} != {test_rows}")
        path = workspace / "baseline_predictions.csv"
        if not path.is_file():
            problems.append("baseline: missing baseline_predictions.csv")
        elif count_csv_data_rows(path) != test_rows:
            problems.append(f"baseline: prediction rows != {test_rows}")
        return problems
    if phase == "train":
        metrics, problems = _read_metrics(workspace / "train_metrics.json", "train")
        problems += adapter_artifact_problems(workspace / "adapter")
        if metrics:
            checkpoint = metrics.get("best_model_checkpoint")
            if not checkpoint:
                problems.append("train: best_model_checkpoint missing")
            else:
                problems += complete_checkpoint_problems(checkpoint)
        return problems
    if phase == "adapter_eval":
        metrics, problems = _read_metrics(workspace / "finetuned_metrics.json", "adapter_eval")
        if metrics:
            if metrics.get("mode") != "adapter":
                problems.append(f"adapter_eval: mode {metrics.get('mode')!r} != 'adapter'")
            if metrics.get("examples") != test_rows:
                problems.append(f"adapter_eval: examples {metrics.get('examples')!r} != {test_rows}")
        problems += adapter_artifact_problems(workspace / "adapter")
        path = workspace / "finetuned_predictions.csv"
        if not path.is_file():
            problems.append("adapter_eval: missing finetuned_predictions.csv")
        elif count_csv_data_rows(path) != test_rows:
            problems.append(f"adapter_eval: prediction rows != {test_rows}")
        return problems
    if phase == "comparison":
        path = workspace / "comparison.json"
        if not path.is_file():
            problems.append("comparison: missing comparison.json")
        else:
            try:
                comparison = read_json(path)
            except (OSError, ValueError) as exc:
                problems.append(f"comparison: unreadable comparison.json: {exc}")
            else:
                if not isinstance(comparison, dict) or "test_sha256" not in comparison:
                    problems.append("comparison: missing test_sha256 block")
        return problems
    if phase == "acceptance":
        path = model_dir / "acceptance.json"
        if not path.is_file():
            problems.append("acceptance: missing acceptance.json")
        else:
            try:
                record = read_json(path)
            except (OSError, ValueError) as exc:
                problems.append(f"acceptance: unreadable acceptance.json: {exc}")
            else:
                if record.get("status") != "ok":
                    problems.append(f"acceptance: status {record.get('status')!r} != 'ok'")
        return problems
    if phase == "cleanup":
        path = model_dir / "cleanup.json"
        if not path.is_file():
            problems.append("cleanup: missing cleanup.json")
        else:
            try:
                record = read_json(path)
            except (OSError, ValueError) as exc:
                problems.append(f"cleanup: unreadable cleanup.json: {exc}")
            else:
                if record.get("status") != "ok":
                    problems.append(f"cleanup: status {record.get('status')!r} != 'ok'")
        return problems
    raise RunnerError(f"unknown phase: {phase}")


def completed_model_problems(
    spec: ModelSpec,
    run_dir: str | Path,
    manifest: dict,
    frozen: dict | None = None,
    *,
    require_cleanup: bool = True,
) -> list[str]:
    """Truthful completed-model evidence, excluding deliberately deleted scratch.

    `require_cleanup=False` checks everything except the cleanup phase/record so a
    resume can legitimately finish a pending cleanup after acceptance.
    """
    frozen = FROZEN_SPLITS if frozen is None else frozen
    run_dir = Path(run_dir)
    model_dir = run_dir / "models" / spec.slug
    workspace = model_dir / "workspace"
    test_rows = frozen["test"]["rows"]
    problems: list[str] = []

    run_info_path = model_dir / "run_info.json"
    if not run_info_path.is_file():
        problems.append("completed: missing run_info.json")
    else:
        try:
            run_info = read_json(run_info_path)
        except (OSError, ValueError) as exc:
            problems.append(f"completed: unreadable run_info.json: {exc}")
        else:
            if run_info.get("status") != "completed":
                problems.append(f"completed: run_info status {run_info.get('status')!r}")
            for phase in PHASES:
                if not require_cleanup and phase == "cleanup":
                    continue
                status = (run_info.get("phases") or {}).get(phase) or {}
                if status.get("status") != "ok":
                    problems.append(f"completed: phase {phase} is not ok")

    records = ("acceptance.json", "cleanup.json") if require_cleanup else ("acceptance.json",)
    for name in records:
        path = model_dir / name
        if not path.is_file():
            problems.append(f"completed: missing {name}")
            continue
        try:
            record = read_json(path)
        except (OSError, ValueError) as exc:
            problems.append(f"completed: unreadable {name}: {exc}")
            continue
        if not isinstance(record, dict) or record.get("status") != "ok":
            status = record.get("status") if isinstance(record, dict) else record
            problems.append(f"completed: {name} status {status!r} != 'ok'")

    selection_path = model_dir / "preflight" / "selection.json"
    if not selection_path.is_file():
        problems.append("completed: missing preflight/selection.json")
    else:
        try:
            selection = read_json(selection_path)
        except (OSError, ValueError) as exc:
            problems.append(f"completed: unreadable preflight/selection.json: {exc}")
        else:
            problems += [f"completed: {problem}" for problem in selection_problems(selection, spec)]

    preflight_path = model_dir / "preflight" / "preflight.json"
    if not preflight_path.is_file():
        problems.append("completed: missing preflight/preflight.json")
    else:
        try:
            preflight = read_json(preflight_path)
        except (OSError, ValueError) as exc:
            problems.append(f"completed: unreadable preflight/preflight.json: {exc}")
        else:
            for field, attribute in (
                ("model", "slug"), ("repo", "repo"), ("requested_revision", "revision"),
            ):
                expected = getattr(spec, attribute)
                if preflight.get(field) != expected:
                    problems.append(
                        f"completed: preflight {field} {preflight.get(field)!r} != {expected!r}"
                    )
            if preflight.get("decision") != "go":
                problems.append(f"completed: preflight decision {preflight.get('decision')!r} != 'go'")

    for name, label in (
        ("baseline_metrics.json", "baseline"),
        ("train_metrics.json", "train"),
        ("finetuned_metrics.json", "finetuned"),
    ):
        metrics, metric_problems = _read_metrics(workspace / name, f"completed: {label}")
        if metrics is not None and label in ("baseline", "finetuned"):
            if metrics.get("examples") != test_rows:
                problems.append(
                    f"completed: {label} examples {metrics.get('examples')!r} != {test_rows}"
                )
        problems += metric_problems
    for name, label in (
        ("baseline_predictions.csv", "baseline"),
        ("finetuned_predictions.csv", "finetuned"),
    ):
        path = workspace / name
        if not path.is_file():
            problems.append(f"completed: missing {name}")
        else:
            try:
                rows = count_csv_data_rows(path)
            except OSError as exc:
                problems.append(f"completed: unreadable {name}: {exc}")
            else:
                if rows != test_rows:
                    problems.append(f"completed: {label} prediction rows {rows} != {test_rows}")
    comparison, comparison_problems = _read_metrics(
        workspace / "comparison.json", "completed: comparison"
    )
    if comparison is not None and "test_sha256" not in comparison:
        problems.append("completed: comparison.json missing test_sha256 block")
    problems += comparison_problems

    snapshot = model_dir / "config.yaml"
    if not snapshot.is_file():
        problems.append("completed: missing config.yaml")
    else:
        recorded = (manifest.get("config_snapshot_sha256") or {}).get(spec.slug)
        if recorded is None:
            problems.append(f"completed: no recorded snapshot digest for {spec.slug}")
        elif sha256_file(snapshot) != recorded:
            problems.append(f"completed: config.yaml digest != manifest for {spec.slug}")
    return problems


# --------------------------------------------------------------------------- #
# full acceptance
# --------------------------------------------------------------------------- #

def acceptance_failures(
    *,
    spec: ModelSpec,
    baseline: dict,
    finetuned: dict,
    train: dict,
    comparison: dict,
    baseline_prediction_rows: int,
    finetuned_prediction_rows: int,
    workspace: str | Path,
    vram_peak_reserved_gib: float | None,
    vram_total_gib: float | None = None,
    vram_repeat_approved: bool = False,
    vram_peak_problems: list[str] | None = None,
    frozen: dict | None = None,
) -> list[str]:
    frozen = FROZEN_SPLITS if frozen is None else frozen
    workspace = Path(workspace)
    problems: list[str] = []

    for label, metrics in (("baseline", baseline), ("finetuned", finetuned), ("train", train)):
        problems += identity_problems(metrics, spec, label)
        problems += quantization_problems(metrics, label)
    precision = train.get("precision") or {}
    if precision.get("load_in_4bit") is not True:
        problems.append(f"train: precision.load_in_4bit is not True: {precision.get('load_in_4bit')!r}")

    if quantization_view(baseline) != quantization_view(finetuned):
        problems.append("baseline and finetuned effective quantization differ")
    if quantization_view(train) != quantization_view(baseline):
        problems.append("train and baseline effective quantization differ")

    if baseline.get("mode") != "base" or baseline.get("checkpoint") != "base":
        problems.append(
            f"baseline: expected mode/checkpoint 'base', found "
            f"{baseline.get('mode')!r}/{baseline.get('checkpoint')!r}"
        )
    if finetuned.get("mode") != "adapter":
        problems.append(f"finetuned: mode {finetuned.get('mode')!r} != 'adapter'")
    expected_adapter = (workspace / "adapter").resolve()
    if not str(finetuned.get("adapter") or ""):
        problems.append("finetuned: adapter path missing")
    elif _resolved(finetuned["adapter"]) != expected_adapter:
        problems.append(f"finetuned: adapter {finetuned.get('adapter')!r} != {expected_adapter}")

    sha = train.get("sha256") or {}
    for split, expected in frozen.items():
        if sha.get(split) != expected["sha256"]:
            problems.append(f"train: {split} sha256 {sha.get(split)!r} != {expected['sha256']!r}")
    if baseline.get("test_sha256") != frozen["test"]["sha256"]:
        problems.append("baseline: test_sha256 does not match the frozen test split")
    if finetuned.get("test_sha256") != frozen["test"]["sha256"]:
        problems.append("finetuned: test_sha256 does not match the frozen test split")

    if not comparison:
        problems.append("comparison.json missing or empty")
    else:
        hashes = comparison.get("test_sha256") or {}
        if hashes.get("match") is not True:
            problems.append("comparison: test_sha256 match is not True")
        if hashes.get("baseline") != frozen["test"]["sha256"] or hashes.get("finetuned") != frozen["test"]["sha256"]:
            problems.append("comparison: test_sha256 values do not match the frozen test split")
        if (comparison.get("base_identity") or {}).get("match") is not True:
            problems.append("comparison: base identity match is not True")

    test_rows = frozen["test"]["rows"]
    if baseline.get("examples") != test_rows:
        problems.append(f"baseline: examples {baseline.get('examples')!r} != {test_rows}")
    if finetuned.get("examples") != test_rows:
        problems.append(f"finetuned: examples {finetuned.get('examples')!r} != {test_rows}")
    if baseline_prediction_rows != test_rows:
        problems.append(f"baseline: {baseline_prediction_rows} prediction rows != {test_rows}")
    if finetuned_prediction_rows != test_rows:
        problems.append(f"finetuned: {finetuned_prediction_rows} prediction rows != {test_rows}")
    if train.get("train_examples") != frozen["train"]["rows"]:
        problems.append(
            f"train: train_examples {train.get('train_examples')!r} != {frozen['train']['rows']}"
        )
    if train.get("val_examples") != frozen["val"]["rows"]:
        problems.append(f"train: val_examples {train.get('val_examples')!r} != {frozen['val']['rows']}")

    problems += best_checkpoint_problems(train, require_final=True)
    problems += trainable_params_problems(train, spec)

    ceiling = VRAM_REPEAT_MAX_GIB if vram_repeat_approved else VRAM_GO_GIB
    if vram_peak_problems:
        problems += [f"VRAM: {problem}" for problem in vram_peak_problems]
    if vram_peak_reserved_gib is None or vram_peak_reserved_gib > ceiling:
        problems.append(
            f"VRAM: peak reserved {vram_peak_reserved_gib!r} GiB exceeds the "
            f"{'amber (repeat-approved) ' if vram_repeat_approved else ''}ceiling {ceiling} GiB"
        )
    if (
        vram_total_gib is not None
        and vram_peak_reserved_gib is not None
        and (vram_total_gib - vram_peak_reserved_gib) < VRAM_HEADROOM_MIN_GIB
    ):
        problems.append(
            f"VRAM: headroom {vram_total_gib - vram_peak_reserved_gib:.2f} GiB "
            f"< {VRAM_HEADROOM_MIN_GIB} GiB"
        )
    return problems


# --------------------------------------------------------------------------- #
# cleanup policy
# --------------------------------------------------------------------------- #

def cleanup_plan(
    run_dir: str | Path, model_dir: str | Path, slug: str, short_tmp_root: str | Path
) -> list[Path]:
    """Known-safe descendants deleted only after a model is accepted.

    Preflight attempt scratch is already deleted on a successful go (failed
    attempt diagnostics are retained, so they are not in this list).
    """
    run_dir, model_dir = Path(run_dir), Path(model_dir)
    workspace = model_dir / "workspace"
    return [
        run_dir / ".cache" / slug,
        Path(short_tmp_root) / run_dir.name / slug,
        workspace / "adapter",
        workspace / "trainer",
        workspace / "train.jsonl",
        workspace / "val.jsonl",
        workspace / "test.jsonl",
    ]


def validate_cleanup_targets(
    targets: list[Path],
    allowed_roots: tuple[str | Path, ...],
    protected: tuple[str | Path, ...] | None = None,
) -> None:
    protected = PROTECTED_PATHS if protected is None else protected
    for target in targets:
        resolved = _resolved(target)
        if not any(ensure_within(resolved, root) for root in allowed_roots):
            raise RunnerError(f"cleanup target outside allowed roots: {target}")
        if protected_intersection(resolved, protected):
            raise RunnerError(f"cleanup target intersects a protected path: {target}")


def path_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def delete_targets(
    targets: list[Path],
    allowed_roots: tuple[str | Path, ...],
    protected: tuple[str | Path, ...] | None = None,
) -> dict:
    """Delete validated targets, returning per-target and total bytes removed."""
    validate_cleanup_targets(targets, allowed_roots, protected)
    records = []
    total = 0
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        size = path_size(target)
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
        records.append({"path": str(target), "bytes": size})
        total += size
    return {"targets": records, "bytes_deleted": total, "deleted_at": utc_now()}


# --------------------------------------------------------------------------- #
# subprocess plumbing
# --------------------------------------------------------------------------- #

def cache_paths(run_dir: str | Path, slug: str, short_tmp_root: str | Path) -> dict:
    run_dir = Path(run_dir)
    cache = run_dir / ".cache" / slug
    return {
        "cache_root": cache,
        "hf_home": cache / "hf",
        "hub_cache": cache / "hub",
        "xdg_cache": cache / "xdg",
        "inductor_cache": cache / "inductor",
        "triton_cache": cache / "triton",
        "tmpdir": Path(short_tmp_root) / run_dir.name / slug,
    }


def isolated_env(
    run_dir: str | Path, slug: str, short_tmp_root: str | Path, project_root: str | Path
) -> dict:
    paths = cache_paths(run_dir, slug, short_tmp_root)
    for key in ("hf_home", "hub_cache", "xdg_cache", "inductor_cache", "triton_cache", "tmpdir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(paths["hf_home"]),
            "HUGGINGFACE_HUB_CACHE": str(paths["hub_cache"]),
            "HF_HUB_CACHE": str(paths["hub_cache"]),
            "XDG_CACHE_HOME": str(paths["xdg_cache"]),
            "TORCHINDUCTOR_CACHE_DIR": str(paths["inductor_cache"]),
            "TRITON_CACHE_DIR": str(paths["triton_cache"]),
            "TMPDIR": str(paths["tmpdir"]),
            "PYTHONUNBUFFERED": "1",
        }
    )
    src_dir = str(Path(project_root) / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    return env


def run_phase_subprocess(
    *,
    python: str,
    phase: str,
    config_path: str | Path,
    workspace: str | Path,
    extra: list[str] | None,
    env: dict,
    log_path: str | Path,
) -> int:
    cmd = [python, "-m", "specialist.cli", phase, str(config_path), "--run-dir", str(workspace)]
    if extra:
        cmd += list(extra)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(
            cmd, cwd=str(workspace), env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return result.returncode


def run_selection_subprocess(
    *,
    python: str,
    script_path: str | Path,
    slug: str,
    config_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    meta_path: str | Path,
    env: dict,
    log_path: str | Path,
) -> int:
    cmd = [
        python, str(script_path), "--select-rows",
        "--slug", slug,
        "--config", str(config_path),
        "--data-dir", str(data_dir),
        "--out", str(out_dir),
        "--meta", str(meta_path),
    ]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(
            cmd, cwd=str(Path(script_path).resolve().parent.parent), env=env,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return result.returncode


def cmd_select_rows(args) -> int:
    """Internal tokenizer/processor-only preflight selection (no model weights)."""
    spec = spec_by_slug(args.slug)
    config = load_config(args.config)
    problems = validate_config(config, spec)
    if problems:
        raise RunnerError("selection config invalid:\n  - " + "\n  - ".join(problems))

    model_cfg = config["model"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    repo_kwargs = {"trust_remote_code": trust_remote_code, "revision": spec.revision}
    if spec.kind == "vision":
        from transformers import AutoProcessor

        source = AutoProcessor.from_pretrained(spec.repo, **repo_kwargs)
    else:
        from transformers import AutoTokenizer

        source = AutoTokenizer.from_pretrained(spec.repo, **repo_kwargs)

    from specialist.model import configured_chat_template_kwargs

    resolved = revision_from_object(source)
    if not resolved:
        raise RunnerError(
            f"tokenizer/processor for {spec.repo} exposed no resolved commit hash or "
            "snapshot path; refusing a selection that cannot prove the exact revision"
        )
    if resolved != spec.revision:
        raise RunnerError(
            f"tokenizer/processor resolved revision {resolved} != pinned {spec.revision}"
        )

    chat_kwargs = configured_chat_template_kwargs(config.get("evaluation"))
    system_prompt = model_cfg["system_prompt"]
    rows_by_split = {
        split: read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        for split in PREFLIGHT_NEEDS
    }
    result = select_preflight_rows(
        rows_by_split,
        lambda row, _split: rendered_supervised_length(source, row, system_prompt, chat_kwargs),
    )
    if result["missing"]:
        raise RunnerError(
            f"not enough rows with rendered supervised prompt >= {MIN_SUPERVISED_TOKENS} "
            f"tokens: {result['missing']} (scanned {result['scanned']})"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    for split, picks in result["selected"].items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for pick in picks:
                f.write(json.dumps(pick["row"], ensure_ascii=False) + "\n")

    tokenizer = getattr(source, "tokenizer", source)
    metadata = {
        "model": spec.slug,
        "repo": spec.repo,
        "requested_revision": spec.revision,
        "resolved_revision": resolved,
        "kind": spec.kind,
        "threshold": MIN_SUPERVISED_TOKENS,
        "needs": PREFLIGHT_NEEDS,
        "chat_template_kwargs": chat_kwargs,
        "data_dir": str(args.data_dir),
        "scanned": result["scanned"],
        "selected": {
            split: [
                {"index": pick["index"], "issue_number": pick["issue_number"], "length": pick["length"]}
                for pick in picks
            ]
            for split, picks in result["selected"].items()
        },
        "tokenizer": {
            "class": type(source).__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
        },
        "selected_at": utc_now(),
    }
    write_json_atomic(args.meta, metadata)
    print(f"selection: {metadata['selected']}")
    return 0


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

class Runner:
    def __init__(
        self,
        *,
        project_root: str | Path = PROJECT_ROOT,
        root: str | Path = RUN_ROOT,
        data_dir: str | Path = FROZEN_DATA_DIR,
        short_tmp_root: str | Path = SHORT_TMP_ROOT,
        dry_run: bool = False,
        resume: str | Path | None = None,
        retry_failed: bool = False,
        out=print,
    ):
        self.project_root = _resolved(project_root)
        self.root = _resolved(root)
        self.data_dir = _resolved(data_dir)
        self.short_tmp_root = _resolved(short_tmp_root)
        # Single interpreter: probes, phases and manifest all use this one.
        self.python = str(Path(sys.executable).resolve())
        self.dry_run = dry_run
        self.resume = resume
        self.retry_failed = retry_failed
        self.out = out
        self.run_dir: Path | None = None
        self.manifest: dict | None = None
        self.gpu: dict | None = None

    # -- invariant guard ---------------------------------------------------- #

    def _invariant_check(
        self, *, manifest: dict | None, require_clean_git: bool = True
    ) -> dict:
        """Clean git + same HEAD/config digests/environment/data/GPU before every model."""
        problems: list[str] = []
        versions = probe_versions()
        problems += check_pinned_versions(versions)
        if manifest is not None:
            if manifest.get("environment") != versions:
                problems.append("environment (pinned versions) differs from the run manifest")
            if manifest.get("python") != self.python:
                problems.append(
                    f"python interpreter {self.python!r} != run interpreter {manifest.get('python')!r}"
                )

        gpu = None
        try:
            gpu = probe_gpu_identity()
        except Exception as exc:
            problems.append(f"GPU identity probe failed: {type(exc).__name__}: {exc}")
        if gpu is not None:
            if manifest is not None:
                recorded = manifest.get("gpu") or {}
                if recorded.get("name") != gpu["name"]:
                    problems.append(f"GPU name {gpu['name']!r} != run GPU {recorded.get('name')!r}")
                if abs(float(recorded.get("total_gib") or 0.0) - gpu["total_gib"]) > 0.01:
                    problems.append(
                        f"GPU total {gpu['total_gib']:.2f} GiB != run total "
                        f"{recorded.get('total_gib')!r} GiB"
                    )
                if recorded.get("visible_devices") != gpu["visible_devices"]:
                    problems.append("visible CUDA device count differs from the run manifest")
        processes = probe_gpu_compute_processes()
        if processes:
            problems.append(f"foreign GPU compute processes active: {processes}")

        data_report, data_problems = frozen_data_report(self.data_dir)
        problems += data_problems
        if manifest is not None:
            recorded_splits = (manifest.get("data") or {}).get("splits") or {}
            for name, entry in data_report.items():
                recorded = recorded_splits.get(name) or {}
                if recorded.get("sha256") != entry.get("sha256") or recorded.get("rows") != entry.get("rows"):
                    problems.append(f"frozen {name} data differs from the run manifest")

        free_gib = probe_disk_free_gib(existing_ancestor(self.root))
        if free_gib < MIN_FREE_GIB:
            problems.append(f"free disk {free_gib:.1f} GiB < {MIN_FREE_GIB} GiB")

        if require_clean_git:
            dirty = probe_git_status(self.project_root)
            allowed_untracked = (
                (f"benchmarks/{TRACK}/{self.run_dir.name}/",) if self.run_dir is not None else ()
            )
            problems += git_tree_problems(dirty, allowed_untracked=allowed_untracked)
            if manifest is not None:
                head = probe_git_head(self.project_root)
                if manifest.get("git_commit") != head:
                    problems.append(
                        f"git commit {head} != run commit {manifest.get('git_commit')}"
                    )

        for spec in EXPECTED_MODELS:
            path = self.project_root / spec.config
            if not path.is_file():
                problems.append(f"missing config: {path}")
                continue
            try:
                config = load_config(path)
            except yaml.YAMLError as exc:
                problems.append(f"config {path} is not valid YAML: {exc}")
                continue
            problems += validate_config(config, spec, data_dir=self.data_dir)
            if manifest is not None:
                recorded = (manifest.get("config_sha256") or {}).get(spec.slug)
                actual = sha256_file(path)
                if recorded is None:
                    problems.append(f"{spec.slug}: config digest missing from run manifest")
                elif recorded != actual:
                    problems.append(
                        f"{spec.slug}: source config sha256 {actual} != manifest {recorded}"
                    )

        problems += validate_namespace(self.root, self.project_root)

        if problems:
            raise RunnerError("invariant check failed:\n  - " + "\n  - ".join(problems))
        return {"versions": versions, "gpu": gpu, "data": data_report, "free_gib": free_gib}

    def _dry_run(self) -> int:
        summary = self._invariant_check(manifest=None, require_clean_git=True)
        gpu = summary["gpu"]
        self.out("DRY RUN OK  (no download, no GPU model load, no phase subprocess)")
        self.out(f"  track root      : {self.root}")
        self.out(f"  run namespace   : {self.root}/<UTC-run-id>/")
        self.out(f"  python          : {self.python}")
        self.out(f"  free disk       : {summary['free_gib']:.1f} GiB (>= {MIN_FREE_GIB})")
        self.out(
            f"  GPU (torch)     : {gpu['name']} "
            f"{gpu['total_gib']:.2f} GiB usable, {gpu['visible_devices']} visible"
        )
        for spec in EXPECTED_MODELS:
            self.out(
                f"  {spec.index}. {spec.slug}: {spec.repo}@{spec.revision} "
                f"kind={spec.kind} batch={spec.per_device_train_batch_size} "
                f"ga={spec.gradient_accumulation_steps} collator={spec.vision_collator}"
            )
        self.out(
            "  frozen data     : "
            + ", ".join(
                f"{name}={entry['rows']} rows/{entry['sha256'][:12]}"
                for name, entry in summary["data"].items()
            )
        )
        self.out(
            f"  preflight needs : {PREFLIGHT_NEEDS} rows >= {MIN_SUPERVISED_TOKENS} tokens; "
            f"VRAM gate <= {VRAM_GO_GIB} GiB (repeat to {VRAM_REPEAT_MAX_GIB})"
        )
        return 0

    def _snapshot_digest_check(self, spec: ModelSpec, manifest: dict, snapshot: Path) -> None:
        digest = sha256_file(snapshot)
        recorded = (manifest.get("config_snapshot_sha256") or {}).get(spec.slug)
        if recorded is None:
            manifest.setdefault("config_snapshot_sha256", {})[spec.slug] = digest
            self._persist_manifest()
        elif recorded != digest:
            raise RunnerError(
                f"{spec.slug}: config snapshot sha256 {digest} != manifest {recorded}; "
                "refusing to continue"
            )

    # -- run lifecycle ------------------------------------------------------ #

    def run(self) -> int:
        try:
            if self.dry_run:
                return self._dry_run()
            if self.resume:
                return self._resume_run(Path(self.resume))
            return self._new_run()
        except KeyboardInterrupt:
            self._mark_aborted("interrupted by SIGINT")
            raise
        except Exception as exc:
            self._mark_aborted(f"{type(exc).__name__}: {exc}")
            raise

    def _mark_aborted(self, reason: str) -> None:
        if self.manifest is None or self.run_dir is None:
            return
        self._normalize_attempts_on_abort(reason)
        manifest = self.manifest
        manifest["last_error"] = reason
        try:
            self._finalize_manifest(manifest)
        except RunnerError:
            pass  # LATEST failure is already recorded in the manifest

    def _normalize_attempts_on_abort(self, reason: str) -> None:
        """Persist interrupted attempts/model status so a resume can recover truthfully."""
        for spec in EXPECTED_MODELS:
            path = self.run_dir / "models" / spec.slug / "run_info.json"
            if not path.is_file():
                continue
            try:
                run_info = read_json(path)
            except (OSError, ValueError):
                continue
            running_phase = None
            for phase in PHASES:
                if (run_info.get("phases", {}).get(phase) or {}).get("status") == "running":
                    running_phase = phase
            self._normalize_run_info(run_info)
            if run_info.get("status") in (None, "running"):
                run_info["status"] = "failed"
                run_info["failed_at"] = utc_now()
                run_info["failure"] = {
                    "phase": running_phase,
                    "error": reason,
                    "retryable": True,
                }
            try:
                write_json_atomic(path, run_info)
            except OSError:
                pass

    def _persist_manifest(self) -> None:
        if self.manifest is not None and self.run_dir is not None:
            write_json_atomic(self.run_dir / "manifest.json", self.manifest)

    def _finalize_manifest(self, manifest: dict) -> None:
        """Persist the truthful manifest first, then atomically update track LATEST."""
        statuses = [model["status"] for model in manifest["models"]]
        if statuses and all(status == "completed" for status in statuses):
            manifest["status"] = "completed"
            manifest["finished_at"] = utc_now()
            manifest["latest_written"] = False
            manifest["latest_error"] = None
            self._persist_manifest()  # completed manifest on disk before LATEST
            try:
                write_track_latest(self.root, manifest["run_id"])
            except Exception as exc:
                manifest["latest_error"] = f"{type(exc).__name__}: {exc}"
                self._persist_manifest()
                raise RunnerError(f"track LATEST not updated after completion: {exc}") from exc
            manifest["latest_written"] = True
            self._persist_manifest()
            return
        manifest["status"] = "partial" if any(status == "completed" for status in statuses) else "failed"
        manifest["finished_at"] = utc_now()
        manifest["latest_written"] = False
        self._persist_manifest()

    def _new_run(self) -> int:
        summary = self._invariant_check(manifest=None, require_clean_git=True)
        self.gpu = summary["gpu"]
        self.root.mkdir(parents=True, exist_ok=True)
        run_id = new_run_id(self.root)
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True)

        manifest = {
            "track": TRACK,
            "run_id": run_id,
            "status": "running",
            "created_at": utc_now(),
            "finished_at": None,
            "last_error": None,
            "git_commit": probe_git_head(self.project_root),
            "python": self.python,
            "environment": summary["versions"],
            "gpu": summary["gpu"],
            "config_sha256": self._config_hashes(),
            "config_snapshot_sha256": {},
            "data": {"source": str(self.data_dir), "splits": summary["data"]},
            "vram": {
                "go_gib": VRAM_GO_GIB,
                "repeat_max_gib": VRAM_REPEAT_MAX_GIB,
                "total_gib": summary["gpu"]["total_gib"],
            },
            "models": [
                {
                    "index": spec.index,
                    "slug": spec.slug,
                    "repo": spec.repo,
                    "revision": spec.revision,
                    "kind": spec.kind,
                    "status": "pending",
                    "failure": None,
                    "run_info": str(Path("models") / spec.slug / "run_info.json"),
                }
                for spec in EXPECTED_MODELS
            ],
        }
        self.manifest = manifest
        self._persist_manifest()
        self.out(f"Run directory: {self.run_dir}")

        for spec in EXPECTED_MODELS:
            entry = self._manifest_model(spec)
            entry["status"] = "running"
            self._persist_manifest()
            try:
                self._run_model(spec, manifest)
                self._require_completed_model(spec, manifest)
            except PhaseFailure as exc:
                entry["status"] = "failed"
                entry["failure"] = {"phase": exc.phase, "error": str(exc), "retryable": exc.retryable}
                self._persist_manifest()
                raise RunnerError(f"model {spec.slug} failed: {exc}") from exc
            except RunnerError as exc:
                entry["status"] = "failed"
                entry["failure"] = {"phase": "completed_model", "error": str(exc), "retryable": False}
                self._persist_manifest()
                raise
            entry["status"] = "completed"
            entry["failure"] = None
            self._persist_manifest()

        self._finalize_manifest(manifest)
        if manifest["status"] == "completed":
            self.out(f"Track complete: {manifest['run_id']}")
        return 0

    def _resume_run(self, run_dir: Path) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir = _resolved(run_dir)
        if not ensure_within(run_dir, self.root):
            raise RunnerError(f"resume run dir is outside the track root {self.root}: {run_dir}")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RunnerError(f"no manifest.json in resume dir: {run_dir}")
        manifest = read_json(manifest_path)
        if manifest.get("track") != TRACK:
            raise RunnerError(f"manifest track {manifest.get('track')!r} != {TRACK!r}")
        self.run_dir = run_dir
        self.manifest = manifest
        manifest["finished_at"] = None
        manifest["last_error"] = None
        self._persist_manifest()

        summary = self._invariant_check(manifest=manifest, require_clean_git=True)
        self.gpu = summary["gpu"]
        self.out(f"Resuming run: {run_dir} (status={manifest.get('status')})")

        for spec in EXPECTED_MODELS:
            entry = self._manifest_model(spec)
            if entry["status"] == "completed":
                problems = completed_model_problems(spec, run_dir, manifest)
                if not problems:
                    self.out(f"[{spec.slug}] completed; skipped")
                    continue
                base_problems = completed_model_problems(
                    spec, run_dir, manifest, require_cleanup=False
                )
                if base_problems:
                    path = run_dir / "models" / spec.slug / "run_info.json"
                    if path.is_file():
                        try:
                            run_info = read_json(path)
                        except (OSError, ValueError):
                            run_info = None
                        if run_info is not None:
                            run_info["status"] = "failed"
                            run_info["failed_at"] = utc_now()
                            run_info["failure"] = {
                                "phase": "runner_setup",
                                "error": "completed evidence invalid: " + "; ".join(problems),
                                "retryable": False,
                            }
                            try:
                                write_json_atomic(path, run_info)
                            except OSError:
                                pass
                    entry["status"] = "failed"
                    entry["failure"] = {
                        "phase": "completed_model",
                        "error": "completed evidence invalid: " + "; ".join(problems),
                        "retryable": False,
                    }
                    self._persist_manifest()
                    raise RunnerError(
                        f"model {spec.slug} is marked completed but evidence is invalid: "
                        + "; ".join(problems)
                    )
                self.out(f"[{spec.slug}] cleanup pending after acceptance; resuming cleanup")
            # Mark the resumed entry running before any work happens.
            entry["status"] = "running"
            entry["failure"] = None
            self._persist_manifest()
            try:
                self._run_model(spec, manifest)
                self._require_completed_model(spec, manifest)
            except PhaseFailure as exc:
                entry["status"] = "failed"
                entry["failure"] = {"phase": exc.phase, "error": str(exc), "retryable": exc.retryable}
                self._persist_manifest()
                raise RunnerError(f"model {spec.slug} failed: {exc}") from exc
            except RunnerError as exc:
                entry["status"] = "failed"
                entry["failure"] = {"phase": "completed_model", "error": str(exc), "retryable": False}
                self._persist_manifest()
                raise
            entry["status"] = "completed"
            entry["failure"] = None
            self._persist_manifest()

        self._finalize_manifest(manifest)
        if manifest["status"] == "completed":
            self.out(f"Track complete: {manifest['run_id']}")
        return 0

    def _require_completed_model(self, spec: ModelSpec, manifest: dict) -> None:
        problems = completed_model_problems(spec, self.run_dir, manifest)
        if problems:
            raise RunnerError(
                f"model {spec.slug} completed but evidence is invalid: " + "; ".join(problems)
            )

    def _config_hashes(self) -> dict:
        hashes = {}
        for spec in EXPECTED_MODELS:
            path = self.project_root / spec.config
            hashes[spec.slug] = sha256_file(path) if path.is_file() else None
        return hashes

    def _manifest_model(self, spec: ModelSpec) -> dict:
        for entry in self.manifest["models"]:
            if entry["slug"] == spec.slug:
                return entry
        raise RunnerError(f"manifest has no entry for {spec.slug}")

    # -- per-model execution ------------------------------------------------ #

    def _run_model(self, spec: ModelSpec, manifest: dict) -> None:
        """Setup wrapper: any non-phase exception becomes a truthful runner_setup failure."""
        model_dir = self.run_dir / "models" / spec.slug
        try:
            self._run_model_inner(spec, manifest, model_dir)
        except PhaseFailure:
            raise
        except Exception as exc:
            self._record_setup_failure(spec, model_dir, exc)
            raise PhaseFailure(
                "runner_setup", f"{type(exc).__name__}: {exc}", retryable=False
            ) from exc

    def _record_setup_failure(self, spec: ModelSpec, model_dir: Path, exc: Exception) -> None:
        path = model_dir / "run_info.json"
        run_info = None
        if path.is_file():
            try:
                run_info = read_json(path)
            except (OSError, ValueError):
                run_info = None
        if run_info is None:
            run_info = {
                "slug": spec.slug,
                "repo": spec.repo,
                "revision": spec.revision,
                "kind": spec.kind,
                "created_at": utc_now(),
                "phases": {},
                "cleanup": None,
            }
        running_phase = next(
            (
                phase for phase in PHASES
                if (run_info.get("phases", {}).get(phase) or {}).get("status") == "running"
            ),
            None,
        )
        self._normalize_run_info(run_info)
        run_info["status"] = "failed"
        run_info["failed_at"] = utc_now()
        run_info["failure"] = {
            "phase": "runner_setup",
            "error": f"{type(exc).__name__}: {exc}",
            "retryable": False,
            "related_phase": running_phase,
        }
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, run_info)
        except OSError:
            pass

    def _run_model_inner(self, spec: ModelSpec, manifest: dict, model_dir: Path) -> None:
        run_info_path = model_dir / "run_info.json"
        if run_info_path.is_file():
            run_info = read_json(run_info_path)
        else:
            run_info = {
                "slug": spec.slug,
                "repo": spec.repo,
                "revision": spec.revision,
                "kind": spec.kind,
                "created_at": utc_now(),
                "status": "running",
                "config_sha256": (manifest.get("config_sha256") or {}).get(spec.slug),
                "data": manifest.get("data"),
                "phases": {},
                "cleanup": None,
            }
        self._normalize_run_info(run_info)
        self._check_phase_consistency(spec, run_info)

        # Invariant guard before every model.
        self._invariant_check(manifest=manifest, require_clean_git=True)

        if run_info.get("status") == "completed":
            problems = completed_model_problems(spec, self.run_dir, manifest)
            if not problems:
                return
            cleanup_only = completed_model_problems(
                spec, self.run_dir, manifest, require_cleanup=False
            )
            if cleanup_only:
                run_info["status"] = "failed"
                run_info["failed_at"] = utc_now()
                run_info["failure"] = {
                    "phase": "runner_setup",
                    "error": "completed evidence invalid: " + "; ".join(problems),
                    "retryable": False,
                }
                model_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(run_info_path, run_info)
                raise PhaseFailure(
                    "runner_setup", "completed evidence invalid: " + "; ".join(problems),
                    retryable=False,
                )
            # Everything except cleanup is complete: continue the phase loop so the
            # pending cleanup can finish (accepted phases are skipped).

        model_dir.mkdir(parents=True, exist_ok=True)
        workspace = model_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        config_snapshot = model_dir / "config.yaml"
        source_config = self.project_root / spec.config
        if config_snapshot.is_file():
            config = load_config(config_snapshot)
        else:
            config = load_config(source_config)
            config_snapshot.write_text(
                f"# Snapshot of {source_config} taken {utc_now()}\n"
                + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        config_problems = validate_config(config, spec, data_dir=self.data_dir)
        if config_problems:
            raise RunnerError(
                f"{spec.slug}: snapshot config is not the fixed track config:\n  - "
                + "\n  - ".join(config_problems)
            )
        self._snapshot_digest_check(spec, manifest, config_snapshot)
        run_info["config_source"] = str(source_config)
        run_info["config_snapshot"] = str(config_snapshot)
        run_info["config_sha256"] = (
            manifest.get("config_sha256") or {}
        ).get(spec.slug) or sha256_file(source_config)

        copied = copy_frozen_into_workspace(self.data_dir, workspace)
        run_info["data_copies"] = copied

        env = isolated_env(self.run_dir, spec.slug, self.short_tmp_root, self.project_root)
        context = {
            "spec": spec,
            "model_dir": model_dir,
            "workspace": workspace,
            "config_snapshot": config_snapshot,
            "config": config,
            "env": env,
            "run_info": run_info,
            "run_info_path": run_info_path,
            "log_dir": model_dir / "logs",
        }
        run_info["status"] = "running"
        run_info.pop("failed_at", None)
        run_info.pop("failure", None)
        self._save_run_info(context)

        try:
            for phase in PHASES:
                self._run_phase(context, phase)
        except PhaseFailure as exc:
            run_info["status"] = "failed"
            run_info["failed_at"] = utc_now()
            run_info["failure"] = {"phase": exc.phase, "error": str(exc), "retryable": exc.retryable}
            self._save_run_info(context)
            raise
        run_info["status"] = "completed"
        run_info["completed_at"] = utc_now()
        run_info.pop("failed_at", None)
        run_info.pop("failure", None)
        self._save_run_info(context)
        completion = completed_model_problems(spec, self.run_dir, manifest)
        if completion:
            run_info["status"] = "failed"
            run_info["failed_at"] = utc_now()
            run_info["failure"] = {
                "phase": "runner_setup",
                "error": "completion evidence invalid: " + "; ".join(completion),
                "retryable": False,
            }
            self._save_run_info(context)
            raise PhaseFailure(
                "runner_setup", "completion evidence invalid: " + "; ".join(completion),
                retryable=False,
            )
        cleanup_bytes = (run_info.get("cleanup") or {}).get("bytes_deleted", 0)
        self.out(f"[{spec.slug}] completed; deleted {cleanup_bytes / 1024**2:.1f} MiB of scratch")

    def _normalize_run_info(self, run_info: dict) -> None:
        for phase in PHASES:
            info = run_info["phases"].get(phase)
            if not info:
                continue
            for attempt in info.get("attempts") or []:
                if attempt.get("status") == "running":
                    attempt["status"] = "failed"
                    attempt["error"] = attempt.get("error") or "interrupted (external)"
                    # Explicit external interruption is in the transient allowlist.
                    attempt["retryable"] = True
                    attempt["ended"] = attempt.get("ended") or utc_now()
            if info.get("status") == "running":
                info["status"] = "failed"
                info["failed_at"] = info.get("failed_at") or utc_now()

    def _check_phase_consistency(self, spec: ModelSpec, run_info: dict) -> None:
        seen_unfinished = False
        for phase in PHASES:
            info = run_info["phases"].get(phase) or {"status": "pending"}
            status = info.get("status", "pending")
            if status == "ok" and seen_unfinished:
                raise RunnerError(
                    f"{spec.slug}: inconsistent phase history: {phase} is ok after an "
                    "unfinished phase; refusing to continue"
                )
            if status != "ok":
                seen_unfinished = True

    def _save_run_info(self, context: dict) -> None:
        write_json_atomic(context["run_info_path"], context["run_info"])

    def _run_phase(self, context: dict, phase: str) -> None:
        spec = context["spec"]
        run_info = context["run_info"]
        info = run_info["phases"].setdefault(phase, {"status": "pending", "attempts": []})

        if info.get("status") == "ok":
            # Once acceptance is ok the workspace may already have been cleaned
            # (resume after a crash between acceptance and cleanup), so earlier
            # phases are skipped without re-validation; acceptance itself and
            # cleanup are always validated.
            acceptance_ok = (run_info["phases"].get("acceptance") or {}).get("status") == "ok"
            if acceptance_ok and phase not in ("acceptance", "cleanup"):
                self.out(f"[{spec.slug}] {phase}: skipped (already successful)")
                return
            problems = artifact_problems(phase, context)
            if problems:
                raise RunnerError(
                    f"{spec.slug}/{phase}: marked successful but artifacts are invalid "
                    f"({problems}); refusing to rerun"
                )
            self.out(f"[{spec.slug}] {phase}: skipped (already successful)")
            return

        attempts = info.get("attempts") or []
        if attempts:
            last = attempts[-1]
            if not self.retry_failed:
                raise RunnerError(
                    f"{spec.slug}/{phase} previously failed ({last.get('error')}); "
                    "re-run with --resume <run-dir> --retry-failed to retry once"
                )
            if last.get("retryable") is not True:
                raise RunnerError(
                    f"{spec.slug}/{phase} failure is not in the transient allowlist and is "
                    f"not retryable ({last.get('error')})"
                )
            if len(attempts) >= 2:
                raise RunnerError(
                    f"{spec.slug}/{phase} already used its one retry ({last.get('error')})"
                )

        attempt_no = len(attempts) + 1
        attempt = {"attempt": attempt_no, "status": "running", "started": utc_now()}
        if attempt_no > 1:
            attempt["retry"] = self._prepare_retry(context, phase, attempt_no)
        attempts.append(attempt)
        info["attempts"] = attempts
        info["status"] = "running"
        info.pop("failed_at", None)
        info.pop("finished_at", None)
        self._save_run_info(context)
        started = time.perf_counter()
        self.out(f"[{spec.slug}] {phase}: attempt {attempt_no} ...")
        try:
            self._phase_action(context, phase, attempt_no)
            problems = artifact_problems(phase, context)
            if problems:
                raise PhaseFailure(
                    phase, "invalid artifacts: " + "; ".join(problems), retryable=False
                )
        except PhaseFailure as exc:
            attempt.update(
                status="failed", ended=utc_now(),
                seconds=round(time.perf_counter() - started, 3),
                error=str(exc), retryable=exc.retryable, log=exc.log,
            )
            info["status"] = "failed"
            info["failed_at"] = utc_now()
            self._save_run_info(context)
            self.out(f"[{spec.slug}] {phase}: FAILED ({exc})")
            raise
        except Exception as exc:  # unknown/in-process bugs are hard stops
            attempt.update(
                status="failed", ended=utc_now(),
                seconds=round(time.perf_counter() - started, 3),
                error=f"{type(exc).__name__}: {exc}", retryable=False,
            )
            info["status"] = "failed"
            info["failed_at"] = utc_now()
            self._save_run_info(context)
            self.out(f"[{spec.slug}] {phase}: FAILED ({exc})")
            raise PhaseFailure(phase, f"{type(exc).__name__}: {exc}", retryable=False) from exc

        attempt.update(status="ok", ended=utc_now(), seconds=round(time.perf_counter() - started, 3))
        info["status"] = "ok"
        info["finished_at"] = utc_now()
        info.pop("failed_at", None)
        self._save_run_info(context)
        self.out(f"[{spec.slug}] {phase}: ok ({attempt['seconds']:.1f}s)")

    def _phase_action(self, context: dict, phase: str, attempt_no: int) -> None:
        if phase == "preflight":
            self._preflight(context, attempt_no)
        elif phase == "baseline":
            self._run_subprocess(context, phase, "baseline", context["workspace"], None,
                                 phase, attempt_no)
        elif phase == "train":
            config_path = context.get("active_train_config") or context["config_snapshot"]
            self._run_subprocess(context, phase, "train", context["workspace"], None,
                                 phase, attempt_no, config_path=config_path)
        elif phase == "adapter_eval":
            self._run_subprocess(
                context, phase, "evaluate", context["workspace"],
                ["--checkpoint", str(context["workspace"] / "adapter")], phase, attempt_no,
            )
        elif phase == "comparison":
            comparison = build_comparison(context["workspace"], context["config"])
            write_json_atomic(context["workspace"] / "comparison.json", comparison)
        elif phase == "acceptance":
            self._acceptance(context)
        elif phase == "cleanup":
            self._cleanup(context)
        else:  # pragma: no cover - defensive
            raise RunnerError(f"unknown phase: {phase}")

    # -- retry preparation --------------------------------------------------- #

    def _prepare_retry(self, context: dict, phase: str, attempt_no: int) -> dict:
        if phase == "train":
            return self._prepare_train_retry(context, attempt_no)
        names = PHASE_PARTIAL_FILES.get(phase, ())
        targets = [context["workspace"] / name for name in names]
        targets = [path for path in targets if path.exists() or path.is_symlink()]
        if not targets:
            return {"mode": "clean", "cleared": {"targets": [], "bytes_deleted": 0}}
        result = delete_targets(targets, allowed_roots=(self.run_dir,))
        return {"mode": "clean", "cleared": result}

    def _prepare_train_retry(self, context: dict, attempt_no: int) -> dict:
        workspace = context["workspace"]
        trainer_dir = workspace / "trainer"
        candidates: list[Path] = []
        if trainer_dir.is_dir():
            for path in trainer_dir.glob("checkpoint-*"):
                if path.is_dir() and re.search(r"checkpoint-\d+$", path.name):
                    candidates.append(path)
        candidates.sort(key=lambda p: int(p.name.rsplit("-", 1)[1]), reverse=True)

        incomplete: list[str] = []
        for checkpoint in candidates:
            problems = complete_checkpoint_problems(checkpoint)
            if problems:
                incomplete.append(str(checkpoint))
                continue
            config = load_config(context["config_snapshot"])
            config.setdefault("training", {})["resume_from_checkpoint"] = str(checkpoint)
            config_path = context["log_dir"] / f"train-a{attempt_no}-resume-config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                f"# Retained train retry config for attempt {attempt_no}; resumes from {checkpoint}\n"
                + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            context["active_train_config"] = config_path
            return {
                "mode": "resume",
                "checkpoint": str(checkpoint),
                "step": int(checkpoint.name.rsplit("-", 1)[1]),
                "config": str(config_path),
            }

        # No complete checkpoint: fresh identical restart; never resume a partial one.
        targets = [workspace / "adapter", workspace / "trainer", workspace / "train_metrics.json"]
        targets = [path for path in targets if path.exists() or path.is_symlink()]
        deleted = delete_targets(targets, allowed_roots=(self.run_dir,)) if targets else {
            "targets": [], "bytes_deleted": 0
        }
        context.pop("active_train_config", None)
        return {
            "mode": "fresh_restart",
            "incomplete_checkpoints": incomplete,
            "deleted": deleted,
        }

    # -- subprocess phases ---------------------------------------------------- #

    def _run_subprocess(
        self, context: dict, phase: str, phase_arg: str, workspace: Path, extra,
        log_stem: str, attempt: int, config_path: Path | None = None,
    ) -> None:
        log_path = context["log_dir"] / f"{log_stem}-a{attempt}.log"
        code = run_phase_subprocess(
            python=self.python,
            phase=phase_arg,
            config_path=config_path or context["config_snapshot"],
            workspace=workspace,
            extra=extra,
            env=context["env"],
            log_path=log_path,
        )
        if code != 0:
            tail = read_tail(log_path)
            retryable, reason = classify_failure(tail, code)
            raise PhaseFailure(
                phase, f"{phase_arg} exit code {code} ({reason})",
                retryable=retryable, log=str(log_path),
            )

    # -- preflight ------------------------------------------------------------ #

    def _preflight(self, context: dict, attempt: int) -> None:
        spec = context["spec"]
        model_dir = context["model_dir"]
        pre_dir = model_dir / "preflight"
        attempt_dir = pre_dir / f"attempt-{attempt}"
        if attempt_dir.exists():
            raise PhaseFailure(
                "preflight", f"attempt workspace already exists: {attempt_dir}", retryable=False
            )
        selection_path = pre_dir / "selection.json"
        data_dir = attempt_dir / "data"

        log_path = context["log_dir"] / f"selection-a{attempt}.log"
        code = run_selection_subprocess(
            python=self.python,
            script_path=SCRIPT_PATH,
            slug=spec.slug,
            config_path=context["config_snapshot"],
            data_dir=self.data_dir,
            out_dir=data_dir,
            meta_path=selection_path,
            env=context["env"],
            log_path=log_path,
        )
        if code != 0:
            tail = read_tail(log_path)
            retryable, reason = classify_failure(tail, code)
            raise PhaseFailure("preflight", f"selection exit code {code} ({reason})",
                               retryable=retryable, log=str(log_path))
        selection = read_json(selection_path)
        selection_problems = self._selection_problems(selection, spec)
        if selection_problems:
            raise PhaseFailure("preflight", "; ".join(selection_problems), retryable=False)

        first_ws = attempt_dir / "first" / "ws"
        first_problems = self._preflight_tiny_run(context, attempt, data_dir, first_ws, "first")
        if first_problems:
            raise PhaseFailure("preflight", "first run invalid: " + "; ".join(first_problems),
                               retryable=False)
        first_metrics = self._read_preflight_metrics(first_ws)
        summary: dict = {
            "model": spec.slug,
            "repo": spec.repo,
            "requested_revision": spec.revision,
            "kind": spec.kind,
            "attempt": attempt,
            "selection": selection,
            "token_lengths": {
                split: [
                    pick.get("length")
                    for pick in (selection.get("selected") or {}).get(split) or []
                ]
                for split in PREFLIGHT_NEEDS
            },
            "first": run_summary(first_metrics, spec),
            "repeat": None,
            "total_gpu_gib": self.gpu["total_gib"] if self.gpu else None,
        }
        peaks = summary["first"]["peaks"]
        if peaks["problems"]:
            raise PhaseFailure("preflight", "invalid peak metrics: " + "; ".join(peaks["problems"]),
                               retryable=False)
        decision, reason = vram_decision(
            peaks["reserved_gib"], summary["total_gpu_gib"],
            offload=summary["first"]["offload"],
        )
        if decision == "repeat":
            self.out(f"[{spec.slug}] preflight amber ({reason}); repeating once in fresh scratch")
            repeat_ws = attempt_dir / "repeat" / "ws"
            repeat_problems = self._preflight_tiny_run(context, attempt, data_dir, repeat_ws, "repeat")
            if repeat_problems:
                raise PhaseFailure("preflight", "repeat run invalid: " + "; ".join(repeat_problems),
                                   retryable=False)
            repeat_metrics = self._read_preflight_metrics(repeat_ws)
            summary["repeat"] = run_summary(repeat_metrics, spec)
            repeat_peaks = summary["repeat"]["peaks"]
            if repeat_peaks["problems"]:
                raise PhaseFailure(
                    "preflight", "invalid repeat peak metrics: " + "; ".join(repeat_peaks["problems"]),
                    retryable=False,
                )
            decision, reason = vram_decision(
                peaks["reserved_gib"], summary["total_gpu_gib"],
                offload=summary["first"]["offload"] or summary["repeat"]["offload"],
                repeat_gib=repeat_peaks["reserved_gib"],
            )
        summary["decision"] = decision
        summary["decision_reason"] = reason
        if decision == "go":
            summary["scratch_deleted"] = delete_targets(
                [attempt_dir], allowed_roots=(self.run_dir, self.short_tmp_root)
            )
        write_json_atomic(pre_dir / "preflight.json", summary)
        if decision != "go":
            raise PhaseFailure("preflight", reason, retryable=False)

    def _preflight_tiny_run(
        self, context: dict, attempt: int, data_dir: Path, ws: Path, label: str
    ) -> list[str]:
        ws.mkdir(parents=True, exist_ok=False)
        for split in PREFLIGHT_NEEDS:
            shutil.copy2(data_dir / f"{split}.jsonl", ws / f"{split}.jsonl")
        steps = (
            ("train", "train", None),
            ("baseline", "baseline", None),
            ("adapter", "evaluate", ["--checkpoint", str(ws / "adapter")]),
        )
        for name, phase_arg, extra in steps:
            self._run_subprocess(
                context, "preflight", phase_arg, ws, extra,
                f"preflight-{label}-{name}", attempt,
            )
        return preflight_run_problems(context["spec"], ws)

    def _read_preflight_metrics(self, ws: Path) -> dict:
        return {
            "train": read_json(ws / "train_metrics.json"),
            "baseline": read_json(ws / "baseline_metrics.json"),
            "adapter": read_json(ws / "finetuned_metrics.json"),
        }

    def _selection_problems(self, selection: dict, spec: ModelSpec) -> list[str]:
        return selection_problems(selection, spec)

    # -- acceptance and cleanup ---------------------------------------------- #

    def _acceptance(self, context: dict) -> None:
        spec = context["spec"]
        workspace = context["workspace"]
        baseline = read_json(workspace / "baseline_metrics.json")
        finetuned = read_json(workspace / "finetuned_metrics.json")
        train = read_json(workspace / "train_metrics.json")
        comparison = read_json(workspace / "comparison.json")
        peaks = peak_maxima(train, baseline, finetuned)
        preflight_path = context["model_dir"] / "preflight" / "preflight.json"
        preflight = read_json(preflight_path) if preflight_path.is_file() else {}
        repeat_approved = preflight.get("decision") == "go" and preflight.get("repeat") is not None
        failures = acceptance_failures(
            spec=spec,
            baseline=baseline,
            finetuned=finetuned,
            train=train,
            comparison=comparison,
            baseline_prediction_rows=count_csv_data_rows(workspace / "baseline_predictions.csv"),
            finetuned_prediction_rows=count_csv_data_rows(workspace / "finetuned_predictions.csv"),
            workspace=workspace,
            vram_peak_reserved_gib=peaks["reserved_gib"],
            vram_total_gib=self.gpu["total_gib"] if self.gpu else None,
            vram_repeat_approved=repeat_approved,
            vram_peak_problems=peaks["problems"],
        )
        record = {
            "model": spec.slug,
            "status": "ok" if not failures else "failed",
            "failures": failures,
            "peaks": peaks,
            "checked_at": utc_now(),
        }
        write_json_atomic(context["model_dir"] / "acceptance.json", record)
        if failures:
            raise PhaseFailure("acceptance", "; ".join(failures), retryable=False)

    def _cleanup(self, context: dict) -> None:
        spec = context["spec"]
        before = probe_gpu_compute_processes()
        if before:
            raise PhaseFailure(
                "cleanup", f"GPU busy before cleanup: {before}", retryable=False
            )
        result = delete_targets(
            cleanup_plan(self.run_dir, context["model_dir"], spec.slug, self.short_tmp_root),
            allowed_roots=(self.run_dir, self.short_tmp_root),
        )
        after = probe_gpu_compute_processes()
        record = {
            "status": "ok" if not after else "gpu-busy",
            "model": spec.slug,
            "gpu_before": before,
            "gpu_after": after,
            "deleted": result,
            "completed_at": utc_now(),
        }
        write_json_atomic(context["model_dir"] / "cleanup.json", record)
        context["run_info"]["cleanup"] = record
        self._save_run_info(context)
        if after:
            raise PhaseFailure(
                "cleanup", f"GPU compute processes still active after cleanup: {after}",
                retryable=False,
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_qlora_large.py",
        description=(
            "Sequential runner for the fixed four-model QLoRA-large benchmark. "
            "Writes only under benchmarks/qlora-large/<UTC-run-id>/."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="validate everything, run nothing")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None, help="resume a partial run directory")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="with --resume: allow one retry of a failed transiently-retryable phase",
    )
    parser.add_argument("--select-rows", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--slug", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--meta", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.select_rows:
        for name in ("slug", "config", "data_dir", "out", "meta"):
            if not getattr(args, name):
                parser.error(f"--select-rows requires --{name.replace('_', '-')}")
        try:
            return cmd_select_rows(args)
        except RunnerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")

    runner = Runner(
        root=RUN_ROOT,
        dry_run=args.dry_run,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    try:
        return runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted; manifest left partial", file=sys.stderr)
        return 130
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
