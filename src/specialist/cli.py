"""Command line interface.

Commands:
  specialist gather   <config> [--run-dir DIR]
  specialist prepare  <config> [--run-dir DIR]
  specialist baseline <config> [--run-dir DIR]
  specialist train    <config> [--run-dir DIR]
  specialist evaluate <config> --checkpoint <path-or-base> [--run-dir DIR]
  specialist run      <config> [--run-dir DIR]

Without --run-dir, phases use `paths.data_dir` as workspace. `run` creates a
unique UTC run directory under `paths.runs_dir` and executes
gather -> prepare -> baseline -> train -> evaluate(adapter) -> comparison as
subprocess CLI phases (fresh process per model load: Unsloth/transformers import
order stays clean and VRAM is released between phases).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _project_root(config_path: str | Path) -> Path:
    """Config dir, or its parent when the config lives in a `configs/` dir."""
    config_dir = Path(config_path).resolve().parent
    return config_dir.parent if config_dir.name == "configs" else config_dir


def resolve_path(value: str | Path, base: Path) -> Path:
    """Expand ~ and environment variables, then resolve against `base` if relative."""
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root = _project_root(path)
    paths = config.setdefault("paths", {})
    paths.setdefault("data_dir", "data")
    paths.setdefault("runs_dir", "runs")
    paths["data_dir"] = str(resolve_path(paths["data_dir"], root))
    paths["runs_dir"] = str(resolve_path(paths["runs_dir"], root))

    for spec in config.get("sources", {}).values():
        if spec.get("file"):
            spec["file"] = str(resolve_path(spec["file"], root))

    return config


def phase_workspace(config: dict, run_dir: str | None) -> Path:
    """Workspace for one phase: explicit run dir, else the configured data dir."""
    if run_dir:
        workspace = Path(run_dir).expanduser().resolve()
    else:
        workspace = Path(config["paths"]["data_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def create_run_dir(runs_dir: str | Path) -> Path:
    """Unique UTC run dir: runs/YYYYMMDDTHHMMSSZ (suffix on collision)."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = runs_dir / stamp
    counter = 1
    while candidate.exists():
        candidate = runs_dir / f"{stamp}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def _phase_env() -> dict:
    """Make the package importable for `python -m specialist.cli` subprocesses."""
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    return env


def run_phase(
    phase: str,
    config_path: str | Path,
    run_dir: str | Path,
    extra: list[str] | None = None,
) -> None:
    cmd = [
        sys.executable, "-m", "specialist.cli", phase,
        str(config_path), "--run-dir", str(run_dir),
    ]
    if extra:
        cmd += extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(run_dir), env=_phase_env())


def _require_base_field_match(baseline: dict, finetuned: dict, field: str, label: str) -> None:
    """Fail when baseline/fine-tuned runs disagree on the base model identity."""
    left, right = baseline.get(field), finetuned.get(field)
    if left != right:
        raise RuntimeError(
            f"baseline and fine-tuned runs disagree on {label}: {left!r} != {right!r}"
        )


def _quantization_view(metrics: dict) -> dict:
    """Quantization signature for cross-run matching; absent key = legacy BF16.

    Offload is compared as counts, not module names: adapter wrapping changes
    module-key prefixes, so an exact `hf_device_map` comparison would reject
    identical placements.
    """
    quant = metrics.get("quantization") or {}
    offload = quant.get("offload") or {}
    return {
        "requested_load_in_4bit": bool(quant.get("requested_load_in_4bit", False)),
        "effective_load_in_4bit": bool(quant.get("effective_load_in_4bit", False)),
        "quant_type": quant.get("quant_type"),
        "compute_dtype": quant.get("compute_dtype"),
        "double_quant": quant.get("double_quant"),
        # Absent (legacy) and empty (new BF16) compare equal.
        "quantized_parameter_devices": quant.get("quantized_parameter_devices") or [],
        "offload_counts": {
            "cpu": len(offload.get("cpu") or []),
            "disk": len(offload.get("disk") or []),
            "meta": len(offload.get("meta") or []),
        },
    }


def build_comparison(run_dir: str | Path, config: dict) -> dict:
    run_dir = Path(run_dir)
    baseline = json.loads((run_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    finetuned = json.loads((run_dir / "finetuned_metrics.json").read_text(encoding="utf-8"))

    if baseline["test_sha256"] != finetuned["test_sha256"]:
        raise RuntimeError(
            "test split hash mismatch between baseline and fine-tuned runs: "
            f"{baseline['test_sha256']} != {finetuned['test_sha256']}"
        )
    if baseline["split_path"] != finetuned["split_path"]:
        raise RuntimeError(
            "baseline and fine-tuned runs used different split files: "
            f"{baseline['split_path']} != {finetuned['split_path']}"
        )

    # Base identity/revision and effective quantization must match; `.get`
    # keeps artifacts from before these fields existed (legacy BF16 runs) valid.
    for field, label in (
        ("base_model", "base model"),
        ("model_kind", "model kind"),
        ("requested_revision", "requested base revision"),
        ("resolved_revision", "resolved base revision"),
        ("resolved_revision_source", "resolved revision source"),
    ):
        _require_base_field_match(baseline, finetuned, field, label)

    baseline_quant = _quantization_view(baseline)
    finetuned_quant = _quantization_view(finetuned)
    if baseline_quant != finetuned_quant:
        raise RuntimeError(
            "baseline and fine-tuned runs used different quantization settings: "
            f"{baseline_quant} != {finetuned_quant}"
        )

    def summarize(m: dict) -> dict:
        return {
            "checkpoint": m["checkpoint"],
            "mode": m["mode"],
            "adapter": m["adapter"],
            "quantization": _quantization_view(m),
            "strict_accuracy": m["strict_accuracy"],
            "semantic_accuracy": m["semantic_accuracy"],
            "valid_output_rate": m["valid_output_rate"],
            "per_class_recall": m["per_class_recall"],
            "per_class_support": m["per_class_support"],
            "confusion": m["confusion"],
            "predicted_counts": m["predicted_counts"],
            "mean_latency_ms": m["mean_latency_ms"],
            "median_latency_ms": m["median_latency_ms"],
            "issues_per_second": m["issues_per_second"],
            "output_tokens_per_second": m["output_tokens_per_second"],
            "total_generation_time_s": m["total_generation_time_s"],
            "peak_allocated_gb": m["peak_allocated_gb"],
            "peak_reserved_gb": m["peak_reserved_gb"],
            "mean_input_tokens": m["mean_input_tokens"],
            "median_input_tokens": m["median_input_tokens"],
            "total_generated_tokens": m["total_generated_tokens"],
        }

    baseline_summary = summarize(baseline)
    finetuned_summary = summarize(finetuned)

    return {
        "run_dir": str(run_dir),
        "config": config.get("name"),
        "base_identity": {
            "base_model": baseline.get("base_model"),
            "model_kind": baseline.get("model_kind"),
            "requested_revision": baseline.get("requested_revision"),
            "resolved_revision": baseline.get("resolved_revision"),
            "resolved_revision_source": baseline.get("resolved_revision_source"),
            "quantization": baseline_quant,
            "match": True,
        },
        "split": baseline["split"],
        "split_path": baseline["split_path"],
        "test_sha256": {
            "baseline": baseline["test_sha256"],
            "finetuned": finetuned["test_sha256"],
            "match": True,
        },
        "labels": baseline["labels"],
        "baseline": baseline_summary,
        "finetuned": finetuned_summary,
        "accuracy": {
            "baseline": {
                "strict": baseline["strict_accuracy"],
                "semantic": baseline["semantic_accuracy"],
            },
            "finetuned": {
                "strict": finetuned["strict_accuracy"],
                "semantic": finetuned["semantic_accuracy"],
            },
            "delta_pp": {
                "strict": (finetuned["strict_accuracy"] - baseline["strict_accuracy"]) * 100,
                "semantic": (finetuned["semantic_accuracy"] - baseline["semantic_accuracy"]) * 100,
            },
        },
    }


def cmd_gather(config: dict, run_dir: str | None) -> None:
    from .gather import gather

    gather(config, phase_workspace(config, run_dir))


def cmd_prepare(config: dict, run_dir: str | None) -> None:
    from .prepare import prepare

    prepare(config, phase_workspace(config, run_dir))


def cmd_baseline(config: dict, run_dir: str | None) -> None:
    from .evaluate import evaluate

    evaluate(config, phase_workspace(config, run_dir), "base")


def cmd_train(config: dict, run_dir: str | None) -> None:
    from .train import train

    train(config, phase_workspace(config, run_dir))


def cmd_evaluate(config: dict, run_dir: str | None, checkpoint: str) -> None:
    from .evaluate import evaluate

    evaluate(config, phase_workspace(config, run_dir), checkpoint)


def cmd_run(config_path: str | Path, run_dir: str | None, config: dict) -> None:
    if run_dir:
        run_dir_path = Path(run_dir).expanduser().resolve()
        run_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        run_dir_path = create_run_dir(config["paths"]["runs_dir"])

    snapshot = run_dir_path / "config.yaml"
    snapshot.write_text(
        f"# Snapshot of {Path(config_path).resolve()} taken "
        f"{datetime.now(timezone.utc).isoformat()}\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir_path}")

    for phase in ("gather", "prepare", "baseline", "train"):
        run_phase(phase, snapshot, run_dir_path)

    run_phase(
        "evaluate",
        snapshot,
        run_dir_path,
        extra=["--checkpoint", str(run_dir_path / config["training"]["adapter_subdir"])],
    )

    comparison = build_comparison(run_dir_path, config)
    (run_dir_path / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Run directory   : {run_dir_path}")
    print(
        "Strict accuracy : "
        f"{comparison['baseline']['strict_accuracy']:.4f} -> "
        f"{comparison['finetuned']['strict_accuracy']:.4f} "
        f"({comparison['accuracy']['delta_pp']['strict']:+.2f} pp)"
    )
    print(
        "Semantic accuracy: "
        f"{comparison['baseline']['semantic_accuracy']:.4f} -> "
        f"{comparison['finetuned']['semantic_accuracy']:.4f} "
        f"({comparison['accuracy']['delta_pp']['semantic']:+.2f} pp)"
    )
    print("Saved: comparison.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specialist",
        description="Config-driven SLM fine-tuning harness for GitHub issue triage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_dir_help = "workspace directory (default: paths.data_dir from the config)"

    def add_phase(name: str, help_text: str) -> None:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("config", help="path to the YAML config")
        sub.add_argument("--run-dir", default=None, help=run_dir_help)

    add_phase("gather", "copy/parse configured raw sources or fetch them with `gh issue list`")
    add_phase("prepare", "clean raw issues and write temporal train/val/test splits")
    add_phase("baseline", "evaluate the base model on the test split")
    add_phase("train", "LoRA SFT with Unsloth; writes adapter/, trainer/, train_metrics.json")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a checkpoint on the test split (shared baseline/fine-tuned evaluator)",
    )
    evaluate.add_argument("config", help="path to the YAML config")
    evaluate.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "'base' for the configured base model; a directory containing "
            "adapter_config.json is loaded as a LoRA adapter (PeftModel); any "
            "other HF model id/path is evaluated as a base model"
        ),
    )
    evaluate.add_argument("--run-dir", default=None, help=run_dir_help)

    add_phase("run", "create a UTC run dir and execute every phase, then compare")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "gather":
        cmd_gather(config, args.run_dir)
    elif args.command == "prepare":
        cmd_prepare(config, args.run_dir)
    elif args.command == "baseline":
        cmd_baseline(config, args.run_dir)
    elif args.command == "train":
        cmd_train(config, args.run_dir)
    elif args.command == "evaluate":
        cmd_evaluate(config, args.run_dir, args.checkpoint)
    elif args.command == "run":
        cmd_run(args.config, args.run_dir, config)
    else:  # pragma: no cover - argparse enforces choices
        raise SystemExit(f"unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
