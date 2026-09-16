"""Cheap tests for the QLoRA-large runner: no network, no GPU, no model loads.

Mocks cover every subprocess/subprocess-like effect (phase runners, selection,
git/GPU/disk probes) and all runs use temporary project roots, temporary frozen
data and temporary benchmark roots.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_qlora_large as runner_mod  # noqa: E402
from specialist.cli import load_config  # noqa: E402

GOOD_LOSSES = (0.51, 0.42, 0.33)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def good_quant() -> dict:
    return {
        "requested_load_in_4bit": True,
        "effective_load_in_4bit": True,
        "quant_type": ["nf4"],
        "compute_dtype": ["torch.bfloat16"],
        "double_quant": True,
        "quantized_module_count": 10,
        "quantized_parameter_count": 10,
        "packed_parameter_count": 10,
        "non_cuda_parameter_count": 0,
        "bitsandbytes_8bit_module_count": 0,
        "quantized_parameter_devices": ["cuda:0"],
        "offload": {"cpu": [], "disk": [], "meta": []},
    }


def make_frozen_data(tmp: Path, counts=None) -> tuple[Path, dict]:
    counts = counts or {"train": 5, "val": 3, "test": 3}
    data_dir = tmp / "frozen-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, count in counts.items():
        rows = [
            {
                "issue_number": index,
                "input": f"{name} issue {index}",
                "label": "bug" if index % 2 else "feature-request",
            }
            for index in range(count)
        ]
        path = data_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        splits[name] = {"sha256": runner_mod.sha256_file(path), "rows": count}
    (data_dir / "bugs.json").write_text("[]\n", encoding="utf-8")
    (data_dir / "features.json").write_text("[]\n", encoding="utf-8")
    return data_dir, splits


def make_config(spec: runner_mod.ModelSpec, data_dir: Path, **overrides) -> dict:
    kind = overrides.get("kind", spec.kind)
    sources = {
        label: {
            "github_label": label,
            "repo": "microsoft/vscode",
            "file": str(data_dir / filename),
        }
        for label, filename in runner_mod.SOURCE_FILES.items()
    }
    if overrides.get("sources") is not None:
        sources = overrides["sources"]
    return {
        "name": spec.slug,
        "paths": {"data_dir": "../../data", "runs_dir": "../../runs"},
        "sources": sources,
        "model": {
            "base_model": overrides.get("repo", spec.repo),
            "revision": overrides.get("revision", spec.revision),
            "kind": kind,
            "load_in_4bit": overrides.get("load_in_4bit", True),
            "max_seq_length": overrides.get("max_seq_length", 2048),
            "use_exact_model_name": overrides.get("use_exact_model_name", True),
            "trust_remote_code": overrides.get("trust_remote_code", False),
            "system_prompt": overrides.get("system_prompt", runner_mod.SYSTEM_PROMPT),
        },
        "lora": {
            "r": overrides.get("r", 16),
            "alpha": overrides.get("alpha", 32),
            "dropout": overrides.get("dropout", 0),
            "target_modules": overrides.get(
                "target_modules",
                spec.lora_targets if isinstance(spec.lora_targets, str)
                else list(spec.lora_targets),
            ),
            "bias": overrides.get("bias", "none"),
            "gradient_checkpointing": overrides.get("gradient_checkpointing", "unsloth"),
            "random_state": overrides.get("random_state", 42),
            "finetune_vision_layers": overrides.get("finetune_vision_layers", False),
            "finetune_language_layers": overrides.get("finetune_language_layers", True),
            "finetune_attention_modules": overrides.get("finetune_attention_modules", True),
            "finetune_mlp_modules": overrides.get("finetune_mlp_modules", True),
        },
        "training": {
            "output_subdir": overrides.get("output_subdir", "trainer"),
            "adapter_subdir": overrides.get("adapter_subdir", "adapter"),
            "per_device_train_batch_size": overrides.get(
                "batch", spec.per_device_train_batch_size
            ),
            "per_device_eval_batch_size": overrides.get("eval_batch", 1),
            "gradient_accumulation_steps": overrides.get(
                "ga", spec.gradient_accumulation_steps
            ),
            "num_train_epochs": overrides.get("epochs", 3),
            "logging_steps": overrides.get("logging_steps", 10),
            "eval_strategy": overrides.get("eval_strategy", "epoch"),
            "save_strategy": overrides.get("save_strategy", "epoch"),
            "save_total_limit": overrides.get("save_total_limit", 2),
            "load_best_model_at_end": overrides.get("best", True),
            "metric_for_best_model": overrides.get("metric", "eval_loss"),
            "greater_is_better": overrides.get("greater", False),
            "bf16": overrides.get("bf16", True),
            "fp16": overrides.get("fp16", False),
            "optim": overrides.get("optim", "adamw_8bit"),
            "learning_rate": overrides.get("learning_rate", 2.0e-4),
            "warmup_ratio": overrides.get("warmup_ratio", 0.05),
            "weight_decay": overrides.get("weight_decay", 0.01),
            "lr_scheduler_type": overrides.get("lr_scheduler_type", "cosine"),
            "seed": overrides.get("seed", 42),
            "report_to": overrides.get("report_to", "none"),
            "vision_collator": overrides.get("vision_collator", spec.vision_collator),
        },
        "evaluation": {
            "split": overrides.get("split", "test"),
            "max_user_tokens": overrides.get("max_user_tokens", 1800),
            "max_new_tokens": overrides.get("max_new_tokens", 12),
            "do_sample": overrides.get("do_sample", False),
            "pad_eos": overrides.get("pad_eos", True),
            "chat_template_kwargs": overrides.get(
                "chat_template_kwargs", dict(spec.chat_template_kwargs)
            ),
        },
        "benchmark": {
            "display_name": overrides.get("display_name", spec.display_name),
            "checkpoint": overrides.get("bench_checkpoint", spec.repo),
            "revision": overrides.get("bench_revision", spec.revision),
            "official_checkpoint": overrides.get("bench_official", spec.repo),
            "pinned_revision": overrides.get("bench_pinned", spec.revision),
            "method": overrides.get("method", "QLoRA NF4"),
        },
    }


def write_predictions(path: Path, rows: int, mode: str = "baseline") -> None:
    fields = ["issue_number", "target", "prediction", "strict_prediction", "raw_output"]
    if mode == "finetuned":
        fields += ["input_tokens", "output_tokens", "latency_ms"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({field: index for field in fields})


def write_safetensors(path: Path) -> None:
    """Minimal valid safetensors file (header JSON + one f32 tensor)."""
    header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + b"\x00\x00\x80?")


def create_checkpoint(checkpoint: Path, step: int) -> None:
    import torch

    checkpoint.mkdir(parents=True, exist_ok=True)
    runner_mod.write_json_atomic(checkpoint / "trainer_state.json", {"global_step": step})
    runner_mod.write_json_atomic(checkpoint / "adapter_config.json", {"r": 16})
    write_safetensors(checkpoint / "adapter_model.safetensors")
    torch.save({"state": {}, "param_groups": []}, checkpoint / "optimizer.pt")
    torch.save({"lr": 1e-4}, checkpoint / "scheduler.pt")
    torch.save(torch.get_rng_state(), checkpoint / "rng_state.pth")


def create_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    runner_mod.write_json_atomic(adapter_dir / "adapter_config.json", {"r": 16})
    write_safetensors(adapter_dir / "adapter_model.safetensors")


def trainable_summary(vision_like: bool = False) -> dict:
    names = ["base_model.model.q_proj.lora_A.weight"]
    vision_names = []
    if vision_like:
        vision_names = ["base_model.model.vision_tower.encoder.lora_A.weight"]
        names += vision_names
    return {
        "status": "available",
        "count": len(names),
        "numel": 1024 * len(names),
        "names": names,
        "names_truncated": False,
        "vision_like_count": len(vision_names),
        "vision_like_numel": 512 * len(vision_names),
        "vision_like_names": vision_names,
    }


def train_metrics(
    spec: runner_mod.ModelSpec,
    frozen: dict,
    workspace: Path,
    *,
    tiny: bool = False,
    quant=None,
    revision: str | None = None,
    kind: str | None = None,
    peaks=(7.0, 7.5),
    trainable: dict | None = None,
    losses=GOOD_LOSSES,
) -> dict:
    steps = (1, 2, 3) if tiny else (200, 400, 600)
    best_index = losses.index(min(losses))
    checkpoint = workspace / "trainer" / f"checkpoint-{steps[best_index]}"
    create_checkpoint(checkpoint, steps[best_index])
    create_adapter(workspace / "adapter")
    return {
        "base_model": spec.repo,
        "requested_revision": revision or spec.revision,
        "resolved_revision": revision or spec.revision,
        "resolved_revision_source": "model_config",
        "model_kind": kind or spec.kind,
        "quantization": quant or good_quant(),
        "sha256": {name: entry["sha256"] for name, entry in frozen.items()},
        "train_examples": 8 if tiny else frozen["train"]["rows"],
        "val_examples": 2 if tiny else frozen["val"]["rows"],
        "log_history": [
            {"step": step, "epoch": float(index + 1), "eval_loss": loss}
            for index, (step, loss) in enumerate(zip(steps, losses))
        ],
        "best_model_checkpoint": str(checkpoint),
        "best_metric": min(losses),
        "best_epoch": float(best_index + 1),
        "final_epoch": 3.0,
        "final_eval_epoch": 3.0,
        "final_eval_loss": losses[-1],
        "precision": {"bf16": True, "fp16": False, "load_in_4bit": True},
        "lora": (
            {"finetune_vision_layers": False} if (kind or spec.kind) == "vision" else {}
        ),
        "trainable_parameters": trainable or trainable_summary(),
        "setup_peak_allocated_gib": peaks[0] - 0.5,
        "setup_peak_reserved_gib": peaks[0],
        "train_peak_allocated_gib": peaks[1] - 0.5,
        "train_peak_reserved_gib": peaks[1],
    }


def eval_metrics(
    spec: runner_mod.ModelSpec,
    frozen: dict,
    *,
    mode: str,
    examples: int,
    adapter: Path | None = None,
    quant=None,
    revision: str | None = None,
    kind: str | None = None,
    peaks=(6.5, 7.0),
) -> dict:
    metrics = {
        "base_model": spec.repo,
        "requested_revision": revision or spec.revision,
        "resolved_revision": revision or spec.revision,
        "resolved_revision_source": "model_config",
        "model_kind": kind or spec.kind,
        "quantization": quant or good_quant(),
        "test_sha256": frozen["test"]["sha256"],
        "examples": examples,
        "mode": mode,
        "checkpoint": "base" if mode == "base" else str(adapter),
        "adapter": None if mode == "base" else str(adapter),
        "load_peak_allocated_gb": peaks[0] - 0.5,
        "load_peak_reserved_gb": peaks[0],
        "peak_allocated_gb": peaks[1] - 0.5,
        "peak_reserved_gb": peaks[1],
    }
    return metrics


class FakeTrack:
    """Temporary project root with configs, frozen data and a track root."""

    def __init__(self, tmp: Path, counts=None):
        self.project = tmp / "repo"
        (self.project / "configs" / "qlora-large").mkdir(parents=True)
        self.benchmarks = self.project / "benchmarks"
        self.root = self.benchmarks / runner_mod.TRACK
        self.data_dir, self.splits = make_frozen_data(tmp, counts)
        self.short_tmp = tmp / "short_tmp"
        for spec in runner_mod.EXPECTED_MODELS:
            (self.project / spec.config).write_text(
                yaml.safe_dump(make_config(spec, self.data_dir), sort_keys=False),
                encoding="utf-8",
            )
        self.protected = (
            self.benchmarks / "20260915T112519Z",
            self.benchmarks / "20260915T100825Z",
            self.benchmarks / "LATEST",
        )
        self.calls: list[tuple[str, str]] = []
        self.configs_seen: list[tuple[str, str, str]] = []
        self.pythons_seen: set[str] = set()
        self.error_plan: dict[tuple[str, str], list] = {}
        self.peak_plan: dict[tuple[str, str], tuple] = {}
        self.trainable_plan: dict[tuple[str, str], dict] = {}
        self.versions = dict(runner_mod.PINNED_VERSIONS, python="3.11.9")
        self.gpu = {"device_index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "total_gib": 15.46, "visible_devices": 1}

    def plan_error(self, slug: str, phase: str, *, returns=1, log="connection reset"):
        self.error_plan.setdefault((slug, phase), []).append((returns, log))

    def _maybe_fail(self, slug: str, phase: str, log_path: Path) -> bool:
        plan = self.error_plan.get((slug, phase)) or []
        if not plan:
            return False
        returns, text = plan.pop(0)
        log_path.write_text(text + "\n", encoding="utf-8")
        return True

    def spec_from_workspace(self, workspace: Path) -> runner_mod.ModelSpec:
        for slug in (spec.slug for spec in runner_mod.EXPECTED_MODELS):
            if slug in Path(workspace).parts:
                return runner_mod.spec_by_slug(slug)
        raise AssertionError(f"no slug in {workspace}")

    def phase(self, *, python, phase, config_path, workspace, extra, env, log_path):
        workspace = Path(workspace)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        spec = self.spec_from_workspace(workspace)
        stem = log_path.name.rsplit("-a", 1)[0]
        self.calls.append((spec.slug, stem))
        self.configs_seen.append((spec.slug, stem, str(config_path)))
        self.pythons_seen.add(python)
        if self._maybe_fail(spec.slug, stem, log_path):
            return 1
        log_path.write_text("fake phase ok\n", encoding="utf-8")
        tiny = stem.startswith("preflight-")
        rows = 1 if tiny else self.splits["test"]["rows"]
        peaks = self.peak_plan.get((spec.slug, stem))
        trainable = self.trainable_plan.get((spec.slug, stem))
        if phase == "train":
            runner_mod.write_json_atomic(
                workspace / "train_metrics.json",
                train_metrics(spec, self.splits, workspace, tiny=tiny,
                              peaks=peaks or (7.0, 7.5), trainable=trainable),
            )
        elif phase == "baseline":
            runner_mod.write_json_atomic(
                workspace / "baseline_metrics.json",
                eval_metrics(spec, self.splits, mode="base", examples=rows,
                             peaks=peaks or (6.5, 7.0)),
            )
            write_predictions(workspace / "baseline_predictions.csv", rows)
        elif phase == "evaluate":
            adapter = workspace / "adapter"
            runner_mod.write_json_atomic(
                workspace / "finetuned_metrics.json",
                eval_metrics(spec, self.splits, mode="adapter", examples=rows,
                             adapter=adapter, peaks=peaks or (6.5, 7.0)),
            )
            write_predictions(workspace / "finetuned_predictions.csv", rows, mode="finetuned")
        return 0

    def selection(self, *, python, script_path, slug, config_path, data_dir, out_dir, meta_path, env, log_path):
        spec = runner_mod.spec_by_slug(slug)
        out_dir, meta_path, log_path = Path(out_dir), Path(meta_path), Path(log_path)
        out_dir.mkdir(parents=True, exist_ok=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake selection\n", encoding="utf-8")
        selected = {}
        for split, need in runner_mod.PREFLIGHT_NEEDS.items():
            rows = [{"issue_number": index, "input": "x", "label": "bug"} for index in range(need)]
            write_jsonl(out_dir / f"{split}.jsonl", rows)
            selected[split] = [
                {"index": index, "issue_number": index, "length": 2600 - index}
                for index in range(need)
            ]
        runner_mod.write_json_atomic(
            meta_path,
            {
                "model": slug,
                "repo": spec.repo,
                "requested_revision": spec.revision,
                "resolved_revision": spec.revision,
                "kind": spec.kind,
                "threshold": runner_mod.MIN_SUPERVISED_TOKENS,
                "needs": runner_mod.PREFLIGHT_NEEDS,
                "scanned": {split: 20 for split in runner_mod.PREFLIGHT_NEEDS},
                "selected": selected,
            },
        )
        return 0

    def comparison(self, workspace, config):
        frozen_test = self.splits["test"]["sha256"]
        return {
            "test_sha256": {
                "baseline": frozen_test,
                "finetuned": frozen_test,
                "match": True,
            },
            "base_identity": {"match": True},
        }

    def patches(self):
        return [
            mock.patch.object(runner_mod, "FROZEN_SPLITS", self.splits),
            mock.patch.object(runner_mod, "PROTECTED_PATHS", self.protected),
            mock.patch.object(runner_mod, "probe_git_head", return_value="deadbeef"),
            mock.patch.object(runner_mod, "probe_git_status", return_value=[]),
            mock.patch.object(runner_mod, "probe_versions", return_value=self.versions),
            mock.patch.object(runner_mod, "probe_disk_free_gib", return_value=120.0),
            mock.patch.object(runner_mod, "probe_gpu_identity", side_effect=lambda: dict(self.gpu)),
            mock.patch.object(runner_mod, "probe_gpu_compute_processes", return_value=[]),
            mock.patch.object(runner_mod, "run_phase_subprocess", side_effect=self.phase),
            mock.patch.object(runner_mod, "run_selection_subprocess", side_effect=self.selection),
            mock.patch.object(runner_mod, "build_comparison", side_effect=self.comparison),
        ]

    def runner(self, **kwargs):
        kwargs.setdefault("project_root", self.project)
        kwargs.setdefault("root", self.root)
        kwargs.setdefault("data_dir", self.data_dir)
        kwargs.setdefault("short_tmp_root", self.short_tmp)
        kwargs.setdefault("out", lambda *args, **kwargs: None)
        return runner_mod.Runner(**kwargs)

    def run_dir(self) -> Path:
        return next(path for path in self.root.iterdir() if path.is_dir())


class PatchedTrackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.track = FakeTrack(Path(self.tmp.name))
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        for patcher in self.track.patches():
            self._stack.enter_context(patcher)


class TrackConstantsTest(PatchedTrackTest):
    def test_fixed_order_and_expected_values(self):
        self.assertEqual(
            [spec.slug for spec in runner_mod.EXPECTED_MODELS],
            [
                "01-qwen3-8b",
                "02-ministral-3-8b-instruct",
                "03-qwen3.5-9b",
                "04-ministral-3-14b-instruct",
            ],
        )
        self.assertEqual(
            [spec.kind for spec in runner_mod.EXPECTED_MODELS],
            ["language", "vision", "vision", "vision"],
        )
        self.assertEqual(
            [(spec.per_device_train_batch_size, spec.gradient_accumulation_steps)
             for spec in runner_mod.EXPECTED_MODELS],
            [(2, 4), (2, 4), (2, 4), (1, 8)],
        )
        self.assertEqual(
            [spec.vision_collator for spec in runner_mod.EXPECTED_MODELS],
            [False, True, False, True],
        )
        for spec in runner_mod.EXPECTED_MODELS:
            self.assertEqual(
                spec.per_device_train_batch_size * spec.gradient_accumulation_steps,
                runner_mod.EFFECTIVE_BATCH,
            )
            self.assertTrue((ROOT / spec.config).is_file(), spec.config)
        self.assertEqual(runner_mod.PINNED_VERSIONS["tqdm"], "4.70.1")

    def test_real_configs_validate_against_fixed_track(self):
        for spec in runner_mod.EXPECTED_MODELS:
            config = load_config(ROOT / spec.config)
            self.assertEqual(
                runner_mod.validate_config(config, spec, data_dir=runner_mod.FROZEN_DATA_DIR),
                [],
                spec.slug,
            )


class ConfigValidationTest(unittest.TestCase):
    spec = runner_mod.EXPECTED_MODELS[0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)

    def config(self, **overrides) -> dict:
        return make_config(self.spec, self.data_dir, **overrides)

    def test_expected_config_passes(self):
        self.assertEqual(
            runner_mod.validate_config(self.config(), self.spec, data_dir=self.data_dir), []
        )

    def assert_refused(self, key_fragment: str, **overrides):
        problems = runner_mod.validate_config(
            self.config(**overrides), self.spec, data_dir=self.data_dir
        )
        self.assertTrue(
            any(key_fragment in problem for problem in problems),
            f"expected {key_fragment!r} in {problems}",
        )

    def test_model_and_batch_fields_refused(self):
        self.assert_refused("base_model", repo="other/model")
        self.assert_refused("revision", revision="0" * 40)
        self.assert_refused("kind", kind="vision")
        self.assert_refused("load_in_4bit", load_in_4bit=False)
        self.assert_refused("max_seq_length", max_seq_length=1024)
        self.assert_refused("use_exact_model_name", use_exact_model_name=False)
        self.assert_refused("trust_remote_code", trust_remote_code=True)
        self.assert_refused("system_prompt", system_prompt="do something else")
        self.assert_refused("per_device_train_batch_size", batch=3)
        self.assert_refused("gradient_accumulation_steps", ga=2)
        self.assert_refused("per_device_train_batch_size", batch=4, ga=4)

    def test_lora_fields_refused(self):
        self.assert_refused("lora.r", r=8)
        self.assert_refused("lora.alpha", alpha=16)
        self.assert_refused("lora.dropout", dropout=0.1)
        self.assert_refused("lora.bias", bias="all")
        self.assert_refused("gradient_checkpointing", gradient_checkpointing=False)
        self.assert_refused("random_state", random_state=7)
        self.assert_refused("target_modules", target_modules=["q_proj"])

    def test_training_fields_refused(self):
        for key, value in (
            ("eval_batch", 2), ("epochs", 2), ("logging_steps", 5),
            ("eval_strategy", "steps"), ("save_strategy", "no"),
            ("save_total_limit", 5), ("best", False), ("metric", "loss"),
            ("greater", True), ("bf16", False), ("fp16", True),
            ("optim", "adamw_torch"), ("learning_rate", 1e-4),
            ("warmup_ratio", 0.1), ("weight_decay", 0.0),
            ("lr_scheduler_type", "linear"), ("seed", 1), ("report_to", "wandb"),
        ):
            self.assert_refused(key, **{key: value})

    def test_vision_flags_and_collator_refused(self):
        vision = runner_mod.EXPECTED_MODELS[1]
        config = make_config(vision, self.data_dir)
        config["lora"]["finetune_vision_layers"] = True
        self.assertTrue(
            any("finetune_vision_layers" in problem
                for problem in runner_mod.validate_config(config, vision, data_dir=self.data_dir))
        )
        config = make_config(vision, self.data_dir)
        config["training"]["vision_collator"] = False
        self.assertTrue(
            any("vision_collator" in problem
                for problem in runner_mod.validate_config(config, vision, data_dir=self.data_dir))
        )

    def test_evaluation_and_benchmark_fields_refused(self):
        self.assert_refused("evaluation.split", split="val")
        self.assert_refused("max_user_tokens", max_user_tokens=1024)
        self.assert_refused("max_new_tokens", max_new_tokens=32)
        self.assert_refused("do_sample", do_sample=True)
        self.assert_refused("pad_eos", pad_eos=False)
        self.assert_refused("chat_template_kwargs", chat_template_kwargs={})
        self.assert_refused("benchmark.checkpoint", bench_checkpoint="other/model")
        self.assert_refused("benchmark.revision", bench_revision="0" * 40)
        self.assert_refused("benchmark.official_checkpoint", bench_official="other/model")
        self.assert_refused("benchmark.pinned_revision", bench_pinned="0" * 40)
        self.assert_refused("benchmark.display_name", display_name="Other")
        self.assert_refused("benchmark.method", method="LoRA BF16")

    def test_source_labels_and_paths_refused(self):
        sources = {
            "bug": {"github_label": "bug", "repo": "microsoft/vscode",
                    "file": str(self.data_dir / "bugs.json")},
        }
        self.assert_refused("sources", sources=sources)
        sources = {
            "bug": {"github_label": "defect", "repo": "microsoft/vscode",
                    "file": str(self.data_dir / "bugs.json")},
            "feature-request": {"github_label": "feature-request", "repo": "microsoft/vscode",
                                "file": str(self.data_dir / "features.json")},
        }
        self.assert_refused("sources.bug.github_label", sources=sources)
        sources = {
            "bug": {"github_label": "bug", "repo": "other/repo",
                    "file": str(self.data_dir / "bugs.json")},
            "feature-request": {"github_label": "feature-request", "repo": "microsoft/vscode",
                                "file": str(self.data_dir / "features.json")},
        }
        self.assert_refused("sources.bug.repo", sources=sources)
        sources = {
            "bug": {"github_label": "bug", "repo": "microsoft/vscode",
                    "file": str(self.data_dir / "wrong.json")},
            "feature-request": {"github_label": "feature-request", "repo": "microsoft/vscode",
                                "file": str(self.data_dir / "features.json")},
        }
        self.assert_refused("sources.bug.file", sources=sources)
        sources = {
            "bug": {"github_label": "bug", "repo": "microsoft/vscode", "file": None},
            "feature-request": {"github_label": "feature-request", "repo": "microsoft/vscode",
                                "file": str(self.data_dir / "features.json")},
        }
        self.assert_refused("sources.bug.file", sources=sources)

    def test_static_validation_refuses_swapped_identity_for_one_model(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        track = FakeTrack(Path(tmp.name))
        swapped = make_config(runner_mod.EXPECTED_MODELS[1], track.data_dir, repo="Qwen/Qwen3-8B")
        (track.project / runner_mod.EXPECTED_MODELS[1].config).write_text(
            yaml.safe_dump(swapped, sort_keys=False), encoding="utf-8"
        )
        with mock.patch.object(runner_mod, "FROZEN_SPLITS", track.splits), \
                mock.patch.object(runner_mod, "PROTECTED_PATHS", track.protected), \
                mock.patch.object(runner_mod, "probe_versions", return_value=track.versions), \
                mock.patch.object(runner_mod, "probe_disk_free_gib", return_value=120.0), \
                mock.patch.object(runner_mod, "probe_gpu_identity", return_value=track.gpu), \
                mock.patch.object(runner_mod, "probe_gpu_compute_processes", return_value=[]), \
                mock.patch.object(runner_mod, "probe_git_head", return_value="deadbeef"), \
                mock.patch.object(runner_mod, "probe_git_status", return_value=[]):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                track.runner(dry_run=True).run()
        self.assertIn("base_model", str(ctx.exception))


class FrozenDataTest(unittest.TestCase):
    def test_matching_hashes_and_counts_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, splits = make_frozen_data(Path(tmp))
            report, problems = runner_mod.frozen_data_report(data_dir, splits)
        self.assertEqual(problems, [])
        self.assertEqual(report["train"]["rows"], 5)
        self.assertEqual(report["val"]["sha256"], splits["val"]["sha256"])

    def test_wrong_hash_count_and_missing_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, splits = make_frozen_data(Path(tmp))
            splits["train"]["sha256"] = "0" * 64
            splits["val"]["rows"] += 1
            splits["test"] = {"sha256": "1" * 64, "rows": 3}
            (data_dir / "test.jsonl").unlink()
            _, problems = runner_mod.frozen_data_report(data_dir, splits)
        joined = " ".join(problems)
        self.assertIn("train sha256 mismatch", joined)
        self.assertIn("val row count mismatch", joined)
        self.assertIn("missing frozen split", joined)

    def test_copy_verifies_bytes_and_never_writes_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, splits = make_frozen_data(Path(tmp))
            source_hash = runner_mod.sha256_file(data_dir / "train.jsonl")
            copied = runner_mod.copy_frozen_into_workspace(data_dir, Path(tmp) / "ws", splits)
            self.assertEqual(copied["train"]["sha256"], source_hash)
            self.assertEqual(runner_mod.sha256_file(data_dir / "train.jsonl"), source_hash)
            self.assertEqual(
                runner_mod.sha256_file(Path(tmp) / "ws" / "train.jsonl"), source_hash
            )


class GitGuardTest(unittest.TestCase):
    def test_only_exact_allowed_untracked_paths_pass(self):
        active = "benchmarks/qlora-large/20260101T000000Z/"
        lines = [f"?? {active}manifest.json", f"?? {active}models/01-qwen3-8b/run_info.json"]
        self.assertEqual(runner_mod.git_tree_problems(lines, allowed_untracked=(active,)), [])
        # exact-prefix only: sibling runs and collapsed directories are rejected
        problems = runner_mod.git_tree_problems(
            [f"?? {active}manifest.json",
             "?? benchmarks/qlora-large/20250101T000000Z/manifest.json",
             "?? benchmarks/qlora-large/LATEST"],
            allowed_untracked=(active,),
        )
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("outside the active run" in problem for problem in problems))
        # a collapsed directory entry is not allowed either
        problems = runner_mod.git_tree_problems(
            ["?? benchmarks/qlora-large/20260101T000000Z/"],
            allowed_untracked=("benchmarks/qlora-large/20260101T000000Z/manifest.json",),
        )
        self.assertTrue(any("outside the active run" in problem for problem in problems))
        # new runs / dry-run allow nothing
        problems = runner_mod.git_tree_problems([f"?? {active}manifest.json"])
        self.assertTrue(problems)
        problems = runner_mod.git_tree_problems([" M src/specialist/train.py"])
        self.assertTrue(any("tracked change" in problem for problem in problems))

    def test_probe_git_status_uses_full_untracked_listing(self):
        result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(runner_mod.subprocess, "run", return_value=result) as run:
            runner_mod.probe_git_status("/tmp")
        command = run.call_args.args[0]
        self.assertIn("--untracked-files=all", command)

class GitInvariantTest(PatchedTrackTest):
    def test_dry_run_allows_no_untracked_and_sibling_runs_block(self):
        with mock.patch.object(
            runner_mod, "probe_git_status",
            return_value=["?? benchmarks/qlora-large/20250101T000000Z/manifest.json"],
        ), mock.patch.object(
            runner_mod, "run_phase_subprocess", side_effect=AssertionError("dry-run phase")
        ):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(dry_run=True).run()
        self.assertIn("outside the active run", str(ctx.exception))
        with mock.patch.object(
            runner_mod, "probe_git_status", return_value=[" M src/x.py"]
        ):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(dry_run=True).run()
        self.assertIn("tracked change", str(ctx.exception))

    def test_resume_allows_only_the_active_run_path(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        active = f"benchmarks/qlora-large/{run_dir.name}/"
        lines = [f"?? {active}manifest.json", f"?? {active}models/01-qwen3-8b/run_info.json"]
        with mock.patch.object(runner_mod, "probe_git_status", return_value=lines):
            self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)

        # sibling prior run left untracked blocks a resume of a partial run
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir2 = next(
            path for path in self.track.root.iterdir()
            if path.is_dir() and path.name != run_dir.name
        )
        sibling = [
            f"?? {active}manifest.json",
            "?? benchmarks/qlora-large/19990101T000000Z/manifest.json",
        ]
        with mock.patch.object(runner_mod, "probe_git_status", return_value=sibling):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(resume=run_dir2, retry_failed=True).run()
        self.assertIn("outside the active run", str(ctx.exception))


class InterpreterIdentityTest(PatchedTrackTest):
    def test_phases_and_manifest_use_resolved_current_interpreter(self):
        self.assertEqual(self.track.runner().run(), 0)
        run_dir = self.track.run_dir()
        expected = str(Path(sys.executable).resolve())
        self.assertEqual(self.track.pythons_seen, {expected})
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["python"], expected)
        self.assertEqual(runner_mod.Runner().python, expected)

    def test_resume_refuses_a_different_recorded_interpreter(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with mock.patch.object(runner_mod, "probe_git_status", return_value=[]):
            with self.assertRaises(runner_mod.RunnerError):
                self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        manifest["python"] = "/usr/bin/other-python"
        runner_mod.write_json_atomic(run_dir / "manifest.json", manifest)
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("python interpreter", str(ctx.exception))

    def test_cli_has_no_python_override(self):
        parser = runner_mod.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--python", "/usr/bin/other"])


class NamespaceAndCleanupTest(PatchedTrackTest):
    def test_namespace_refuses_protected_or_project_root(self):
        problems = runner_mod.validate_namespace(
            self.track.benchmarks, self.track.project
        )
        self.assertTrue(any("protected" in problem for problem in problems))
        problems = runner_mod.validate_namespace(self.track.project, self.track.project)
        self.assertTrue(any("project root" in problem for problem in problems))

    def test_track_latest_write_refuses_root_pointer(self):
        with self.assertRaises(runner_mod.RunnerError):
            runner_mod.write_track_latest(self.track.benchmarks, "20260101T000000Z")
        self.assertFalse((self.track.benchmarks / "LATEST").exists())

    def test_cleanup_plan_deletes_only_known_descendants(self):
        run_dir = self.track.root / "RUN"
        model_dir = run_dir / "models" / "01-qwen3-8b"
        workspace = model_dir / "workspace"
        for relative in ("workspace/adapter", "workspace/trainer", "workspace"):
            (model_dir / relative).mkdir(parents=True, exist_ok=True)
        create_adapter(workspace / "adapter")
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            (workspace / name).write_text("{}\n", encoding="utf-8")
        for name in (
            "baseline_metrics.json", "train_metrics.json", "finetuned_metrics.json",
            "baseline_predictions.csv", "finetuned_predictions.csv", "comparison.json",
            "dataset_stats.json",
        ):
            (workspace / name).write_text("{}", encoding="utf-8")
        for name in ("config.yaml", "run_info.json", "acceptance.json", "cleanup.json"):
            (model_dir / name).write_text("{}", encoding="utf-8")
        (model_dir / "preflight").mkdir(parents=True, exist_ok=True)
        (model_dir / "preflight" / "selection.json").write_text("{}", encoding="utf-8")
        (model_dir / "preflight" / "preflight.json").write_text("{}", encoding="utf-8")
        (model_dir / "logs").mkdir(parents=True, exist_ok=True)
        (model_dir / "logs" / "baseline-a1.log").write_text("log", encoding="utf-8")
        cache = run_dir / ".cache" / "01-qwen3-8b"
        cache.mkdir(parents=True)
        (cache / "blob.bin").write_bytes(b"x" * 16)
        tmp_dir = self.track.short_tmp / run_dir.name / "01-qwen3-8b"
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "temp.bin").write_bytes(b"y" * 8)

        plan = runner_mod.cleanup_plan(run_dir, model_dir, "01-qwen3-8b", self.track.short_tmp)
        plan_names = {path.name for path in plan}
        self.assertIn("adapter", plan_names)
        self.assertIn("trainer", plan_names)
        self.assertIn("01-qwen3-8b", plan_names)
        self.assertNotIn("preflight", plan_names)

        result = runner_mod.delete_targets(
            plan, allowed_roots=(run_dir, self.track.short_tmp)
        )
        self.assertGreaterEqual(result["bytes_deleted"], 16 + 8 + 9)
        self.assertFalse(cache.exists())
        self.assertFalse(tmp_dir.exists())
        self.assertFalse((workspace / "train.jsonl").exists())
        for name in (
            "baseline_metrics.json", "train_metrics.json", "finetuned_metrics.json",
            "baseline_predictions.csv", "finetuned_predictions.csv", "comparison.json",
            "dataset_stats.json",
        ):
            self.assertTrue((workspace / name).is_file(), name)
        self.assertTrue((model_dir / "config.yaml").is_file())
        self.assertTrue((model_dir / "run_info.json").is_file())
        self.assertTrue((model_dir / "acceptance.json").is_file())
        self.assertTrue((model_dir / "cleanup.json").is_file())
        self.assertTrue((model_dir / "preflight" / "selection.json").is_file())
        self.assertTrue((model_dir / "preflight" / "preflight.json").is_file())
        self.assertTrue((model_dir / "logs" / "baseline-a1.log").is_file())

    def test_cleanup_refuses_targets_outside_allowed_roots(self):
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("keep", encoding="utf-8")
        with self.assertRaises(runner_mod.RunnerError):
            runner_mod.delete_targets([outside], allowed_roots=(self.track.root,))
        self.assertTrue(outside.is_file())

    def test_cleanup_refuses_protected_targets(self):
        with self.assertRaises(runner_mod.RunnerError):
            runner_mod.validate_cleanup_targets(
                [self.track.protected[0]], allowed_roots=(self.track.benchmarks,)
            )

    def test_isolated_caches_are_per_model_and_run_scoped_tmp(self):
        run_dir = self.track.root / "20260101T000000Z"
        env = runner_mod.isolated_env(
            run_dir, "01-qwen3-8b", self.track.short_tmp, self.track.project
        )
        self.assertEqual(env["HF_HOME"], str(run_dir / ".cache" / "01-qwen3-8b" / "hf"))
        self.assertEqual(
            env["TMPDIR"],
            str(self.track.short_tmp / "20260101T000000Z" / "01-qwen3-8b"),
        )
        self.assertEqual(
            env["TORCHINDUCTOR_CACHE_DIR"],
            str(run_dir / ".cache" / "01-qwen3-8b" / "inductor"),
        )
        self.assertTrue(Path(env["TMPDIR"]).is_dir())


class VramDecisionTest(unittest.TestCase):
    def test_green_goes(self):
        self.assertEqual(runner_mod.vram_decision(11.9, 15.46)[0], "go")
        self.assertEqual(runner_mod.vram_decision(13.50, 15.46)[0], "go")

    def test_amber_requires_one_repeat(self):
        self.assertEqual(runner_mod.vram_decision(13.8, 15.46)[0], "repeat")
        self.assertEqual(
            runner_mod.vram_decision(13.8, 15.46, repeat_gib=13.9)[0], "go"
        )

    def test_amber_repeat_outcomes(self):
        verdict, reason = runner_mod.vram_decision(13.8, 15.46, repeat_gib=14.2)
        self.assertEqual(verdict, "stop")
        self.assertIn("differs", reason)
        self.assertEqual(runner_mod.vram_decision(13.8, 15.46, repeat_gib=14.3)[0], "stop")

    def test_red_headroom_offload_oom_and_unknown_total_stop(self):
        self.assertEqual(runner_mod.vram_decision(14.5, 15.46)[0], "stop")
        self.assertEqual(runner_mod.vram_decision(13.8, 14.5)[0], "stop")
        self.assertEqual(runner_mod.vram_decision(11.0, 15.46, offload=True)[0], "stop")
        self.assertEqual(runner_mod.vram_decision(11.0, 15.46, oom=True)[0], "stop")
        self.assertEqual(runner_mod.vram_decision(13.8, 15.46, repeat_gib=14.4)[0], "stop")
        self.assertEqual(runner_mod.vram_decision(11.0, None)[0], "stop")

    def test_peak_maxima_requires_valid_values(self):
        peaks = runner_mod.peak_maxima(
            {"setup_peak_reserved_gib": 11.0, "setup_peak_allocated_gib": 10.5,
             "train_peak_reserved_gib": 12.0, "train_peak_allocated_gib": 11.5},
            {"load_peak_reserved_gb": 10.0, "load_peak_allocated_gb": 9.5,
             "peak_reserved_gb": 13.0, "peak_allocated_gb": 12.5},
            {"load_peak_reserved_gb": 10.5, "load_peak_allocated_gb": 10.0,
             "peak_reserved_gb": 12.5, "peak_allocated_gb": 12.0},
        )
        self.assertEqual(peaks["problems"], [])
        self.assertEqual(peaks["reserved_gib"], 13.0)

        broken = runner_mod.peak_maxima(
            {"setup_peak_reserved_gib": 11.0, "setup_peak_allocated_gib": 10.5,
             "train_peak_reserved_gib": 0.0, "train_peak_allocated_gib": 11.5},
            {"load_peak_reserved_gb": 10.0, "load_peak_allocated_gb": 9.5,
             "peak_reserved_gb": 13.0, "peak_allocated_gb": 12.5},
            {"load_peak_reserved_gb": 10.5, "load_peak_allocated_gb": 10.0,
             "peak_reserved_gb": 12.5, "peak_allocated_gb": 12.0},
        )
        self.assertTrue(broken["problems"])
        self.assertIsNone(broken["reserved_gib"])

        missing = runner_mod.peak_maxima({}, {}, {})
        self.assertEqual(len(missing["problems"]), len(runner_mod.PEAK_FIELDS))
        self.assertIsNone(missing["reserved_gib"])

    def test_cuda_peak_helpers_propagate_probe_errors(self):
        import importlib

        for module_name in ("specialist.train", "specialist.evaluate"):
            module = importlib.import_module(module_name)
            broken_torch = mock.MagicMock()
            broken_torch.cuda.is_available.return_value = True
            broken_torch.cuda.max_memory_allocated.side_effect = RuntimeError("probe failed")
            with mock.patch.dict(sys.modules, {"torch": broken_torch}):
                with self.assertRaises(RuntimeError):
                    module._cuda_peak_gib()
            no_cuda = mock.MagicMock()
            no_cuda.cuda.is_available.return_value = False
            with mock.patch.dict(sys.modules, {"torch": no_cuda}):
                self.assertEqual(module._cuda_peak_gib(), (0.0, 0.0))


class FailureClassificationTest(unittest.TestCase):
    def test_transient_allowlist_matches(self):
        for text in (
            "ConnectionResetError: connection reset by peer",
            "requests.exceptions.ReadTimeout: HTTPSConnectionPool read timed out",
            "OSError: AF_UNIX path too long",
            "HTTP Error 503: Service Unavailable",
            "KeyboardInterrupt",
        ):
            retryable, _ = runner_mod.classify_failure(text, 1)
            self.assertTrue(retryable, text)

    def test_unknown_and_hard_failures_are_not_retryable(self):
        for text in (
            "CUDA out of memory",
            "assert_effective_quantization failed: quant_type ['fp4']",
            "loss is nan",
            "RuntimeError: unsupported architecture",
            "KeyError: 'missing_key'",
            "some unknown error without words from the allowlist",
        ):
            retryable, _ = runner_mod.classify_failure(text, 1)
            self.assertFalse(retryable, text)
        self.assertTrue(runner_mod.classify_failure("", 130)[0])
        self.assertTrue(runner_mod.classify_failure("", 143)[0])
        self.assertFalse(runner_mod.classify_failure("", 1)[0])


class PreflightSelectionTest(unittest.TestCase):
    def test_rendered_length_handles_batch_encoding_like_mappings(self):
        from collections import UserDict

        class Encoding(UserDict):
            pass

        class MappingSource:
            def apply_chat_template(self, messages, **kwargs):
                return Encoding({"input_ids": [1] * 2050, "attention_mask": [1] * 2050})

        class AttributeSource:
            def apply_chat_template(self, messages, **kwargs):
                class Out:
                    input_ids = [1] * 7

                return Out()

        class ListSource:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * 9

        row = {"input": "x", "label": "bug"}
        self.assertEqual(
            runner_mod.rendered_supervised_length(MappingSource(), row, "s", {}), 2050
        )
        self.assertEqual(
            runner_mod.rendered_supervised_length(AttributeSource(), row, "s", {}), 7
        )
        self.assertEqual(
            runner_mod.rendered_supervised_length(ListSource(), row, "s", {}), 9
        )

    def test_first_rows_meeting_threshold_are_selected(self):
        rows = [{"issue_number": i, "input": f"i{i}", "label": "bug"} for i in range(20)]
        lengths = {0: 100, 1: 2050, 2: 3000, 3: 2047, 4: 4000, 5: 2100, 6: 2200}

        def length_fn(row, split):
            return lengths[row["issue_number"]]

        result = runner_mod.select_preflight_rows(
            {"train": rows}, length_fn, needs={"train": 4}, threshold=2048
        )
        self.assertEqual([pick["index"] for pick in result["selected"]["train"]], [1, 2, 4, 5])
        self.assertEqual(result["missing"], {})
        self.assertEqual(result["scanned"], {"train": 20})

    def test_missing_rows_reported_when_threshold_not_met(self):
        rows = [{"issue_number": i, "input": "x", "label": "bug"} for i in range(3)]
        result = runner_mod.select_preflight_rows(
            {"train": rows}, lambda row, split: 100, needs={"train": 8}, threshold=2048
        )
        self.assertEqual(result["missing"], {"train": 8})

    def test_selection_metadata_problems_detected(self):
        runner = runner_mod.Runner()
        spec = runner_mod.EXPECTED_MODELS[0]
        selection = {
            "model": spec.slug,
            "repo": spec.repo,
            "requested_revision": spec.revision,
            "resolved_revision": spec.revision,
            "threshold": 2048,
            "selected": {
                "train": [{"length": 2048}] * 8,
                "val": [{"length": 100}] * 2,
                "test": [{"length": 3000}],
            },
        }
        problems = runner._selection_problems(selection, spec)
        self.assertTrue(any("val" in problem for problem in problems))
        self.assertTrue(any("below" in problem for problem in problems))

    def test_revision_from_object_uses_snapshot_paths_and_rejects_unresolved(self):
        sha = "b968826d9c46dd6066d109eabc6255188de91218"

        class SnapshotSource:
            def __init__(self):
                self.init_kwargs = {
                    "vocab_file": f"/cache/models--Qwen--Qwen3-8B/snapshots/{sha}/vocab.json",
                    "name_or_path": "Qwen/Qwen3-8B",
                }

        class CommitHashSource:
            class Config:
                _commit_hash = "a" * 40

            def __init__(self):
                self.config = self.Config()

        class UnresolvedSource:
            def __init__(self):
                self.init_kwargs = {"name_or_path": "Qwen/Qwen3-8B"}

        self.assertEqual(runner_mod.revision_from_object(SnapshotSource()), sha)
        self.assertEqual(runner_mod.revision_from_object(CommitHashSource()), "a" * 40)
        self.assertIsNone(runner_mod.revision_from_object(UnresolvedSource()))

    def test_selection_resolved_revision_must_match(self):
        runner = runner_mod.Runner()
        spec = runner_mod.EXPECTED_MODELS[0]
        selection = {
            "model": spec.slug,
            "repo": spec.repo,
            "requested_revision": spec.revision,
            "resolved_revision": "0" * 40,
            "threshold": 2048,
            "selected": {
                "train": [{"length": 2048}] * 8,
                "val": [{"length": 2048}] * 2,
                "test": [{"length": 2048}],
            },
        }
        problems = runner._selection_problems(selection, spec)
        self.assertTrue(any("resolved revision" in problem for problem in problems))

        del selection["resolved_revision"]
        problems = runner._selection_problems(selection, spec)
        self.assertTrue(any("resolved revision" in problem for problem in problems))


class PreflightProofTest(unittest.TestCase):
    """Unit-level proof checks applied identically to first and repeat runs."""

    spec = runner_mod.EXPECTED_MODELS[0]
    vision_spec = runner_mod.EXPECTED_MODELS[1]
    frozen = {
        "train": {"sha256": "a" * 64, "rows": 1593},
        "val": {"sha256": "b" * 64, "rows": 200},
        "test": {"sha256": "c" * 64, "rows": 200},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name) / "first" / "ws"
        self.ws.mkdir(parents=True)
        create_adapter(self.ws / "adapter")
        runner_mod.write_json_atomic(
            self.ws / "train_metrics.json",
            train_metrics(self.spec, self.frozen, self.ws, tiny=True),
        )
        runner_mod.write_json_atomic(
            self.ws / "baseline_metrics.json",
            eval_metrics(self.spec, self.frozen, mode="base", examples=1),
        )
        runner_mod.write_json_atomic(
            self.ws / "finetuned_metrics.json",
            eval_metrics(self.spec, self.frozen, mode="adapter", examples=1,
                         adapter=self.ws / "adapter"),
        )
        write_predictions(self.ws / "baseline_predictions.csv", 1)
        write_predictions(self.ws / "finetuned_predictions.csv", 1, mode="finetuned")

    def problems(self, spec=None) -> list[str]:
        return runner_mod.preflight_run_problems(spec or self.spec, self.ws)

    def assert_failure(self, fragment: str, spec=None):
        problems = self.problems(spec)
        self.assertTrue(
            any(fragment in problem for problem in problems),
            f"expected {fragment!r} in {problems}",
        )

    def test_valid_run_passes(self):
        self.assertEqual(self.problems(), [])

    def test_baseline_mode_and_adapter_path_refused(self):
        metrics = runner_mod.read_json(self.ws / "baseline_metrics.json")
        metrics["mode"] = "adapter"
        runner_mod.write_json_atomic(self.ws / "baseline_metrics.json", metrics)
        self.assert_failure("baseline: expected mode/checkpoint 'base'")

    def test_adapter_path_mismatch_refused(self):
        metrics = runner_mod.read_json(self.ws / "finetuned_metrics.json")
        metrics["adapter"] = str(self.ws / "elsewhere")
        runner_mod.write_json_atomic(self.ws / "finetuned_metrics.json", metrics)
        self.assert_failure("adapter: path")

    def test_prediction_rows_must_be_exactly_one(self):
        write_predictions(self.ws / "baseline_predictions.csv", 2)
        self.assert_failure("baseline: 2 prediction rows != 1")
        write_predictions(self.ws / "finetuned_predictions.csv", 3, mode="finetuned")
        self.assert_failure("adapter: 3 prediction rows != 1")

    def test_quantization_parity_refused(self):
        metrics = runner_mod.read_json(self.ws / "finetuned_metrics.json")
        metrics["quantization"]["quant_type"] = ["fp4"]
        runner_mod.write_json_atomic(self.ws / "finetuned_metrics.json", metrics)
        self.assert_failure("quant_type")

    def test_epochs_and_best_checkpoint_refused(self):
        metrics = runner_mod.read_json(self.ws / "train_metrics.json")
        metrics["log_history"] = metrics["log_history"][:2]
        runner_mod.write_json_atomic(self.ws / "train_metrics.json", metrics)
        self.assert_failure("eval epochs")

    def test_vision_like_trainable_parameters_refused(self):
        vision_ws = Path(self.tmp.name) / "vision-ws"
        vision_ws.mkdir()
        create_adapter(vision_ws / "adapter")
        runner_mod.write_json_atomic(
            vision_ws / "train_metrics.json",
            train_metrics(self.vision_spec, self.frozen, vision_ws, tiny=True,
                          trainable=trainable_summary(vision_like=True)),
        )
        runner_mod.write_json_atomic(
            vision_ws / "baseline_metrics.json",
            eval_metrics(self.vision_spec, self.frozen, mode="base", examples=1),
        )
        runner_mod.write_json_atomic(
            vision_ws / "finetuned_metrics.json",
            eval_metrics(self.vision_spec, self.frozen, mode="adapter", examples=1,
                         adapter=vision_ws / "adapter"),
        )
        write_predictions(vision_ws / "baseline_predictions.csv", 1)
        write_predictions(vision_ws / "finetuned_predictions.csv", 1, mode="finetuned")
        problems = runner_mod.preflight_run_problems(self.vision_spec, vision_ws)
        self.assertTrue(any("vision-like" in problem for problem in problems), problems)

    def test_missing_peak_values_refused(self):
        metrics = runner_mod.read_json(self.ws / "train_metrics.json")
        metrics["train_peak_reserved_gib"] = 0
        runner_mod.write_json_atomic(self.ws / "train_metrics.json", metrics)
        peaks = runner_mod.peak_maxima(metrics, {}, {})
        self.assertTrue(peaks["problems"])
        self.assertIsNone(peaks["reserved_gib"])


class DryRunTest(PatchedTrackTest):
    def test_dry_run_validates_without_phases_or_model_calls(self):
        with mock.patch.object(
            runner_mod, "run_phase_subprocess", side_effect=AssertionError("phase subprocess")
        ), mock.patch.object(
            runner_mod, "run_selection_subprocess", side_effect=AssertionError("selection subprocess")
        ):
            code = self.track.runner(dry_run=True).run()
        self.assertEqual(code, 0)
        self.assertFalse(self.track.root.exists())
        self.assertFalse((self.track.benchmarks / "LATEST").exists())

    def test_version_mismatch_and_gpu_probe_failure_refused(self):
        versions = dict(self.track.versions)
        versions["tqdm"] = "0.0.0"
        with mock.patch.object(runner_mod, "probe_versions", return_value=versions):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(dry_run=True).run()
        self.assertIn("tqdm", str(ctx.exception))

        with mock.patch.object(
            runner_mod, "probe_gpu_identity", side_effect=runner_mod.RunnerError("no CUDA")
        ):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(dry_run=True).run()
        self.assertIn("GPU identity probe failed", str(ctx.exception))

        with mock.patch.object(
            runner_mod, "probe_gpu_compute_processes", return_value=["123 python"]
        ):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(dry_run=True).run()
        self.assertIn("foreign GPU", str(ctx.exception))

    def test_local_torch_build_tag_accepted(self):
        versions = dict(self.track.versions, torch="2.11.0+cu128")
        self.assertEqual(runner_mod.check_pinned_versions(versions), [])

    def test_cli_has_no_root_override_and_retry_requires_resume(self):
        parser = runner_mod.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--benchmarks-root", "/tmp/x"])
        with self.assertRaises(SystemExit):
            runner_mod.main(["--retry-failed"])


class PreflightGateTest(PatchedTrackTest):
    def test_amber_peak_repeats_once_then_continues(self):
        slug = "01-qwen3-8b"
        self.track.peak_plan[(slug, "preflight-first-baseline")] = (13.6, 13.7)
        self.track.peak_plan[(slug, "preflight-repeat-baseline")] = (13.65, 13.75)
        code = self.track.runner().run()
        self.assertEqual(code, 0)
        run_dir = self.track.run_dir()
        preflight = runner_mod.read_json(
            run_dir / "models" / slug / "preflight" / "preflight.json"
        )
        self.assertEqual(preflight["first"]["peaks"]["reserved_gib"], 13.7)
        self.assertIsNotNone(preflight["repeat"])
        self.assertEqual(preflight["repeat"]["peaks"]["reserved_gib"], 13.75)
        self.assertEqual(preflight["decision"], "go")
        self.assertEqual(len(preflight["token_lengths"]["train"]), 8)
        self.assertTrue(preflight["first"]["quantization"]["baseline"]["effective_load_in_4bit"])
        # repeat ran in a fresh directory
        self.assertEqual(self.track.calls.count((slug, "preflight-repeat-train")), 1)
        self.assertEqual(self.track.calls.count((slug, "preflight-repeat-adapter")), 1)
        # scratch deleted on go; retained selection/preflight metadata kept
        pre_dir = run_dir / "models" / slug / "preflight"
        self.assertFalse((pre_dir / "attempt-1").exists())
        self.assertTrue((pre_dir / "selection.json").is_file())
        self.assertTrue((pre_dir / "preflight.json").is_file())
        # warmed cache was retained through the run and deleted only by cleanup
        self.assertFalse((run_dir / ".cache" / slug).exists())
        cleanup = runner_mod.read_json(
            run_dir / "models" / slug / "cleanup.json"
        )
        self.assertTrue(any(".cache" in target["path"] for target in cleanup["deleted"]["targets"]))

    def test_red_peak_is_hard_stop_and_attempt_diagnostics_kept(self):
        slug = "01-qwen3-8b"
        self.track.peak_plan[(slug, "preflight-first-baseline")] = (14.5, 14.6)
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["models"][0]["status"], "failed")
        run_info = runner_mod.read_json(run_dir / "models" / slug / "run_info.json")
        attempts = run_info["phases"]["preflight"]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["retryable"])
        self.assertNotIn((slug, "baseline"), self.track.calls)
        pre_dir = run_dir / "models" / slug / "preflight"
        self.assertTrue((pre_dir / "attempt-1" / "first" / "ws").is_dir())
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("not retryable", str(ctx.exception))

    def test_transient_selection_failure_retries_in_fresh_attempt_dir(self):
        slug = "01-qwen3-8b"
        # selection subprocess fails once with a transient marker
        original = self.track.selection

        calls = {"n": 0}

        def flaky_selection(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=False)
                log_path = Path(kwargs["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("connection reset by peer\n")
                return 1
            return original(**kwargs)

        with mock.patch.object(runner_mod, "run_selection_subprocess", side_effect=flaky_selection):
            with self.assertRaises(runner_mod.RunnerError):
                self.track.runner().run()
            run_dir = self.track.run_dir()
            run_info = runner_mod.read_json(run_dir / "models" / slug / "run_info.json")
            self.assertTrue(run_info["phases"]["preflight"]["attempts"][0]["retryable"])
            code = self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertEqual(code, 0)
        pre_dir = run_dir / "models" / slug / "preflight"
        # fresh attempt-2; failed attempt-1 diagnostics retained
        self.assertTrue((pre_dir / "attempt-1").exists())
        self.assertFalse((pre_dir / "attempt-2").exists())  # deleted after successful go
        run_info = runner_mod.read_json(run_dir / "models" / slug / "run_info.json")
        self.assertEqual(len(run_info["phases"]["preflight"]["attempts"]), 2)


class ArtifactValidatorTest(PatchedTrackTest):
    def context(self, run_dir: Path) -> dict:
        slug = "01-qwen3-8b"
        model_dir = run_dir / "models" / slug
        workspace = model_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "spec": runner_mod.spec_by_slug(slug),
            "model_dir": model_dir,
            "workspace": workspace,
            "config_snapshot": model_dir / "config.yaml",
            "config": {},
            "env": {},
            "run_info": {"phases": {}},
            "run_info_path": model_dir / "run_info.json",
            "log_dir": model_dir / "logs",
        }

    def test_baseline_requires_metrics_and_full_predictions(self):
        context = self.context(self.track.root / "RUN")
        problems = runner_mod.artifact_problems("baseline", context)
        self.assertTrue(any("missing baseline_metrics.json" in problem for problem in problems))
        runner_mod.write_json_atomic(
            context["workspace"] / "baseline_metrics.json",
            eval_metrics(context["spec"], self.track.splits, mode="base",
                         examples=self.track.splits["test"]["rows"]),
        )
        problems = runner_mod.artifact_problems("baseline", context)
        self.assertTrue(any("missing baseline_predictions.csv" in problem for problem in problems))
        write_predictions(context["workspace"] / "baseline_predictions.csv", 2)
        problems = runner_mod.artifact_problems("baseline", context)
        self.assertTrue(any("prediction rows" in problem for problem in problems))

    def test_train_requires_complete_adapter_and_checkpoint(self):
        context = self.context(self.track.root / "RUN")
        create_adapter(context["workspace"] / "adapter")
        metrics = train_metrics(context["spec"], self.track.splits, context["workspace"])
        runner_mod.write_json_atomic(context["workspace"] / "train_metrics.json", metrics)
        self.assertEqual(runner_mod.artifact_problems("train", context), [])
        Path(metrics["best_model_checkpoint"], "optimizer.pt").unlink()
        problems = runner_mod.artifact_problems("train", context)
        self.assertTrue(any("optimizer.pt" in problem for problem in problems), problems)
        (context["workspace"] / "adapter" / "adapter_config.json").unlink()
        problems = runner_mod.artifact_problems("train", context)
        self.assertTrue(any("adapter_config.json" in problem for problem in problems), problems)

    def test_comparison_acceptance_cleanup_must_be_parseable(self):
        context = self.context(self.track.root / "RUN")
        self.assertTrue(any("comparison" in problem
                            for problem in runner_mod.artifact_problems("comparison", context)))
        (context["workspace"] / "comparison.json").write_text("{not json", encoding="utf-8")
        self.assertTrue(any("unreadable" in problem
                            for problem in runner_mod.artifact_problems("comparison", context)))
        runner_mod.write_json_atomic(context["workspace"] / "comparison.json", {"test_sha256": {}})
        self.assertEqual(runner_mod.artifact_problems("comparison", context), [])
        self.assertTrue(any("acceptance" in problem
                            for problem in runner_mod.artifact_problems("acceptance", context)))
        runner_mod.write_json_atomic(context["model_dir"] / "acceptance.json",
                                     {"status": "failed"})
        self.assertTrue(any("status" in problem
                            for problem in runner_mod.artifact_problems("acceptance", context)))
        self.assertTrue(any("cleanup" in problem
                            for problem in runner_mod.artifact_problems("cleanup", context)))
        runner_mod.write_json_atomic(context["model_dir"] / "cleanup.json", {"status": "ok"})
        self.assertEqual(runner_mod.artifact_problems("cleanup", context), [])

    def test_preflight_artifact_requires_go_decision_and_selection(self):
        context = self.context(self.track.root / "RUN")
        pre_dir = context["model_dir"] / "preflight"
        runner_mod.write_json_atomic(pre_dir / "preflight.json", {"decision": "repeat"})
        problems = runner_mod.artifact_problems("preflight", context)
        self.assertTrue(any("decision" in problem for problem in problems))
        self.assertTrue(any("selection.json" in problem for problem in problems))
        runner_mod.write_json_atomic(pre_dir / "preflight.json", {"decision": "go"})
        runner_mod.write_json_atomic(
            pre_dir / "selection.json",
            {
                "model": context["spec"].slug,
                "repo": context["spec"].repo,
                "requested_revision": context["spec"].revision,
                "resolved_revision": context["spec"].revision,
                "threshold": runner_mod.MIN_SUPERVISED_TOKENS,
                "selected": {
                    split: [{"length": 2048}] * need
                    for split, need in runner_mod.PREFLIGHT_NEEDS.items()
                },
            },
        )
        self.assertEqual(runner_mod.artifact_problems("preflight", context), [])


class AcceptanceTest(unittest.TestCase):
    spec = runner_mod.EXPECTED_MODELS[0]
    frozen = {
        "train": {"sha256": "a" * 64, "rows": 3},
        "val": {"sha256": "b" * 64, "rows": 2},
        "test": {"sha256": "c" * 64, "rows": 200},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name) / "models" / self.spec.slug / "workspace"
        self.workspace.mkdir(parents=True)
        create_adapter(self.workspace / "adapter")
        self.baseline = eval_metrics(self.spec, self.frozen, mode="base", examples=200)
        self.finetuned = eval_metrics(
            self.spec, self.frozen, mode="adapter", examples=200,
            adapter=self.workspace / "adapter",
        )
        self.train = train_metrics(self.spec, self.frozen, self.workspace)
        self.comparison = {
            "test_sha256": {
                "baseline": self.frozen["test"]["sha256"],
                "finetuned": self.frozen["test"]["sha256"],
                "match": True,
            },
            "base_identity": {"match": True},
        }

    def failures(self, **overrides):
        kwargs = dict(
            spec=self.spec,
            baseline=self.baseline,
            finetuned=self.finetuned,
            train=self.train,
            comparison=self.comparison,
            baseline_prediction_rows=200,
            finetuned_prediction_rows=200,
            workspace=self.workspace,
            vram_peak_reserved_gib=12.0,
            frozen=self.frozen,
        )
        kwargs.update(overrides)
        return runner_mod.acceptance_failures(**kwargs)

    def assert_failure(self, fragment, **overrides):
        problems = self.failures(**overrides)
        self.assertTrue(
            any(fragment in problem for problem in problems),
            f"expected {fragment!r} in {problems}",
        )

    def test_valid_artifacts_pass(self):
        self.assertEqual(self.failures(), [])

    def test_wrong_quantization_refused(self):
        bad = json.loads(json.dumps(self.baseline))
        bad["quantization"]["quant_type"] = ["fp4"]
        self.assert_failure("quant_type", baseline=bad)
        bad = json.loads(json.dumps(self.baseline))
        bad["quantization"]["effective_load_in_4bit"] = False
        self.assert_failure("effective 4-bit", baseline=bad)
        bad = json.loads(json.dumps(self.baseline))
        bad["quantization"]["compute_dtype"] = ["float16"]
        self.assert_failure("compute_dtype", baseline=bad)

    def test_offload_and_placement_refused(self):
        bad = json.loads(json.dumps(self.finetuned))
        bad["quantization"]["offload"]["cpu"] = ["model.embed_tokens"]
        self.assert_failure("offload", finetuned=bad)
        bad = json.loads(json.dumps(self.train))
        bad["quantization"]["non_cuda_parameter_count"] = 2
        self.assert_failure("not on CUDA", train=bad)

    def test_wrong_revision_refused(self):
        bad = json.loads(json.dumps(self.baseline))
        bad["requested_revision"] = "0" * 40
        bad["resolved_revision"] = "0" * 40
        self.assert_failure("revision", baseline=bad)

    def test_modes_and_adapter_path_refused(self):
        bad = json.loads(json.dumps(self.baseline))
        bad["mode"] = "adapter"
        self.assert_failure("expected mode/checkpoint 'base'", baseline=bad)
        bad = json.loads(json.dumps(self.finetuned))
        bad["mode"] = "base"
        self.assert_failure("mode", finetuned=bad)
        bad = json.loads(json.dumps(self.finetuned))
        bad["adapter"] = str(self.workspace / "elsewhere")
        self.assert_failure("adapter", finetuned=bad)

    def test_frozen_hash_mismatch_refused(self):
        bad = json.loads(json.dumps(self.train))
        bad["sha256"]["train"] = "d" * 64
        self.assert_failure("train sha256", train=bad)
        bad = json.loads(json.dumps(self.baseline))
        bad["test_sha256"] = "e" * 64
        self.assert_failure("test_sha256", baseline=bad)

    def test_epochs_finiteness_and_best_consistency(self):
        bad = json.loads(json.dumps(self.train))
        bad["log_history"] = bad["log_history"][:2]
        self.assert_failure("eval epochs", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["log_history"][1]["eval_loss"] = float("nan")
        self.assert_failure("not finite", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["best_model_checkpoint"] = str(self.workspace / "trainer" / "checkpoint-999")
        self.assert_failure("not found on disk", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["best_metric"] = 0.9
        self.assert_failure("best_metric", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["best_epoch"] = 1.0
        self.assert_failure("best_epoch", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["final_epoch"] = 2.0
        self.assert_failure("final_epoch", train=bad)
        bad = json.loads(json.dumps(self.train))
        bad["final_eval_epoch"] = None
        self.assert_failure("final_eval_epoch", train=bad)

    def test_checkpoint_step_history_consistency(self):
        create_checkpoint(self.workspace / "trainer" / "checkpoint-200", 200)
        bad = json.loads(json.dumps(self.train))
        bad["best_model_checkpoint"] = str(self.workspace / "trainer" / "checkpoint-200")
        self.assert_failure("matching log_history entry", train=bad)

    def test_vision_trainable_parameters_refused(self):
        vision_spec = runner_mod.EXPECTED_MODELS[1]
        vision_ws = Path(self.tmp.name) / "vision-ws"
        vision_ws.mkdir()
        create_adapter(vision_ws / "adapter")
        vision_train = train_metrics(vision_spec, self.frozen, vision_ws,
                                     trainable=trainable_summary(vision_like=True))
        problems = runner_mod.acceptance_failures(
            spec=vision_spec,
            baseline=eval_metrics(vision_spec, self.frozen, mode="base", examples=200),
            finetuned=eval_metrics(vision_spec, self.frozen, mode="adapter", examples=200,
                                   adapter=vision_ws / "adapter"),
            train=vision_train,
            comparison=self.comparison,
            baseline_prediction_rows=200,
            finetuned_prediction_rows=200,
            workspace=vision_ws,
            vram_peak_reserved_gib=12.0,
            frozen=self.frozen,
        )
        self.assertTrue(any("vision-like" in problem for problem in problems), problems)

        missing_summary = json.loads(json.dumps(vision_train))
        del missing_summary["trainable_parameters"]
        problems = runner_mod.acceptance_failures(
            spec=vision_spec,
            baseline=eval_metrics(vision_spec, self.frozen, mode="base", examples=200),
            finetuned=eval_metrics(vision_spec, self.frozen, mode="adapter", examples=200,
                                   adapter=vision_ws / "adapter"),
            train=missing_summary,
            comparison=self.comparison,
            baseline_prediction_rows=200,
            finetuned_prediction_rows=200,
            workspace=vision_ws,
            vram_peak_reserved_gib=12.0,
            frozen=self.frozen,
        )
        self.assertTrue(any("trainable_parameters summary missing" in problem
                            for problem in problems), problems)

    def test_unavailable_trainable_inspection_blocks_vision_only(self):
        unavailable = {"trainable_parameters": {"status": "unavailable", "reason": "no api"}}
        self.assertTrue(
            runner_mod.trainable_params_problems(unavailable, runner_mod.EXPECTED_MODELS[1])
        )
        self.assertEqual(
            runner_mod.trainable_params_problems(unavailable, runner_mod.EXPECTED_MODELS[0]), []
        )

    def test_prediction_and_example_counts_refused(self):
        self.assert_failure("prediction rows", baseline_prediction_rows=199)
        self.assert_failure("prediction rows", finetuned_prediction_rows=201)
        bad = json.loads(json.dumps(self.finetuned))
        bad["examples"] = 199
        self.assert_failure("examples", finetuned=bad)

    def test_vram_threshold_and_peak_problems_refused(self):
        self.assert_failure("VRAM", vram_peak_reserved_gib=13.9)
        self.assert_failure("VRAM", vram_peak_reserved_gib=None)
        self.assert_failure("VRAM", vram_peak_reserved_gib=14.4, vram_repeat_approved=True)
        self.assert_failure("headroom", vram_peak_reserved_gib=13.2, vram_total_gib=14.0)
        self.assert_failure("missing peak", vram_peak_problems=["missing peak train.x"])
        self.assertEqual(
            self.failures(vram_peak_reserved_gib=13.9, vram_repeat_approved=True), []
        )
        self.assertEqual(
            self.failures(vram_peak_reserved_gib=12.0, vram_total_gib=15.46), []
        )


class ManifestLifecycleTest(PatchedTrackTest):
    def test_four_successes_complete_manifest_then_write_track_latest(self):
        legacy = self.track.protected[0]
        legacy.mkdir(parents=True)
        legacy_file = legacy / "manifest.json"
        legacy_file.write_text('{"status": "completed"}\n', encoding="utf-8")
        root_latest = self.track.benchmarks / "LATEST"
        root_latest.write_text("20260915T112519Z\n", encoding="utf-8")

        code = self.track.runner().run()
        self.assertEqual(code, 0)
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["track"], runner_mod.TRACK)
        self.assertEqual(manifest["git_commit"], "deadbeef")
        self.assertTrue(manifest["latest_written"])
        self.assertEqual(manifest["gpu"]["name"], self.track.gpu["name"])
        self.assertEqual([model["status"] for model in manifest["models"]], ["completed"] * 4)
        self.assertEqual(
            (self.track.root / "LATEST").read_text().strip(), run_dir.name
        )
        for model in manifest["models"]:
            run_info = runner_mod.read_json(run_dir / "models" / model["slug"] / "run_info.json")
            self.assertEqual(run_info["status"], "completed")
            self.assertTrue(run_info["completed_at"])
            self.assertNotIn("failed_at", run_info)
            self.assertEqual(
                [run_info["phases"][phase]["status"] for phase in runner_mod.PHASES],
                ["ok"] * len(runner_mod.PHASES),
            )
        model_dir = run_dir / "models" / manifest["models"][0]["slug"]
        workspace = model_dir / "workspace"
        for name in ("baseline_metrics.json", "train_metrics.json", "finetuned_metrics.json",
                     "baseline_predictions.csv", "finetuned_predictions.csv", "comparison.json"):
            self.assertTrue((workspace / name).is_file(), name)
        self.assertFalse((workspace / "train.jsonl").exists())
        self.assertTrue((model_dir / "preflight" / "preflight.json").is_file())
        self.assertTrue((model_dir / "preflight" / "selection.json").is_file())
        self.assertFalse((model_dir / "preflight" / "attempt-1").exists())
        self.assertFalse((run_dir / ".cache" / manifest["models"][0]["slug"]).exists())
        self.assertTrue((model_dir / "cleanup.json").is_file())
        self.assertEqual(legacy_file.read_text(encoding="utf-8"), '{"status": "completed"}\n')
        self.assertEqual(root_latest.read_text(encoding="utf-8"), "20260915T112519Z\n")

    def test_failure_halts_partial_and_never_writes_latest(self):
        self.track.plan_error("02-ministral-3-8b-instruct", "train", log="CUDA out of memory")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(
            [model["status"] for model in manifest["models"]],
            ["completed", "failed", "pending", "pending"],
        )
        self.assertEqual(manifest["models"][1]["failure"]["phase"], "train")
        self.assertFalse(manifest["models"][1]["failure"]["retryable"])
        self.assertFalse(manifest["latest_written"])
        self.assertFalse((self.track.root / "LATEST").exists())
        self.assertFalse((run_dir / "models" / "03-qwen3.5-9b").exists())
        run_info = runner_mod.read_json(
            run_dir / "models" / "02-ministral-3-8b-instruct" / "run_info.json"
        )
        self.assertEqual(run_info["status"], "failed")
        self.assertTrue(run_info["failed_at"])
        self.assertTrue(run_info["phases"]["train"]["failed_at"])

    def test_latest_write_failure_surfaces_without_lying(self):
        # Fail only the LATEST replace; manifest/run_info atomic writes still work.
        real_replace = runner_mod.os.replace  # noqa: lookup before patch

        def selective_replace(src, dst):
            if str(dst).endswith("LATEST"):
                raise OSError("disk full")
            return real_replace(src, dst)

        with mock.patch.object(runner_mod.os, "replace", side_effect=selective_replace):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner().run()
        self.assertIn("LATEST", str(ctx.exception))
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse(manifest["latest_written"])
        self.assertIn("disk full", manifest["latest_error"])
        self.assertFalse((self.track.root / "LATEST").exists())
        self.assertEqual(list(self.track.root.glob("LATEST.tmp*")), [])


class ResumeTest(PatchedTrackTest):
    def test_successful_phases_are_never_rerun_on_resume(self):
        self.assertEqual(self.track.runner().run(), 0)
        run_dir = self.track.run_dir()
        calls_before = list(self.track.calls)
        self.assertEqual(self.track.runner(resume=run_dir).run(), 0)
        self.assertEqual(self.track.calls, calls_before)
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "completed")

    def test_transient_failure_resumes_with_one_explicit_retry(self):
        self.track.plan_error("02-ministral-3-8b-instruct", "baseline",
                              log="requests.exceptions.ConnectionError: connection reset by peer")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "partial")

        calls_before = list(self.track.calls)
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir).run()
        self.assertIn("--retry-failed", str(ctx.exception))
        self.assertEqual(self.track.calls, calls_before)

        self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual([model["status"] for model in manifest["models"]], ["completed"] * 4)
        run_info = runner_mod.read_json(
            run_dir / "models" / "02-ministral-3-8b-instruct" / "run_info.json"
        )
        info = run_info["phases"]["baseline"]
        self.assertEqual(len(info["attempts"]), 2)
        self.assertTrue(info["attempts"][0]["retryable"])
        self.assertEqual(info["attempts"][0]["status"], "failed")
        self.assertNotIn("failed_at", info)
        self.assertTrue(info["finished_at"])
        self.assertEqual(self.track.calls.count(("01-qwen3-8b", "baseline")), 1)
        self.assertEqual(self.track.calls.count(("02-ministral-3-8b-instruct", "baseline")), 2)
        self.assertEqual(self.track.calls.count(("02-ministral-3-8b-instruct", "train")), 1)

    def test_unknown_and_hard_failures_refuse_retry(self):
        self.track.plan_error("01-qwen3-8b", "train", log="RuntimeError: unexpected bug in trainer")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("not retryable", str(ctx.exception))

        track = FakeTrack(Path(self.tmp.name) / "hard")
        for patcher in track.patches():
            patcher.start()
            self.addCleanup(patcher.stop)
        track.plan_error("01-qwen3-8b", "train", log="CUDA out of memory at step 3")
        with self.assertRaises(runner_mod.RunnerError):
            track.runner().run()
        run_dir = next(path for path in track.root.iterdir() if path.is_dir())
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("not retryable", str(ctx.exception))

    def test_retry_limit_enforced(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner(resume=run_dir, retry_failed=True).run()
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("one retry", str(ctx.exception))

    def test_train_retry_resumes_complete_checkpoint_with_attempt_config(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        workspace = run_dir / "models" / "01-qwen3-8b" / "workspace"
        # simulate a crash after the trainer saved a complete checkpoint
        create_adapter(workspace / "adapter")
        checkpoint = workspace / "trainer" / "checkpoint-400"
        create_checkpoint(checkpoint, 400)
        self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)
        attempt = runner_mod.read_json(
            run_dir / "models" / "01-qwen3-8b" / "run_info.json"
        )["phases"]["train"]["attempts"][1]
        self.assertEqual(attempt["retry"]["mode"], "resume")
        self.assertEqual(attempt["retry"]["step"], 400)
        config_path = Path(attempt["retry"]["config"])
        self.assertTrue(config_path.is_file())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["training"]["resume_from_checkpoint"], str(checkpoint))
        # the retried subprocess used the attempt config
        self.assertIn(
            str(config_path),
            [path for slug, stem, path in self.track.configs_seen
             if slug == "01-qwen3-8b" and stem == "train"],
        )

    def test_train_retry_with_only_incomplete_checkpoint_restarts_fresh(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        workspace = run_dir / "models" / "01-qwen3-8b" / "workspace"
        create_adapter(workspace / "adapter")
        incomplete = workspace / "trainer" / "checkpoint-7"
        create_checkpoint(incomplete, 7)
        (incomplete / "optimizer.pt").unlink()
        (workspace / "train_metrics.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)
        attempt = runner_mod.read_json(
            run_dir / "models" / "01-qwen3-8b" / "run_info.json"
        )["phases"]["train"]["attempts"][1]
        self.assertEqual(attempt["retry"]["mode"], "fresh_restart")
        self.assertIn(str(incomplete), attempt["retry"]["incomplete_checkpoints"])
        self.assertFalse(incomplete.exists())

    def test_resume_identity_mismatches_refused(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()

        with mock.patch.object(runner_mod, "probe_git_head", return_value="cafebabe"):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("git commit", str(ctx.exception))

        versions = dict(self.track.versions)
        versions["transformers"] = "9.9.9"
        with mock.patch.object(runner_mod, "probe_versions", return_value=versions):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("environment", str(ctx.exception))

        other_gpu = dict(self.track.gpu, name="Other GPU")
        with mock.patch.object(runner_mod, "probe_gpu_identity", return_value=other_gpu):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("GPU name", str(ctx.exception))

        config_path = self.track.project / runner_mod.EXPECTED_MODELS[0].config
        config = load_config(config_path)
        config["training"]["learning_rate"] = 1.0
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("source config sha256", str(ctx.exception))

    def test_resume_data_hash_mismatch_refused(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        with (self.track.data_dir / "train.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"issue_number": 999, "input": "x", "label": "bug"}\n')
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("frozen train data", str(ctx.exception))

    def test_snapshot_digest_mismatch_refused(self):
        self.track.plan_error("01-qwen3-8b", "train", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        snapshot = run_dir / "models" / "01-qwen3-8b" / "config.yaml"
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir, retry_failed=True).run()
        self.assertIn("snapshot sha256", str(ctx.exception))

    def test_resume_completes_pending_cleanup_after_acceptance(self):
        self.assertEqual(self.track.runner().run(), 0)
        run_dir = self.track.run_dir()
        slug = "01-qwen3-8b"
        model_dir = run_dir / "models" / slug
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        run_info["phases"]["cleanup"] = {"status": "pending", "attempts": []}
        run_info["status"] = "completed"
        runner_mod.write_json_atomic(model_dir / "run_info.json", run_info)
        (model_dir / "cleanup.json").unlink()
        # recreate scratch so cleanup has something real to delete
        create_adapter(model_dir / "workspace" / "adapter")
        (model_dir / "workspace" / "train.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertEqual(self.track.runner(resume=run_dir).run(), 0)
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        self.assertEqual(run_info["phases"]["cleanup"]["status"], "ok")
        self.assertFalse((model_dir / "workspace" / "adapter").exists())
        self.assertFalse((model_dir / "workspace" / "train.jsonl").exists())
        self.assertTrue((model_dir / "cleanup.json").is_file())

    def test_phase_partials_cleared_before_retry(self):
        self.track.plan_error("01-qwen3-8b", "baseline", log="timed out")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        workspace = run_dir / "models" / "01-qwen3-8b" / "workspace"
        stale = workspace / "baseline_metrics.json"
        stale.write_text('{"stale": true}\n', encoding="utf-8")
        (workspace / "baseline_predictions.csv").write_text("stale\n", encoding="utf-8")
        self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)
        attempt = runner_mod.read_json(
            run_dir / "models" / "01-qwen3-8b" / "run_info.json"
        )["phases"]["baseline"]["attempts"][1]
        cleared = {Path(target["path"]).name for target in attempt["retry"]["cleared"]["targets"]}
        self.assertEqual(cleared, {"baseline_metrics.json", "baseline_predictions.csv"})
        metrics = runner_mod.read_json(stale)
        self.assertIn("quantization", metrics)

    def test_canonical_phase_names_in_failures(self):
        self.track.plan_error("01-qwen3-8b", "adapter_eval", log="CUDA out of memory")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["models"][0]["failure"]["phase"], "adapter_eval")
        run_info = runner_mod.read_json(run_dir / "models" / "01-qwen3-8b" / "run_info.json")
        attempt = run_info["phases"]["adapter_eval"]["attempts"][0]
        self.assertIn("adapter_eval: evaluate exit code", attempt["error"])

    def test_interrupted_phase_can_be_retried_after_resume(self):
        calls = {"n": 0}
        original = self.track.phase

        def interrupt_once(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt()
            return original(**kwargs)

        with mock.patch.object(runner_mod, "run_phase_subprocess", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.track.runner().run()
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertNotEqual(manifest["status"], "completed")
        run_info = runner_mod.read_json(run_dir / "models" / "01-qwen3-8b" / "run_info.json")
        attempt = run_info["phases"]["preflight"]["attempts"][0]
        self.assertEqual(attempt["status"], "failed")
        self.assertIn("interrupted", attempt["error"])
        self.assertEqual(self.track.runner(resume=run_dir, retry_failed=True).run(), 0)

    def test_resume_refuses_outside_track_root(self):
        outside = Path(self.tmp.name) / "elsewhere"
        outside.mkdir(parents=True)
        (outside / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(runner_mod.RunnerError):
            self.track.runner(resume=outside).run()


class CleanupPhaseTest(PatchedTrackTest):
    def test_cleanup_probe_failure_fails_closed(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            if calls["n"] > 2:
                raise runner_mod.RunnerError("nvidia-smi broken")
            return []

        with mock.patch.object(runner_mod, "probe_gpu_compute_processes", side_effect=probe):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner().run()
        self.assertIn("nvidia-smi broken", str(ctx.exception))
        run_dir = self.track.run_dir()
        model_dir = run_dir / "models" / "01-qwen3-8b"
        self.assertFalse((model_dir / "cleanup.json").exists())
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        self.assertEqual(run_info["failure"]["phase"], "cleanup")

    def test_gpu_busy_before_cleanup_blocks_deletion(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            # invariant checks pass; the pre-cleanup probe sees a foreign process
            return ["123 python"] if calls["n"] > 2 else []

        with mock.patch.object(runner_mod, "probe_gpu_compute_processes", side_effect=probe):
            with self.assertRaises(runner_mod.RunnerError):
                self.track.runner().run()
        run_dir = self.track.run_dir()
        model_dir = run_dir / "models" / "01-qwen3-8b"
        self.assertFalse((model_dir / "cleanup.json").exists())
        self.assertTrue((model_dir / "workspace" / "adapter").exists())
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        self.assertEqual(run_info["status"], "failed")
        self.assertEqual(run_info["failure"]["phase"], "cleanup")
        self.assertEqual(run_info["phases"]["cleanup"]["status"], "failed")

    def test_gpu_busy_after_cleanup_records_and_fails(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            return ["999 stale"] if calls["n"] > 3 else []

        with mock.patch.object(runner_mod, "probe_gpu_compute_processes", side_effect=probe):
            with self.assertRaises(runner_mod.RunnerError):
                self.track.runner().run()
        run_dir = self.track.run_dir()
        record = runner_mod.read_json(run_dir / "models" / "01-qwen3-8b" / "cleanup.json")
        self.assertEqual(record["status"], "gpu-busy")
        self.assertEqual(record["gpu_after"], ["999 stale"])
        self.assertGreater(record["deleted"]["bytes_deleted"], 0)

    def test_cleanup_phase_order_and_record(self):
        code = self.track.runner().run()
        self.assertEqual(code, 0)
        run_dir = self.track.run_dir()
        record = runner_mod.read_json(run_dir / "models" / "01-qwen3-8b" / "cleanup.json")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["gpu_before"], [])
        self.assertEqual(record["gpu_after"], [])
        self.assertTrue(record["deleted"]["targets"])
        run_info = runner_mod.read_json(run_dir / "models" / "01-qwen3-8b" / "run_info.json")
        self.assertEqual(run_info["cleanup"]["status"], "ok")


class GpuProcessProbeTest(unittest.TestCase):
    def test_empty_output_is_idle(self):
        self.assertEqual(runner_mod.parse_gpu_compute_processes(""), [])
        self.assertEqual(
            runner_mod.parse_gpu_compute_processes("No running processes found\n"), []
        )

    def test_valid_rows_returned(self):
        rows = runner_mod.parse_gpu_compute_processes("12345, python, 512 MiB\n")
        self.assertEqual(rows, ["12345, python, 512 MiB"])

    def test_malformed_output_raises(self):
        for text in ("garbage line\n", "no-pid, python, 512 MiB\n"):
            with self.assertRaises(runner_mod.RunnerError):
                runner_mod.parse_gpu_compute_processes(text)

    def test_probe_fails_closed_on_oserror_nonzero_and_malformed(self):
        with mock.patch.object(runner_mod.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                runner_mod.probe_gpu_compute_processes()
        self.assertIn("failed", str(ctx.exception))

        result = mock.Mock(returncode=1, stdout="", stderr="driver error")
        with mock.patch.object(runner_mod.subprocess, "run", return_value=result):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                runner_mod.probe_gpu_compute_processes()
        self.assertIn("exit code 1", str(ctx.exception))

        result = mock.Mock(returncode=0, stdout="junk\n", stderr="")
        with mock.patch.object(runner_mod.subprocess, "run", return_value=result):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                runner_mod.probe_gpu_compute_processes()
        self.assertIn("malformed", str(ctx.exception))

        result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(runner_mod.subprocess, "run", return_value=result):
            self.assertEqual(runner_mod.probe_gpu_compute_processes(), [])

    def test_invariant_propagates_probe_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = FakeTrack(Path(tmp))
            with mock.patch.object(
                runner_mod, "probe_gpu_compute_processes",
                side_effect=runner_mod.RunnerError("probe down"),
            ):
                with self.assertRaises(runner_mod.RunnerError) as ctx:
                    track.runner(dry_run=True).run()
            self.assertIn("probe down", str(ctx.exception))


class TrackLatestTest(PatchedTrackTest):
    def test_atomic_replace_leaves_no_temp(self):
        self.track.root.mkdir(parents=True, exist_ok=True)
        runner_mod.write_track_latest(self.track.root, "20260101T000000Z")
        self.assertEqual((self.track.root / "LATEST").read_text().strip(), "20260101T000000Z")
        self.assertEqual(list(self.track.root.glob("LATEST.tmp*")), [])

    def test_failed_replace_preserves_existing_and_cleans_temp(self):
        self.track.root.mkdir(parents=True, exist_ok=True)
        latest = self.track.root / "LATEST"
        latest.write_text("old-run\n", encoding="utf-8")
        with mock.patch.object(runner_mod.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                runner_mod.write_track_latest(self.track.root, "new-run")
        self.assertEqual(latest.read_text(encoding="utf-8"), "old-run\n")
        self.assertEqual(list(self.track.root.glob("LATEST.tmp*")), [])


class CheckpointIntegrityTest(PatchedTrackTest):
    def checkpoint(self, name: str = "checkpoint-400", step: int = 400) -> Path:
        path = self.track.root / "ckpt" / name
        create_checkpoint(path, step)
        return path

    def test_valid_checkpoint_passes(self):
        self.assertEqual(runner_mod.complete_checkpoint_problems(self.checkpoint()), [])

    def test_step_mismatch_refused(self):
        problems = runner_mod.complete_checkpoint_problems(
            self.checkpoint("checkpoint-400", 399)
        )
        self.assertTrue(any("global_step 399" in problem for problem in problems), problems)

    def test_zero_byte_file_refused(self):
        checkpoint = self.checkpoint()
        (checkpoint / "optimizer.pt").write_bytes(b"")
        problems = runner_mod.complete_checkpoint_problems(checkpoint)
        self.assertTrue(any("optimizer.pt: zero-byte" in problem for problem in problems), problems)

    def test_corrupt_and_truncated_safetensors_refused(self):
        checkpoint = self.checkpoint()
        (checkpoint / "adapter_model.safetensors").write_bytes(b"short")
        problems = runner_mod.complete_checkpoint_problems(checkpoint)
        self.assertTrue(any("safetensors" in problem for problem in problems), problems)
        # valid header but truncated payload
        header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (checkpoint / "adapter_model.safetensors").write_bytes(
            len(header).to_bytes(8, "little") + header
        )
        problems = runner_mod.complete_checkpoint_problems(checkpoint)
        self.assertTrue(
            any("exceeds file size" in problem or "unreadable" in problem for problem in problems),
            problems,
        )

    def test_corrupt_json_and_torch_states_refused(self):
        checkpoint = self.checkpoint()
        (checkpoint / "trainer_state.json").write_text("{not json", encoding="utf-8")
        (checkpoint / "scheduler.pt").write_bytes(b"not a torch state")
        problems = runner_mod.complete_checkpoint_problems(checkpoint)
        self.assertTrue(any("trainer_state.json" in problem for problem in problems), problems)
        self.assertTrue(any("scheduler.pt" in problem for problem in problems), problems)


class CompletedModelEvidenceTest(PatchedTrackTest):
    spec = runner_mod.EXPECTED_MODELS[0]

    def completed_run(self) -> tuple[Path, dict]:
        self.assertEqual(self.track.runner().run(), 0)
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        return run_dir, manifest

    def test_valid_completion_passes_and_corruption_is_detected(self):
        run_dir, manifest = self.completed_run()
        self.assertEqual(
            runner_mod.completed_model_problems(self.spec, run_dir, manifest), []
        )
        model_dir = run_dir / "models" / self.spec.slug
        workspace = model_dir / "workspace"
        baseline = workspace / "baseline_metrics.json"
        content = baseline.read_text(encoding="utf-8")
        baseline.unlink()
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(any("baseline_metrics.json" in problem for problem in problems), problems)
        baseline.write_text(content, encoding="utf-8")

        selection_path = model_dir / "preflight" / "selection.json"
        selection = runner_mod.read_json(selection_path)
        selection["resolved_revision"] = "0" * 40
        runner_mod.write_json_atomic(selection_path, selection)
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(any("resolved revision" in problem for problem in problems), problems)

        preflight_path = model_dir / "preflight" / "preflight.json"
        preflight = runner_mod.read_json(preflight_path)
        preflight["decision"] = "repeat"
        runner_mod.write_json_atomic(preflight_path, preflight)
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(any("preflight decision" in problem for problem in problems), problems)

        cleanup_path = model_dir / "cleanup.json"
        cleanup = runner_mod.read_json(cleanup_path)
        cleanup["status"] = "gpu-busy"
        runner_mod.write_json_atomic(cleanup_path, cleanup)
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(any("cleanup.json status" in problem for problem in problems), problems)

        snapshot = model_dir / "config.yaml"
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8")
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(any("config.yaml digest" in problem for problem in problems), problems)

    def test_resume_refuses_corrupt_completed_evidence_without_rerunning(self):
        run_dir, manifest = self.completed_run()
        model_dir = run_dir / "models" / self.spec.slug
        (model_dir / "workspace" / "baseline_metrics.json").unlink()
        calls_before = list(self.track.calls)
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            self.track.runner(resume=run_dir).run()
        self.assertIn("evidence is invalid", str(ctx.exception))
        self.assertEqual(self.track.calls, calls_before)
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["models"][0]["status"], "failed")
        self.assertEqual(manifest["models"][0]["failure"]["phase"], "completed_model")
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        self.assertEqual(run_info["status"], "failed")
        self.assertTrue(run_info["failed_at"])

    def test_pending_cleanup_only_is_resumable(self):
        run_dir, manifest = self.completed_run()
        model_dir = run_dir / "models" / self.spec.slug
        run_info = runner_mod.read_json(model_dir / "run_info.json")
        run_info["phases"]["cleanup"] = {"status": "pending", "attempts": []}
        runner_mod.write_json_atomic(model_dir / "run_info.json", run_info)
        (model_dir / "cleanup.json").unlink()
        problems = runner_mod.completed_model_problems(self.spec, run_dir, manifest)
        self.assertTrue(problems)
        self.assertEqual(
            runner_mod.completed_model_problems(
                self.spec, run_dir, manifest, require_cleanup=False
            ),
            [],
        )
        self.assertEqual(self.track.runner(resume=run_dir).run(), 0)
        self.assertEqual(
            runner_mod.read_json(model_dir / "run_info.json")["phases"]["cleanup"]["status"],
            "ok",
        )


class SetupFailureTest(PatchedTrackTest):
    def test_setup_failure_recorded_as_runner_setup_and_stops(self):
        calls = {"n": 0}

        def disk(path):
            calls["n"] += 1
            return 120.0 if calls["n"] <= 2 else 1.0

        with mock.patch.object(runner_mod, "probe_disk_free_gib", side_effect=disk):
            with self.assertRaises(runner_mod.RunnerError) as ctx:
                self.track.runner().run()
        self.assertIn("runner_setup", str(ctx.exception))
        run_dir = self.track.run_dir()
        manifest = runner_mod.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["models"][0]["status"], "completed")
        self.assertEqual(manifest["models"][1]["status"], "failed")
        self.assertEqual(manifest["models"][1]["failure"]["phase"], "runner_setup")
        self.assertEqual(manifest["models"][2]["status"], "pending")
        run_info = runner_mod.read_json(
            run_dir / "models" / "02-ministral-3-8b-instruct" / "run_info.json"
        )
        self.assertEqual(run_info["status"], "failed")
        self.assertTrue(run_info["failed_at"])
        self.assertEqual(run_info["failure"]["phase"], "runner_setup")
        self.assertFalse(
            any(slug == "02-ministral-3-8b-instruct" for slug, _stem in self.track.calls)
        )


if __name__ == "__main__":
    unittest.main()
