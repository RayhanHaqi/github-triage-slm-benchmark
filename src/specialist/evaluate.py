"""Shared evaluator used for both baseline and fine-tuned runs.

Test file, prompt, truncation, decoding, parsers, metrics and artifact formats
are identical between modes; only the model (base vs base+PeftModel) and the
prediction CSV column set (frozen-compatible per mode) differ.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import model as model_mod

try:  # optional nicety; not required for correctness
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# Frozen baseline_eval.py wrote the first five columns; finetuned_eval.py added
# the timing/token columns. Keep each mode byte-format compatible with its frozen
# counterpart; the metrics JSON carries the full measurement set for both.
BASELINE_FIELDS = [
    "issue_number",
    "target",
    "prediction",
    "strict_prediction",
    "raw_output",
]
FINETUNED_FIELDS = BASELINE_FIELDS + ["input_tokens", "output_tokens", "latency_ms"]


def resolve_checkpoint(checkpoint: str) -> tuple[str, str | None]:
    """Resolve `--checkpoint` to (mode, target).

    - `base` (case-insensitive, or empty)        -> ("base", None)
    - path to a dir containing adapter_config    -> ("adapter", path)
    - anything else (HF model id or model dir)   -> ("base", value), evaluated as base
    """
    value = str(checkpoint).strip()
    if not value or value.lower() == "base":
        return "base", None
    candidate = Path(value).expanduser()
    if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
        return "adapter", str(candidate)
    return "base", value


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(config: dict, workspace: str | Path, checkpoint: str) -> dict:
    """Evaluate `checkpoint` on the configured split; write metrics JSON + predictions CSV."""
    import torch

    workspace = Path(workspace)
    evaluation = config["evaluation"]
    split = evaluation.get("split", "test")
    split_path = workspace / f"{split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"{split} split not found: {split_path} (run prepare first)")

    # Hash the exact bytes used for the run; comparison.json must match this.
    test_sha256 = model_mod.sha256_file(split_path)

    model_cfg = config["model"]
    base_model = model_cfg["base_model"]
    system_prompt = model_cfg["system_prompt"]
    kind = model_cfg.get("kind", "language")
    labels = list(config["sources"].keys())
    max_user_tokens = int(evaluation["max_user_tokens"])
    max_new_tokens = int(evaluation["max_new_tokens"])
    do_sample = bool(evaluation["do_sample"])
    pad_eos = bool(evaluation.get("pad_eos", True))
    chat_template_kwargs = model_mod.configured_chat_template_kwargs(evaluation)

    mode, target = resolve_checkpoint(checkpoint)
    phase = "baseline" if mode == "base" else "finetuned"

    print(f"Loading base model: {base_model} (kind={kind})")
    if mode == "adapter":
        print(f"Loading LoRA adapter: {target}")

    loaded = model_mod.load_model(
        base_model,
        target if mode == "adapter" else None,
        revision=model_cfg.get("revision"),
        kind=kind,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model, tokenizer = loaded.model, loaded.tokenizer

    device = next(model.parameters()).device
    print(f"Model device: {device}")

    rows = load_jsonl(split_path)
    print(f"{split.capitalize()} examples: {len(rows)}")

    # Reset CUDA memory counters after load so peaks cover inference only.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    results = []
    semantic_correct = 0
    strict_correct = 0
    valid_predictions = 0
    latencies = []
    total_generated_tokens = 0
    total_generation_time = 0.0
    input_token_counts = []
    confusion = defaultdict(Counter)

    for row in tqdm(rows):
        target_label = row["label"]
        user_text = row["input"]

        prompt, _user_tokens = model_mod.build_prompt(
            loaded.prompt_source,
            user_text,
            system_prompt,
            max_user_tokens,
            chat_template_kwargs,
        )

        inputs = model_mod.tokenize_prompt(loaded, prompt)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        input_token_counts.append(input_length)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id if pad_eos else None,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        latencies.append(elapsed)
        total_generation_time += elapsed

        generated_ids = output[0][input_length:]
        num_generated_tokens = len(generated_ids)
        total_generated_tokens += num_generated_tokens

        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        strict = model_mod.strict_prediction(raw_output, labels)
        parsed = model_mod.parse_prediction(raw_output, labels)

        if strict == target_label:
            strict_correct += 1
        if parsed is not None:
            valid_predictions += 1
        if parsed == target_label:
            semantic_correct += 1

        predicted_for_matrix = parsed if parsed is not None else "INVALID"
        confusion[target_label][predicted_for_matrix] += 1

        results.append({
            "issue_number": row.get("issue_number"),
            "target": target_label,
            "prediction": parsed or "INVALID",
            "strict_prediction": strict or "INVALID",
            "raw_output": raw_output,
            "input_tokens": input_length,
            "output_tokens": num_generated_tokens,
            "latency_ms": elapsed * 1000,
        })

    n = len(rows)
    strict_accuracy = strict_correct / n if n else 0.0
    semantic_accuracy = semantic_correct / n if n else 0.0
    valid_output_rate = valid_predictions / n if n else 0.0

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    median_latency = statistics.median(latencies) if latencies else 0.0
    issues_per_second = n / total_generation_time if total_generation_time > 0 else 0.0
    output_tokens_per_second = (
        total_generated_tokens / total_generation_time
        if total_generation_time > 0
        else 0.0
    )
    mean_input_tokens = statistics.mean(input_token_counts) if input_token_counts else 0.0
    median_input_tokens = statistics.median(input_token_counts) if input_token_counts else 0.0

    if torch.cuda.is_available():
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0

    per_class_recall = {}
    per_class_support = {}
    for label in labels:
        support = sum(1 for r in results if r["target"] == label)
        correct = sum(
            1 for r in results if r["target"] == label and r["prediction"] == label
        )
        per_class_support[label] = support
        per_class_recall[label] = (correct / support) if support else None

    confusion_matrix = {
        actual: {
            predicted: confusion[actual][predicted]
            for predicted in labels + ["INVALID"]
        }
        for actual in labels
    }
    predicted_counts = {
        predicted: sum(1 for r in results if r["prediction"] == predicted)
        for predicted in labels + ["INVALID"]
    }

    metrics = {
        "phase": phase,
        "checkpoint": checkpoint,
        "mode": mode,
        "base_model": base_model,
        "adapter": target if mode == "adapter" else None,
        **loaded.metadata(),
        "library_versions": model_mod.library_versions(),
        "split": split,
        "split_path": str(split_path),
        "test_sha256": test_sha256,
        "labels": labels,
        "examples": n,
        "strict_accuracy": strict_accuracy,
        "semantic_accuracy": semantic_accuracy,
        "valid_output_rate": valid_output_rate,
        "strict_correct": strict_correct,
        "semantic_correct": semantic_correct,
        "valid_predictions": valid_predictions,
        "per_class_recall": per_class_recall,
        "per_class_support": per_class_support,
        "confusion": confusion_matrix,
        "predicted_counts": predicted_counts,
        "mean_input_tokens": mean_input_tokens,
        "median_input_tokens": median_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "output_tokens_per_second": output_tokens_per_second,
        "mean_latency_ms": mean_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "issues_per_second": issues_per_second,
        "total_generation_time_s": total_generation_time,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }

    metrics_path = workspace / f"{phase}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    predictions_path = workspace / f"{phase}_predictions.csv"
    fieldnames = BASELINE_FIELDS if mode == "base" else FINETUNED_FIELDS
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print(f"{phase.upper()} RESULTS")
    print("=" * 60)
    print(f"Base model         : {base_model}")
    print(f"Adapter            : {target if mode == 'adapter' else '-'}")
    print(f"Examples           : {n}")
    print(f"Strict accuracy    : {strict_accuracy:.4f}")
    print(f"Semantic accuracy  : {semantic_accuracy:.4f}")
    print(f"Valid outputs      : {valid_output_rate:.4f}")
    print(f"Peak VRAM allocated: {peak_allocated_gb:.2f} GB")
    print(f"Peak VRAM reserved : {peak_reserved_gb:.2f} GB")
    print(f"Mean latency       : {metrics['mean_latency_ms']:.2f} ms/issue")
    print(f"Median latency     : {metrics['median_latency_ms']:.2f} ms/issue")
    print(f"Issues/sec         : {issues_per_second:.2f}")
    print(f"Saved: {metrics_path.name}, {predictions_path.name}")

    return metrics
