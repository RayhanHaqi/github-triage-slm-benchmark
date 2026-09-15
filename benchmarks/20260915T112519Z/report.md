# Small Language Model Benchmark - GitHub Issue Triage (frozen VS Code split)

Run ID: `20260915T112519Z` (created 2026-09-15T11:25:19Z; last model finished 2026-09-15T18:14:34Z; all times UTC).
Two-class task: classify a GitHub issue as `bug` or `feature-request`. Each model is evaluated
twice on the same fixed test split: **baseline** (base checkpoint, no adapter) and **fine-tuned**
(same base, BF16, with a LoRA adapter trained on the frozen train split). Adapters are deleted
after evaluation; predictions are kept as `baseline_predictions.csv` / `finetuned_predictions.csv`.
Numbers below are derived from `comparison.json` / `*_metrics.json` / `run_info.json` per model;
`results.csv` has the same values in long form (one row per model+mode, 12 rows).

## Data, split, and reproducibility pins

- Dataset source: `/home/tilakoid/github-triage-data` (microsoft/vscode issues; labels `bug`, `feature-request`).
- Split: **temporal** by `created_at`, seed **42**, 80/10/10 -> **1593 train / 200 val / 200 test** examples; test is balanced (100 bug / 100 feature-request).
- Split file hashes (SHA-256):
  - train `3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b`
  - val `7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348`
  - test `9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025`
- **Test SHA match:** for every model, `comparison.json` records identical test hashes for the baseline and fine-tuned runs (`test_sha256.match = true`), so all 12 evaluations share one test set.
- Decoding: greedy (`do_sample: false`), `max_new_tokens: 12`, `max_user_tokens: 1800`, `max_seq_length: 2048`; Qwen-family runs disable thinking (`enable_thinking: false`).

## Metric definitions

- **Strict accuracy**: decoded output, stripped and lowercased, is exactly one of the two class names.
- **Semantic accuracy**: output parses to exactly one class via a forgiving word-boundary regex; outputs that mention both or neither class count as INVALID.
- **Valid output rate**: share of outputs that parse to exactly one class (strict accuracy can be lower than semantic accuracy when output contains extra text).
- **Confusion / per-class recall / support**: computed on the parsed prediction (`INVALID` when unparsable); support is 100 per class for every run.
- **Mean / median latency (ms)**: wall time of the `model.generate` call only (`time.perf_counter` around it); prompt construction and tokenization are excluded.
- **Issues/s**: 200 test issues divided by the sum of those per-issue `model.generate` latencies.
- **Output tokens/s**: total generated tokens divided by total generation time - an **end-to-end generation throughput including prefill** (prompt processing), **not** pure decode throughput.
- **Total generation time (s)**: sum of the 200 per-issue `model.generate` latencies.
- **Peak allocated / reserved VRAM**: `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` over the whole eval phase, divided by 1024^3 and reported in **GiB** (JSON field names say `_gb`, values are GiB).

## Model revisions

| Model | Official checkpoint | Pinned / resolved revision | Kind |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | vision |
| LFM2.5-1.2B-Instruct | `LiquidAI/LFM2.5-1.2B-Instruct` | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` | language |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | language |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | vision |
| Ministral-3-3B-Instruct-2512-BF16 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `b6d637bef2393152b3da2b2fde72eecdee30557e` | vision |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | vision |

All `baseline_metrics.json` / `finetuned_metrics.json` files report `requested_revision == resolved_revision == pinned revision` (source `model_config`).

## Accuracy and per-class results

Deltas are adapter minus base, in percentage points (pp).

| Model | Mode | Strict acc | Semantic acc | Valid output | Recall bug | Recall feature-req | Delta strict (pp) | Delta semantic (pp) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | base | 0.780 | 0.780 | 1.000 | 0.68 | 0.88 | - | - |
| Qwen3.5-0.8B | adapter | 0.805 | 0.905 | 1.000 | 0.92 | 0.89 | +2.5 | +12.5 |
| LFM2.5-1.2B-Instruct | base | 0.625 | 0.645 | 0.995 | 0.37 | 0.92 | - | - |
| LFM2.5-1.2B-Instruct | adapter | 0.900 | 0.900 | 1.000 | 0.92 | 0.88 | +27.5 | +25.5 |
| Qwen3-1.7B | base | 0.580 | 0.580 | 1.000 | 0.18 | 0.98 | - | - |
| Qwen3-1.7B | adapter | 0.890 | 0.890 | 1.000 | 0.92 | 0.86 | +31.0 | +31.0 |
| Qwen3.5-2B | base | 0.840 | 0.840 | 1.000 | 0.81 | 0.87 | - | - |
| Qwen3.5-2B | adapter | 0.890 | 0.890 | 1.000 | 0.90 | 0.88 | +5.0 | +5.0 |
| Ministral-3-3B-Instruct-2512-BF16 | base | 0.610 | 0.610 | 1.000 | 0.28 | 0.94 | - | - |
| Ministral-3-3B-Instruct-2512-BF16 | adapter | 0.875 | 0.875 | 1.000 | 0.88 | 0.87 | +26.5 | +26.5 |
| Qwen3.5-4B | base | 0.885 | 0.885 | 1.000 | 0.88 | 0.89 | - | - |
| Qwen3.5-4B | adapter | 0.850 | 0.895 | 1.000 | 0.90 | 0.89 | -3.5 | +1.0 |

Support is 100 per class (200 total) in every row. Confusion counts (test n=200), parsed-prediction view:

| Model | Mode | bug->bug | bug->feat | bug->INV | feat->bug | feat->feat | feat->INV |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | base | 68 | 32 | 0 | 12 | 88 | 0 |
| Qwen3.5-0.8B | adapter | 92 | 8 | 0 | 11 | 89 | 0 |
| LFM2.5-1.2B-Instruct | base | 37 | 62 | 1 | 8 | 92 | 0 |
| LFM2.5-1.2B-Instruct | adapter | 92 | 8 | 0 | 12 | 88 | 0 |
| Qwen3-1.7B | base | 18 | 82 | 0 | 2 | 98 | 0 |
| Qwen3-1.7B | adapter | 92 | 8 | 0 | 14 | 86 | 0 |
| Qwen3.5-2B | base | 81 | 19 | 0 | 13 | 87 | 0 |
| Qwen3.5-2B | adapter | 90 | 10 | 0 | 12 | 88 | 0 |
| Ministral-3-3B-Instruct-2512-BF16 | base | 28 | 72 | 0 | 6 | 94 | 0 |
| Ministral-3-3B-Instruct-2512-BF16 | adapter | 88 | 12 | 0 | 13 | 87 | 0 |
| Qwen3.5-4B | base | 88 | 12 | 0 | 11 | 89 | 0 |
| Qwen3.5-4B | adapter | 90 | 10 | 0 | 11 | 89 | 0 |

## Generation performance

| Model | Mode | Mean latency (ms) | Median latency (ms) | Issues/s | Output tokens/s | Total generation time (s) | Total generated tokens | Mean input tokens | Median input tokens |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | base | 130.2 | 121.5 | 7.7 | 35.3 | 26.0 | 920 | 382.9 | 236.5 |
| Qwen3.5-0.8B | adapter | 197.4 | 171.2 | 5.1 | 27.1 | 39.5 | 1071 | 382.9 | 236.5 |
| LFM2.5-1.2B-Instruct | base | 47.3 | 37.2 | 21.2 | 75.5 | 9.5 | 714 | 382.2 | 234.5 |
| LFM2.5-1.2B-Instruct | adapter | 56.1 | 46.5 | 17.8 | 52.8 | 11.2 | 592 | 382.2 | 234.5 |
| Qwen3-1.7B | base | 66.5 | 52.4 | 15.0 | 43.6 | 13.3 | 580 | 375.4 | 230.0 |
| Qwen3-1.7B | adapter | 91.6 | 76.4 | 10.9 | 27.0 | 18.3 | 494 | 375.4 | 230.0 |
| Qwen3.5-2B | base | 146.9 | 130.6 | 6.8 | 30.8 | 29.4 | 906 | 382.9 | 236.5 |
| Qwen3.5-2B | adapter | 195.3 | 177.5 | 5.1 | 23.0 | 39.1 | 898 | 382.9 | 236.5 |
| Ministral-3-3B-Instruct-2512-BF16 | base | 113.2 | 82.7 | 8.8 | 25.1 | 22.6 | 568 | 372.6 | 220.5 |
| Ministral-3-3B-Instruct-2512-BF16 | adapter | 132.1 | 99.1 | 7.6 | 18.9 | 26.4 | 499 | 372.6 | 220.5 |
| Qwen3.5-4B | base | 259.6 | 213.9 | 3.9 | 17.4 | 51.9 | 901 | 382.9 | 236.5 |
| Qwen3.5-4B | adapter | 325.3 | 252.7 | 3.1 | 14.8 | 65.1 | 966 | 382.9 | 236.5 |

Reminder: latency covers `model.generate` only; output tokens/s is end-to-end generation throughput including prefill (prompt processing), not pure decode throughput. Input-token counts are tokenized prompt lengths (chat template applied, user content truncated to 1800 tokens).

## Peak VRAM (GiB)

| Model | Mode | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---|---|
| Qwen3.5-0.8B | base | 1.80 | 1.90 |
| Qwen3.5-0.8B | adapter | 1.85 | 1.96 |
| LFM2.5-1.2B-Instruct | base | 2.32 | 2.46 |
| LFM2.5-1.2B-Instruct | adapter | 2.46 | 2.59 |
| Qwen3-1.7B | base | 3.50 | 3.87 |
| Qwen3-1.7B | adapter | 3.64 | 3.94 |
| Qwen3.5-2B | base | 4.34 | 4.45 |
| Qwen3.5-2B | adapter | 4.43 | 4.54 |
| Ministral-3-3B-Instruct-2512-BF16 | base | 7.50 | 7.65 |
| Ministral-3-3B-Instruct-2512-BF16 | adapter | 7.74 | 7.95 |
| Qwen3.5-4B | base | 8.92 | 9.29 |
| Qwen3.5-4B | adapter | 9.05 | 9.36 |

These are PyTorch allocator peaks for the evaluation process (base weights + adapter where applicable + activations/KV cache), not whole-GPU totals.

## Training runs (successful attempts only)

Shared recipe: LoRA on full-precision BF16 base weights, **no quantization** (`load_in_4bit: false`), 3 epochs, per-device batch 2 x gradient accumulation 4 (effective batch 8), lr 2e-4 cosine, warmup 0.05, weight decay 0.01, `adamw_8bit`, Unsloth gradient checkpointing, LoRA r=16 / alpha=32 / dropout 0 / bias none, seed 42, `max_seq_length` 2048. Vision towers are frozen; language/attention/MLP linear layers are adapted (LFM2.5 uses its native attention/conv/MLP target list). `Trainer runtime` is the trainer-reported `train_runtime`; `Train wall` is the training-stage wall time from `train_metrics.json`.

| Model | Epochs | Train loss | Final eval loss | Trainer runtime (s) | Train wall (s) | Train phase wall (s) | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 3.0 | 1.50 | 1.74 | 911.7 | 912.9 | 965.7 | attempt 1 failed early (TMPDIR AF_UNIX path; excluded) |
| LFM2.5-1.2B-Instruct | 3.0 | 1.71 | 1.78 | 526.8 | 527.9 | 580.2 | attempt 1 failed early (TMPDIR AF_UNIX path; excluded) |
| Qwen3-1.7B | 3.0 | 1.62 | 1.71 | 748.7 | 749.7 | 831.6 | attempt 1 failed early (TMPDIR AF_UNIX path; excluded) |
| Qwen3.5-2B | 3.0 | 0.38 | 1.63 | 481.0 | 482.3 | 515.0 | attempt 1 failed early (TMPDIR AF_UNIX path); run interrupted externally, resumed from checkpoint-400; metrics cover the resumed segment (steps 400-600) only |
| Ministral-3-3B-Instruct-2512-BF16 | 3.0 | 1.32 | 1.67 | 2504.2 | 2505.5 | 2608.0 | no failed attempts |
| Qwen3.5-4B | 3.0 | 1.25 | 1.51 | 3098.7 | 3099.9 | 3138.2 | attempt 1 failed early (TMPDIR AF_UNIX path); attempts 2-3 failed on epoch-eval OOM; values cover the final successful attempt only (loss-only prediction_step fix) |

`Train wall (s)` from `train_metrics.json` is measured around `trainer.train()` and stops immediately after training returns, so it **excludes adapter saving**; it can still exceed the trainer-reported runtime because it also covers in-loop epoch evaluation and checkpoint saves. The phase wall from `run_info.json` is the whole training phase and additionally includes model load/setup, epoch evaluation, checkpoint handling, and adapter save as applicable. Failed-attempt time is never included in these successful-run figures.

### Qwen3.5-4B: OOM history and fix

The two middle attempts failed during epoch evaluation with CUDA OOM: Unsloth forced full-vocabulary eval logits (2 x 2048 x 248320) and Accelerate then attempted a 3.79 GiB FP32 copy of them; `SFTConfig(prediction_loss_only=True)` was ignored by the Unsloth path. The fix was a local **loss-only `prediction_step`** that bypasses the forced eval logits; it was validated by a **2 x 2048 GPU preflight** (`preflight_eval.json`: status `pass`, `per_device_eval_batch_size: 2`, `effective_max_length: 2048`, two examples with raw lengths 3935 and 3497 tokens, eval_loss 9.681, 31.4 s, peak 9.30 GiB allocated / 9.38 GiB reserved). The successful training then completed in 3098.7 s trainer runtime (3099.9 s wall). The failed-attempt times (31.3 s + 1175.6 s + 1158.2 s) are **not** part of the successful training runtime.

### Qwen3.5-2B: external interruption

The original training process was terminated externally (no model/training exception in the log) after checkpoint-400. Training resumed from `checkpoint-400` at step 400 and completed at step 600; the training metrics and reported runtime (481.0 s trainer / 482.3 s wall, 515.0 s phase) correspond to the resumed segment, steps 400-600 only, not the interrupted segment. This scope is encoded as `training_metric_scope = "resumed steps 400-600 only"` on both Qwen3.5-2B rows in `results.csv`.

## Environment

- Python 3.11.16; PyTorch 2.11.0+cu128; transformers 5.5.0; peft 0.20.0; datasets 4.3.0; trl 0.24.0; unsloth 2026.9.4; unsloth_zoo 2026.9.3; CUDA 12.8.
- GPU: 1x NVIDIA GeForce RTX 5060 Ti (16 GB class; 15.46 GiB reported usable by the CUDA OOM message). All 12 evaluations ran on this single GPU.

## Model-specific notes

| Model | Compatibility / method notes |
|---|---|
| Qwen3.5-0.8B | FastVisionModel; official repo forced with use_exact_model_name; language/attention/MLP LoRA; vision frozen; BF16. |
| LFM2.5-1.2B-Instruct | FastLanguageModel; official repo forced; native LFM2 attention/conv/MLP targets; BF16. |
| Qwen3-1.7B | Reference recipe; official repo forced; thinking disabled; BF16. |
| Qwen3.5-2B | FastVisionModel; official repo forced; language/attention/MLP LoRA; vision frozen; BF16. |
| Ministral-3-3B-Instruct-2512-BF16 | Official BF16 checkpoint forced (not FP8/re-upload); FastVisionModel; vision frozen; vision collator; BF16. |
| Qwen3.5-4B | FastVisionModel; official repo forced; language/attention/MLP LoRA; vision frozen; BF16; reference batch recipe. Local loss-only prediction_step bypasses Unsloth forced eval logits; 2x2048 GPU preflight passed. |

## Artifacts

- `results.csv`: 12 data rows (6 models x 2 modes); scalar columns plus JSON-encoded `per_class_recall`, `per_class_support`, `confusion`, `predicted_counts`; shared reproducibility fields (split hashes, split sizes/method/seed, precision, LoRA recipe, training hyperparameters, library/CUDA/GPU versions) repeated on every row. Training metrics carry a `training_metric_scope` column (`full successful run` for all runs except Qwen3.5-2B, `resumed steps 400-600 only` for both of its rows); `finetune_vision_layers` / `finetune_language_layers` are `N/A (kwargs not passed)` for the two language models.
- Per model: `baseline_predictions.csv` is intentionally 5 columns (`issue_number`, `target`, `prediction`, `strict_prediction`, `raw_output`); `finetuned_predictions.csv` contains those 5 columns plus per-issue `input_tokens`, `output_tokens`, `latency_ms` (8 columns total). Per-model metrics/manifests: `baseline_metrics.json`, `finetuned_metrics.json`, `comparison.json`, `train_metrics.json`, `run_info.json`.
- No rankings are implied; values are reported as measured. Training/eval artifacts (base cache, adapters, trainer checkpoints) were deleted after each run by policy; predictions and metrics remain.
