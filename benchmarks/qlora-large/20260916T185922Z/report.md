# QLoRA-large Benchmark - GitHub Issue Triage (frozen VS Code split)

Run ID: `20260916T185922Z` - manifest status `completed`; created 2026-09-16T18:59:22Z, last
model cleaned 2026-09-17T00:25:18Z (2026-09-17 08:25 UTC+8). All times UTC. Track: **QLoRA
NF4 only** (4-bit base weights, bf16 compute); no BF16 fine-tuning in this track. Each model
is evaluated twice on the same frozen 200-row test split: **baseline** (base checkpoint, no
adapter) and **fine-tuned** (same base + LoRA adapter trained on the frozen 1593-row train
split). Numbers below are derived from `manifest.json`, per-model `*_metrics.json` /
`comparison.json` / `train_metrics.json` / `run_info.json` / `acceptance.json` /
`cleanup.json`; `results.csv` carries the same values as 8 long-form rows.

**Execution provenance (read first).** The manifest `git_commit` - the continuation's
execution checkout - is `b0f91ad46f8491bbe6bba9a453c761cb92efb281`. Models 2-4 were executed in this
run under that commit. Model 1 (Qwen3-8B) was **not rerun**: its 12 evidence files were
imported byte-identical from earlier run `20260916T120019Z` (original execution
`385bbb957733e37d31627ff3f67e9931c95e6946`; evidence pinned at
`429d5ccd885ec9188e76ae1d2685e210e5b34e7f`). Qwen's recorded phases ran
2026-09-16T12:00:19Z-13:01:22Z and are **excluded** from this run's interval. Its
`preflight/selection.json` is the pre-fix legacy artifact (no `resolved_revision_source`);
its metrics record `resolved_revision_source = model_config` and the exact pinned revision.
The new-run manifest copies neither `effective_git_commit` nor `compatibility_deviations`.

## Data, split, and reproducibility pins

- Dataset source: `/home/tilakoid/github-triage-data` (microsoft/vscode issues; labels `bug`, `feature-request`).
- Split (recorded in each config snapshot): temporal per-class by `created_at`, seed 42, 80/10/10 -> **1593 train / 200 val / 200 test**; test balanced (100 bug / 100 feature-request).
- Split file SHA-256: train `3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b`, val `7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348`, test `9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025`.
- `comparison.json` records identical test hashes for baseline and fine-tuned in every model (`test_sha256.match = true`), so all 8 evaluations share one test set.
- Decoding: greedy (`do_sample: false`), `max_new_tokens: 12`, issue text truncated at `max_user_tokens: 1800`, `max_seq_length: 2048`; Qwen-family runs disable thinking (`enable_thinking: false`).

## Metric definitions

- **Strict accuracy**: share of outputs whose stripped, lowercased text exactly equals the correct class label.
- **Semantic accuracy**: share of outputs whose single class parsed by the forgiving word-boundary regex equals the correct label; both/neither -> INVALID and incorrect.
- **Valid output rate**: share of outputs that parse to exactly one class.
- **Latency / throughput**: wall time of `model.generate` only (prompt construction/tokenization excluded); issues/s = 200 / sum of per-issue latencies; output tokens/s is end-to-end generation throughput including prefill, not pure decode.
- **VRAM**: all values are GiB = bytes / 1024^3. Evaluation metric JSON field names use a `_gb` suffix and training fields use `_gib`; both are GiB.

## Precision, LoRA recipe, and model revisions

All four models: official repo forced (`use_exact_model_name: true`), quantized on the fly from the official BF16 checkpoint to **NF4** (`effective_load_in_4bit: true`, `quant_type: ["nf4"]`, `double_quant: true`, compute dtype `torch.bfloat16`, no CPU/disk/meta offload, 0 bitsandbytes 8-bit modules, 0 non-CUDA parameters). Quantized module counts: Qwen3-8B 252, Ministral-3-8B 409, Qwen3.5-9B 358, Ministral-3-14B 451. LoRA: r 16 / alpha 32 / dropout 0 / bias none, gradient checkpointing `unsloth`, random_state 42.

| Model | Official checkpoint | Pinned / resolved revision | Kind | Target modules | Batch recipe |
|---|---|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | language | q/k/v/o/gate/up/down proj | 2 x 4 (eff. 8) |
| Ministral-3-8B-Instruct-2512-BF16 | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` | vision | all-linear, `finetune_vision_layers: false`, language/attention/MLP true | 2 x 4 (eff. 8) |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | vision | all-linear, `finetune_vision_layers: false`, language/attention/MLP true | 2 x 4 (eff. 8) |
| Ministral-3-14B-Instruct-2512-BF16 | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | vision | all-linear, `finetune_vision_layers: false`, language/attention/MLP true | 1 x 8 (eff. 8) |

Shared training hyperparameters (config snapshots): 3 epochs, lr 2e-4 cosine, warmup 0.05, weight decay 0.01, `adamw_8bit`, per-device eval batch 1, `report_to: none`, seed 42.

## Accuracy (n = 200 test issues; deltas are adapter minus base, pp)

| Model | Mode | Strict acc | Semantic acc | Valid output | Delta strict (pp) | Delta semantic (pp) |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.870 | 0.870 | 1.000 | - | - |
| Qwen3-8B | fine-tuned | 0.890 | 0.890 | 1.000 | +2.0 | +2.0 |
| Ministral-3-8B-Instruct-2512-BF16 | base | 0.815 | 0.815 | 1.000 | - | - |
| Ministral-3-8B-Instruct-2512-BF16 | fine-tuned | 0.875 | 0.885 | 0.995 | +6.0 | +7.0 |
| Qwen3.5-9B | base | 0.860 | 0.860 | 1.000 | - | - |
| Qwen3.5-9B | fine-tuned | 0.890 | 0.890 | 1.000 | +3.0 | +3.0 |
| Ministral-3-14B-Instruct-2512-BF16 | base | 0.855 | 0.860 | 1.000 | - | - |
| Ministral-3-14B-Instruct-2512-BF16 | fine-tuned | 0.885 | 0.890 | 1.000 | +3.0 | +3.0 |

Per-class recall (bug / feature-request): Qwen base 0.91/0.83, FT 0.90/0.88; Ministral-3-8B base 0.93/0.70, FT 0.89/0.88; Qwen3.5-9B base 0.94/0.78, FT 0.89/0.89; Ministral-3-14B base 0.95/0.77, FT 0.89/0.89. Support is 100 per class (200 total) in every row. Confusion counts (parsed-prediction view, `bug->bug / bug->feat / bug->INVALID | feat->bug / feat->feat / feat->INVALID`):

| Model | base | fine-tuned |
|---|---|---|
| Qwen3-8B | 91/9/0 \| 17/83/0 | 90/10/0 \| 12/88/0 |
| Ministral-3-8B | 93/7/0 \| 30/70/0 | 89/10/1 \| 12/88/0 |
| Qwen3.5-9B | 94/6/0 \| 22/78/0 | 89/11/0 \| 11/89/0 |
| Ministral-3-14B | 95/5/0 \| 23/77/0 | 89/11/0 \| 11/89/0 |

## Generation performance (`model.generate` only)

| Model | Mode | Mean latency (ms) | Median latency (ms) | Issues/s | Output tokens/s | Total generation time (s) | Total generated tokens | Mean / median input tokens |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 235.144 | 191.640 | 4.253 | 10.462 | 47.029 | 492 | 375.41 / 230.0 |
| Qwen3-8B | fine-tuned | 304.451 | 245.585 | 3.285 | 8.179 | 60.890 | 498 | 375.41 / 230.0 |
| Ministral-3-8B | base | 257.182 | 196.595 | 3.888 | 9.274 | 51.436 | 477 | 372.58 / 220.5 |
| Ministral-3-8B | fine-tuned | 314.600 | 240.187 | 3.179 | 8.201 | 62.920 | 516 | 372.58 / 220.5 |
| Qwen3.5-9B | base | 363.344 | 302.240 | 2.752 | 12.165 | 72.669 | 884 | 382.85 / 236.5 |
| Qwen3.5-9B | fine-tuned | 476.188 | 402.770 | 2.100 | 9.450 | 95.238 | 900 | 382.85 / 236.5 |
| Ministral-3-14B | base | 372.069 | 297.555 | 2.688 | 6.504 | 74.414 | 484 | 372.58 / 220.5 |
| Ministral-3-14B | fine-tuned | 466.129 | 365.839 | 2.145 | 5.471 | 93.226 | 510 | 372.58 / 220.5 |

## Peak VRAM (GiB, PyTorch allocator, 1024^3)

Training process (setup / training peaks) and evaluation process (base load + generation, adapter load + generation). These are process-level allocator peaks, not whole-GPU totals.

| Model | Setup peak (alloc / res) | Train peak (alloc / res) | Base load (alloc / res) | Base generation (alloc / res) | Adapter load (alloc / res) | Adapter generation (alloc / res) |
|---|---|---|---|---|---|---|
| Qwen3-8B | 5.863 / 5.932 | 7.128 / 7.281 | 5.749 / 5.779 | 6.173 / 6.385 | 5.995 / 6.100 | 6.431 / 6.691 |
| Ministral-3-8B | 5.955 / 5.996 | 7.191 / 7.338 | 5.785 / 5.818 | 6.304 / 6.490 | 6.114 / 6.150 | 6.572 / 6.926 |
| Qwen3.5-9B | 7.516 / 7.543 | 8.825 / 9.025 | 7.379 / 7.404 | 7.820 / 7.963 | 7.671 / 7.727 | 7.979 / 8.178 |
| Ministral-3-14B | 9.200 / 9.211 | 9.810 / 9.939 | 9.190 / 9.201 | 9.211 / 9.525 | 9.190 / 9.666 | 9.521 / 9.896 |

Acceptance process peaks (max over the phase, alloc / res GiB): Qwen3-8B 7.128 / 7.281; Ministral-3-8B 7.191 / 7.338; Qwen3.5-9B 8.825 / 9.025; Ministral-3-14B 9.810 / 9.939.

## Training runs

| Model | Epochs | Train loss | Best val loss (epoch) | Final eval loss (epoch 3) | Trainer runtime (s) | Train wall (s) | Train phase wall (s) | Attempts |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (imported) | 3.0 | 1.374 | 1.582 (2) | 1.608 | 3184.27 | 3185.42 | 3235.51 | 1 per phase |
| Ministral-3-8B | 3.0 | 1.268 | 1.739 (2) | 1.804 | 5423.04 | 5424.51 | 5468.11 | 1 per phase |
| Qwen3.5-9B | 3.0 | 1.218 | 1.480 (2) | 1.519 | 5236.63 | 5237.85 | 5282.66 | 1 per phase |
| Ministral-3-14B | 3.0 | 1.208 | 1.689 (2) | 1.768 | 6530.78 | 6532.06 | 6577.00 | 1 per phase |

Best checkpoint (epoch 2, lowest validation loss) was restored before fine-tuned evaluation in all four runs. `Train wall` is measured around `trainer.train()`, including epoch evaluation and checkpoint handling; the phase wall additionally covers model load/setup, adapter save, and final metadata work. Qwen's row is the imported source-run training; its phase wall is not part of this run's interval.

## GPU sharing policy (models 2-4)

Models 2-4 ran under the recorded one-time user-approved policy `rustdesk_shared_gpu_v1`: only the exact `/usr/share/rustdesk/rustdesk` executable was allowed to hold GPU memory, aggregate allowance 512 MiB, and VRAM headroom was judged against an effective capacity = min(torch-visible total, observed free memory - remaining RustDesk allowance). Budgets were point-in-time phase-boundary samples (pre/post each GPU subprocess plus a fresh acceptance sample; 13 retained per model), not continuous monitoring, and no instantaneous safety guarantee is claimed. Retained minimum effective capacity and maximum observed RustDesk usage per model: Ministral-3-8B 12.243 GiB / 257 MiB (RustDesk was observed at pid 1017672 using 257 MiB during early phases); Qwen3.5-9B 12.445 GiB / 0 MiB; Ministral-3-14B 12.473 GiB / 0 MiB. Every accepted reserved peak stayed more than 1 GiB below its retained budget (7.338, 9.025, 9.939 GiB reserved respectively). Cleanup recorded no foreign processes at deletion time in any model and killed nothing.

Qwen3-8B was measured **without** GPU sharing (imported evidence), so its timing/VRAM numbers are not directly comparable to models 2-4. All timing and VRAM values in this report are single-run descriptive measurements, not a clean efficiency ranking.

## Frozen vision note (this run only)

For the three vision models, the accepted training metrics list trainable parameter names with **zero** matches for vision-like tokens (`vision`, `visual`, `image`, `pixel`, `patch`, multimodal/projector tokens), consistent with the recorded `finetune_vision_layers: false`; i.e. the vision towers stayed frozen in this QLoRA run. This statement applies only to this run and does not validate or rewrite any earlier BF16 vision run.

Historical BF16 vision runs may have been affected by Unsloth's `all-linear` override of the freeze flags; their frozen-vision behavior is not established. That historical track remains unchanged.

## Completion, cleanup, and failure history

| Model | Evidence | First phase start | Last phase end | Acceptance | Cleanup | Cleanup deleted |
|---|---|---|---|---|---|---|
| Qwen3-8B | imported (source run `20260916T120019Z`) | 2026-09-16T12:00:19Z | 2026-09-16T13:01:22Z | ok | ok | 17144991844 B (15.97 GiB) |
| Ministral-3-8B | this run | 2026-09-16T19:00:50Z | 2026-09-16T20:42:36Z | ok | ok | 36512854314 B (34.01 GiB) |
| Qwen3.5-9B | this run | 2026-09-16T20:42:36Z | 2026-09-16T22:20:33Z | ok | ok | 20200318260 B (18.81 GiB) |
| Ministral-3-14B | this run | 2026-09-16T22:20:33Z | 2026-09-17T00:25:18Z | ok | ok | 56889710014 B (52.98 GiB) |

The source run `20260916T120019Z` retains the two old Ministral-3-8B preflight failures (attempt 1: selection exit 1; attempt 2: 336 vision-like trainable parameters). Those failures exist only in that source run's history; this run neither rewrites nor copies them - every phase in this run was attempt 1 with no retries, and Qwen's history was imported as-is. Track `LATEST` points to `20260916T185922Z`.

## Limitations

- Test split is a 200-row holdout. Differences of a few points are descriptive only; no statistical significance, general quality, or "bigger model is better" claim can be drawn from them.
- Single-run measurements; adapters/checkpoints were deleted after evaluation per policy, so training metrics are the retained record.
- This QLoRA track is not merged or ranked against the historical BF16 track, and no comparison to it is implied.

## Environment

Python 3.11.16; PyTorch 2.11.0+cu128; transformers 5.5.0; peft 0.20.0; bitsandbytes 0.50.2; datasets 4.3.0; trl 0.24.0; unsloth 2026.9.4; unsloth_zoo 2026.9.3; accelerate 1.15.0; CUDA 12.8; GPU 1x NVIDIA GeForce RTX 5060 Ti (torch-visible total 15.456 GiB).

## Artifacts

- `results.csv`: 8 data rows (4 models x baseline/fine-tuned) with scalar columns, JSON-encoded per-class recall/support/confusion/predicted counts, provenance, split hashes, precision/quantization fields, LoRA/training hyperparameters, library/GPU versions, GPU-sharing policy fields, acceptance/cleanup status.
- Per model: `run_info.json` (shared-GPU samples for models 2-4), `acceptance.json`, `cleanup.json`, `config.yaml`, `preflight/{selection,preflight}.json`; `workspace/` contains `{baseline,finetuned}_metrics.json`, `{baseline,finetuned}_predictions.csv`, `comparison.json`, and `train_metrics.json`.
- Adapters, trainer checkpoints, copied split files and scratch caches were deleted at cleanup (per-model `cleanup.json` records the targets and bytes); `workspace/unsloth_compiled_cache/` is retained for the three vision models as produced.
