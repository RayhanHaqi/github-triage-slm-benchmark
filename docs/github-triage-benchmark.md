# Fine-tuning SLMs for GitHub issue triage: a case-focused benchmark analysis

Two frozen runs fine-tune small language models (SLMs) to classify one `microsoft/vscode` GitHub
issue as `bug` or `feature-request`, then compare each model's outputs before (base checkpoint)
and after (LoRA adapter) fine-tuning on the **same 200-issue test set**. This document is a
case-focused reading of the recorded outputs and metrics: every number below is derived
mechanically from the frozen CSVs/JSON under `benchmarks/`, the two runs are kept separate, and
there is no merged leaderboard.

- **BF16 LoRA track** — `benchmarks/20260915T112519Z/` — six configurations, 0.8B–4B nominal.
- **QLoRA NF4 track** — `benchmarks/qlora-large/20260916T185922Z/` — four configurations, 8B–14B nominal.

## 1. Scope and protocol

- **Task.** Two-class classification of one VS Code issue: `bug` vs `feature-request`. Input is
  issue text only; the model must return one class name. All runs are **text-only** even where the
  loader kind is `vision` (vision-capable checkpoint); no images are used.
- **Frozen data.** Dataset source `/home/tilakoid/github-triage-data` (not committed; see
  limitations). Temporal split by `created_at`, seed 42, 80/10/10 → **1593 train / 200 val / 200
  test**; the test set is balanced 100 `bug` / 100 `feature-request`. Split SHA-256 is identical
  in both runs: train `3d58b700…6202b`, val `7423f47f…7348`, test `9fc58e70…12025` (full values in
  each `results.csv`). `comparison.json` records `test_sha256.match = true` for every model, so all
  20 evaluations share one test set.
- **Decoding (common).** One system prompt — *"Classify the GitHub issue into exactly one category:
  bug or feature-request. Return only the category name."* — with greedy decoding
  (`do_sample: false`), `max_new_tokens: 12`, issue text truncated at `max_user_tokens: 1800`,
  `max_seq_length: 2048`, `pad_eos: true`; Qwen-family runs disable thinking
  (`enable_thinking: false`). Single pass over the 200 issues per evaluation.
- **Hardware.** One NVIDIA GeForce RTX 5060 Ti (torch-visible total 15.456 GiB in the QLoRA
  manifest; the BF16 report notes ~15.46 GiB usable). No multi-GPU.
- **Recipe.** LoRA r=16 / alpha=32 / dropout 0 / bias none, 3 epochs, lr 2e-4 cosine, warmup 0.05,
  weight decay 0.01, `adamw_8bit`, gradient checkpointing `unsloth`, seed 42, `max_seq_length`
  2048, **effective batch 8**. BF16 track: BF16 base weights, no quantization (`load_in_4bit:
  false`), per-device 2 × accumulation 4. QLoRA track: official BF16 checkpoint quantized on the
  fly to **NF4** (double quantized, bf16 compute, no offload), per-device 2 × accumulation 4 except
  Ministral-3-14B at 1 × 8.
- **Selected checkpoint policy (differs per track).** BF16: the adapter saved at the end of
  training (epoch 3, step 600); no best-checkpoint selection is recorded (`train_metrics.json` has
  no `best_epoch`/`best_model_checkpoint`). QLoRA: the lowest-validation-loss checkpoint (epoch 2,
  step 400) was restored before fine-tuned evaluation in all four runs (`best_epoch: 2.0`,
  `best_model_checkpoint: …/checkpoint-400`, `load_best_model_at_end` in the QLoRA runner).
- **Sizes are nominal.** "0.8B" … "14B" are taken from the reported official model names, not
  exact parameter counts.

## 2. Limitations (read first)

1. **Not a controlled size or precision experiment.** The tracks differ in quantization (BF16 vs
   NF4), checkpoint selection (final epoch 3 vs best epoch 2), model families, run dates/commits,
   and GPU-sharing conditions. Do not read Table 1 vs Table 2 as a size or precision effect; the
   ten configurations have no merged ranking.
2. **BF16 vision freeze is not established.** BF16 configs declare `finetune_vision_layers: false`
   but also `target_modules: all-linear`, and no BF16 artifact lists adapted parameter names. The
   QLoRA report ("Frozen vision note", `benchmarks/qlora-large/20260916T185922Z/report.md`) states
   that historical BF16 vision runs may have been affected by Unsloth's all-linear override and
   their frozen-vision behavior is not established. Only the QLoRA run records evidence: its three
   vision models list **zero** vision-like trainable tensors (`vision_like_count: 0`,
   `train_metrics.json`).
3. **Different checkpoint selection.** Values are not "best-vs-best" across tracks (see §1): BF16
   used the final epoch-3 adapter, QLoRA the restored epoch-2 best-validation checkpoint.
4. **Imported Qwen3-8B (QLoRA).** It was **not rerun**: 12 evidence files were imported
   byte-identical from `benchmarks/qlora-large/20260916T120019Z` (original execution commit
   `385bbb957733e37d31627ff3f67e9931c95e6946`, evidence pinned at
   `429d5ccd885ec9188e76ae1d2685e210e5b34e7f`). The other three QLoRA models executed under this
   run's manifest commit `b0f91ad46f8491bbe6bba9a453c761cb92efb281`. Qwen's timing/VRAM was
   measured **without** GPU sharing.
5. **GPU sharing differs.** QLoRA models 2–4 ran under the recorded `rustdesk_shared_gpu_v1`
   policy: only the exact `/usr/share/rustdesk/rustdesk` executable allowed, 512 MiB aggregate
   allowance, and budget samples taken at **phase boundaries** (13 retained per model), not
   continuous monitoring. Retained minimum effective capacity 12.243 / 12.445 / 12.473 GiB; the
   only nonzero observed RustDesk usage was **257 MiB** on Ministral-3-8B (0 MiB elsewhere).
   Timing and VRAM are descriptive, not a clean efficiency ranking.
6. **200-row single pass.** No confidence intervals, significance tests, or causal claims. A few
   points of difference are descriptive only; "bigger model is better" cannot be drawn from this.
7. **Deleted artifacts and uncommitted data.** Adapters, trainer checkpoints and caches were deleted
   after each run (QLoRA `cleanup.json` records the bytes; BF16 `run_info.json` weights policy).
   The dataset lives outside the repo and is not committed (JSONL/raw are gitignored), so exact
   reproduction requires it. Operational steps: `docs/reproduction.md`.
8. **Training-metric scopes and omitted BF16 trainable counts.** BF16 Qwen3.5-2B training metrics
   cover the resumed segment, steps 400–600 only; BF16 Qwen3.5-4B values cover the final
   successful attempt (two epoch-eval OOM attempts excluded from runtime); QLoRA Qwen3-8B
   training rows are the imported source-run metrics. Actual trainable-parameter counts were not
   recorded in the committed BF16 metrics (no such column in `results.csv`), so Table 1 omits
   them; the nominal sizes are not exact parameter counts.

## 3. Artifacts and metric definitions

Artifacts used (relative to the repo root; frozen, read-only):

| Path | Contents |
|---|---|
| [`benchmarks/20260915T112519Z/report.md`](../benchmarks/20260915T112519Z/report.md) | BF16 track narrative and definitions |
| [`benchmarks/20260915T112519Z/results.csv`](../benchmarks/20260915T112519Z/results.csv) | 12 long-form rows (6 configs × base/adapter) |
| [`benchmarks/20260915T112519Z/manifest.json`](../benchmarks/20260915T112519Z/manifest.json) | run status, model revisions |
| `benchmarks/20260915T112519Z/models/<slug>/{baseline,finetuned}_predictions.csv` | 200 rows each: `issue_number,target,prediction,strict_prediction,raw_output` (+ `input_tokens,output_tokens,latency_ms` when fine-tuned) |
| `benchmarks/20260915T112519Z/models/<slug>/{baseline,finetuned}_metrics.json`, `comparison.json`, `train_metrics.json` | per-run metrics and training record |
| [`benchmarks/qlora-large/20260916T185922Z/report.md`](../benchmarks/qlora-large/20260916T185922Z/report.md) | QLoRA track narrative and definitions |
| [`benchmarks/qlora-large/20260916T185922Z/results.csv`](../benchmarks/qlora-large/20260916T185922Z/results.csv) | 8 long-form rows (4 configs × base/adapter) |
| [`benchmarks/qlora-large/20260916T185922Z/manifest.json`](../benchmarks/qlora-large/20260916T185922Z/manifest.json) | run status, revisions, import provenance |
| `benchmarks/qlora-large/20260916T185922Z/models/<slug>/workspace/{baseline,finetuned}_predictions.csv` | 200 rows each, same columns |
| `benchmarks/qlora-large/20260916T185922Z/models/<slug>/workspace/*_metrics.json`, `comparison.json`, `train_metrics.json` | per-run metrics and training record |
| [`src/specialist/model.py`](../src/specialist/model.py) (`strict_prediction`, `parse_prediction`) and [`src/specialist/evaluate.py`](../src/specialist/evaluate.py) | the frozen parsers and evaluation loop that produced the columns |

Metric definitions used in the tables and cases:

- **Gold label** — the `target` column (one of `bug`, `feature-request`; 100 each).
- **Strict accuracy** — the raw output, stripped and lowercased, must equal the gold class name
  exactly (case-insensitive whole-string match). Anything else fails, including `bug.`,
  `bug\nuser…`, or a leaked template fragment. This is the metric behind the `strict_prediction`
  column.
- **Semantic accuracy** — exactly one class parsed from the raw output by a forgiving
  word-boundary regex (`\bbug\b` / `\bfeature[\s-]+request\b`, case-insensitive); both/neither ⇒
  `INVALID` and wrong. This is the `prediction` column.
- **Valid output rate** — share of outputs that parse to exactly one class.
- **Δ strict (pp)** — fine-tuned minus base strict accuracy, in percentage points (can be negative).
- **FT mean latency (ms)** — mean wall time of `model.generate` only over the 200 issues (prompt
  construction/tokenization excluded), from the fine-tuned predictions CSV.
- **Max reserved (GiB)** — PyTorch allocator peak, bytes / 1024³; JSON field names `_gb` / `_gib`
  are both GiB. BF16: the fine-tuned **evaluation** process peak. QLoRA: the maximum over the
  recorded setup / training / evaluation phases — for all four configs the **training** peak is the
  maximum. These are per-process peaks (weights + adapter + activations/KV), not whole-GPU totals.
- **Training (s)** — `trainer_train_runtime_s` / `train_wall_seconds` / `train_phase_wall_seconds`
  from `results.csv` (trainer-reported runtime; wall around `trainer.train()`; whole training phase).

## 4. Checkpoint pins

| Track | Model (official checkpoint) | Pinned revision | Kind |
|---|---|---|---|
| BF16 | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | vision |
| BF16 | `LiquidAI/LFM2.5-1.2B-Instruct` | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` | language |
| BF16 | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | language |
| BF16 | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | vision |
| BF16 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `b6d637bef2393152b3da2b2fde72eecdee30557e` | vision |
| BF16 | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | vision |
| QLoRA | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | language |
| QLoRA | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` | vision |
| QLoRA | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | vision |
| QLoRA | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | vision |

## 5. Table 1 — BF16 LoRA track (six configurations, fixed run order)

All rows: BF16 base weights, no quantization; adapter = final epoch-3 adapter. Rows are in the
original run order (not sorted by accuracy). "Nominal" is from the official model name, not an
exact parameter count. LoRA targets: `all-linear` for the Qwen3.5/Ministral configs, the native
attention/conv/MLP list for LFM2.5, and explicit q/k/v/o/gate/up/down projections for Qwen3-1.7B.
Actual trainable-parameter counts were not recorded in the committed BF16 metrics, so this table
omits them.

| # | Model (official checkpoint) | Family | Kind | Nominal | Precision | Base strict | FT strict | Δ strict (pp) | FT semantic | FT valid | FT mean latency (ms) | Max reserved eval (GiB) | Training (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`) | Qwen3.5 | vision | 0.8B | BF16 no quant | 0.780 | 0.805 | +2.5 | 0.905 | 1.000 | 197.4 | 1.965 | 911.7 / 912.9 / 965.7 |
| 2 | LFM2.5-1.2B-Instruct (`LiquidAI/LFM2.5-1.2B-Instruct`) | LFM2.5 | language | 1.2B | BF16 no quant | 0.625 | 0.900 | +27.5 | 0.900 | 1.000 | 56.1 | 2.592 | 526.8 / 527.9 / 580.2 |
| 3 | Qwen3-1.7B (`Qwen/Qwen3-1.7B`) | Qwen3 | language | 1.7B | BF16 no quant | 0.580 | 0.890 | +31.0 | 0.890 | 1.000 | 91.6 | 3.938 | 748.7 / 749.7 / 831.6 |
| 4 | Qwen3.5-2B (`Qwen/Qwen3.5-2B`) | Qwen3.5 | vision | 2B | BF16 no quant | 0.840 | 0.890 | +5.0 | 0.890 | 1.000 | 195.3 | 4.537 | 481.0 / 482.3 / 515.0 |
| 5 | Ministral-3-3B-Instruct-2512-BF16 (`mistralai/Ministral-3-3B-Instruct-2512-BF16`) | Ministral-3 | vision | 3B | BF16 no quant | 0.610 | 0.875 | +26.5 | 0.875 | 1.000 | 132.1 | 7.951 | 2504.2 / 2505.5 / 2608.0 |
| 6 | Qwen3.5-4B (`Qwen/Qwen3.5-4B`) | Qwen3.5 | vision | 4B | BF16 no quant | 0.885 | 0.850 | -3.5 | 0.895 | 1.000 | 325.3 | 9.357 | 3098.7 / 3099.9 / 3138.2 |

Sources: `benchmarks/20260915T112519Z/results.csv` (all columns) and
`benchmarks/20260915T112519Z/models/<slug>/finetuned_predictions.csv` (raw outputs behind
strict/semantic). BF16 Qwen3.5-2B training time is the resumed segment (steps 400–600 only); BF16
Qwen3.5-4B is the final successful attempt.

## 6. Table 2 — QLoRA NF4 track (four configurations, fixed run order)

All rows: NF4 double-quantized 4-bit base (bf16 compute); adapter = restored best-validation
checkpoint (epoch 2, step 400). Rows are in the original run order. LoRA targets: explicit
q/k/v/o/gate/up/down projections for Qwen3-8B, `all-linear` with `finetune_vision_layers: false`
for the three vision configs. `Trainable params` is `train_metrics.json`
`trainable_parameters.numel` (tensor counts: 504 / 476 / 496 / 560; `vision_like_count: 0`).

| # | Model (official checkpoint) | Family | Kind | Nominal | Precision | Base strict | FT strict | Δ strict (pp) | FT semantic | FT valid | FT mean latency (ms) | Max reserved max-phase (GiB) | Training (s) | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-8B (`Qwen/Qwen3-8B`) | Qwen3 | language | 8B | NF4 / bf16 compute | 0.870 | 0.890 | +2.0 | 0.890 | 1.000 | 304.5 | 7.281 | 3184.3 / 3185.4 / 3235.5 | 43,646,976 |
| 2 | Ministral-3-8B-Instruct-2512-BF16 (`mistralai/Ministral-3-8B-Instruct-2512-BF16`) | Ministral-3 | vision | 8B | NF4 / bf16 compute | 0.815 | 0.875 | +6.0 | 0.885 | 0.995 | 314.6 | 7.338 | 5423.0 / 5424.5 / 5468.1 | 44,564,480 |
| 3 | Qwen3.5-9B (`Qwen/Qwen3.5-9B`) | Qwen3.5 | vision | 9B | NF4 / bf16 compute | 0.860 | 0.890 | +3.0 | 0.890 | 1.000 | 476.2 | 9.025 | 5236.6 / 5237.9 / 5282.7 | 43,278,336 |
| 4 | Ministral-3-14B-Instruct-2512-BF16 (`mistralai/Ministral-3-14B-Instruct-2512-BF16`) | Ministral-3 | vision | 14B | NF4 / bf16 compute | 0.855 | 0.885 | +3.0 | 0.890 | 1.000 | 466.1 | 9.939 | 6530.8 / 6532.1 / 6577.0 | 60,948,480 |

Sources: `benchmarks/qlora-large/20260916T185922Z/results.csv` (all columns except trainable
params); per-model `workspace/train_metrics.json` (trainable params, peak, best epoch);
`workspace/finetuned_predictions.csv` (raw outputs). Qwen3-8B rows are imported from the earlier
source run (no GPU sharing); the training time is that run's.

## 7. Descriptive findings (no ranking)

- **BF16: fine-tuning repaired a `feature-request` bias.** Base bug recall across the six configs
  ranged 0.18-0.88 and rose to 0.88-0.92 after fine-tuning (`per_class_recall` in
  `results.csv`). The largest strict gains came from the weakest baselines: Qwen3-1.7B
  0.580 → 0.890 (+31.0 pp), LFM2.5-1.2B 0.625 → 0.900 (+27.5 pp), Ministral-3-3B 0.610 → 0.875
  (+26.5 pp). Qwen3.5-4B is the only configuration where fine-tuning reduced strict accuracy
  (0.885 → 0.850, −3.5 pp), while its semantic accuracy still rose (+1.0 pp).
- **QLoRA: higher baselines, smaller gains.** Base strict was 0.815–0.870 and fine-tuned strict
  0.875–0.890 (+2.0 to +6.0 pp). Ministral-3-8B had the largest QLoRA strict gain (+6.0 pp) and
  produced the only unparseable fine-tuned output in either track (1/200; valid rate 0.995).
- **Strict vs semantic is a format effect, not just knowledge.** In the BF16 track, Qwen3.5-0.8B's
  fine-tuned **semantic** accuracy (0.905) is 10.0 pp above its strict accuracy (0.805): 22 of 200
  fine-tuned outputs parse to a single class but fail exact match, 20 of them to the gold class
  (the other 2 parse to the wrong class). Qwen3.5-4B shows the same pattern on 9 of 200 outputs,
  all to the gold class (strict 0.850 vs semantic 0.895). In QLoRA, Ministral-3-8B has 2/200
  semantic-only outputs (both gold) plus 1 unparseable; Ministral-3-14B's base semantic accuracy is 0.860 vs strict 0.855 (one baseline semantic-only row) and 0.890 vs 0.885 fine-tuned (1/200
  semantic-only, gold).
- **Fine-tuned generation was slower in all 10 configurations** (mean `model.generate` latency
  only; e.g. BF16 Qwen3.5-0.8B 130.2 → 197.4 ms, QLoRA Qwen3-8B 235.1 → 304.5 ms). Single pass;
  descriptive, not a throughput benchmark.
- **Peak reserved memory tracked model size within each track.** QLoRA max-phase peaks were
  7.281-9.939 GiB against 15.456 GiB torch-visible (the 14B config peaked at 9.939 GiB during
  training); BF16 fine-tuned eval peaks were 1.965-9.357 GiB. These are allocator peaks, not
  whole-GPU totals.
- **Nominal size did not order baseline accuracy in these runs.** BF16 base strict descending:
  0.885 > 0.840 > 0.780 > 0.625 > 0.610 > 0.580 (4B, 2B, 0.8B, 1.2B, 3B, 1.7B). QLoRA base
  strict descending: 0.870 > 0.860 > 0.855 > 0.815 (Qwen3-8B, Qwen3.5-9B, Ministral-3-14B,
  Ministral-3-8B). With 200 single-pass test items this is a descriptive observation only — no
  size or quality claim is made.

## 8. Three representative output cases (deterministic selection, not random)

The three cases below were selected **post-hoc** (after the runs, from the recorded predictions) by
a deterministic rule — not randomly, and not by browsing outputs. Candidate set = the 200 test
issue numbers present in all prediction CSVs of a track; "improvement" = most models in a track
whose fine-tuned strict prediction becomes correct where the baseline strict prediction was wrong;
"regression" = most models whose fine-tuned strict prediction becomes wrong where the baseline was
correct; ties broken by smallest `issue_number`. "Format case" = most fine-tuned rows whose
forgiving parse differs from the strict parse (including `INVALID`), same tie-break. The
deterministic winners used here: BF16 improvement `#335217` (6/6 models), BF16 regression
`#334676` (5/6 models), QLoRA format `#335623` (2/4 models). These cases are illustrative, not representative or random: they were explicitly chosen to highlight an improvement, a regression, and a format failure, and they are not population estimates. Raw outputs are quoted as JSON-escaped strings exactly as recorded in the CSVs (`\n` is a newline, `\u0060` is a backtick); they are kept compact.

### Case A — improvement (BF16 track), [microsoft/vscode#335217](https://github.com/microsoft/vscode/issues/335217), gold `bug`

All six BF16 configurations classified this issue as `feature-request` at baseline; all six output
`bug` after fine-tuning. Each raw output is exactly one label — no format noise.

| Track | Model (checkpoint) | Issue | Gold | Base raw (escaped) | Base strict / semantic | FT raw (escaped) | FT strict / semantic |
|---|---|---|---|---|---|---|---|
| BF16 | `Qwen/Qwen3.5-0.8B` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |
| BF16 | `LiquidAI/LFM2.5-1.2B-Instruct` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |
| BF16 | `Qwen/Qwen3-1.7B` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |
| BF16 | `Qwen/Qwen3.5-2B` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |
| BF16 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |
| BF16 | `Qwen/Qwen3.5-4B` | 335217 | bug | `"feature-request"` | wrong / wrong | `"bug"` | correct / correct |

Source: `benchmarks/20260915T112519Z/models/<slug>/{baseline,finetuned}_predictions.csv`.

### Case B — regression (BF16 track), [microsoft/vscode#334676](https://github.com/microsoft/vscode/issues/334676), gold `feature-request`

Five of six configurations flipped a correct baseline answer to `bug`; LFM2.5-1.2B stayed correct.
Note LFM2.5's baseline raw output is `Feature-request` with a capital F — strict matching is
case-insensitive, so it counts as correct.

| Track | Model (checkpoint) | Issue | Gold | Base raw (escaped) | Base strict / semantic | FT raw (escaped) | FT strict / semantic |
|---|---|---|---|---|---|---|---|
| BF16 | `Qwen/Qwen3.5-0.8B` | 334676 | feature-request | `"feature-request"` | correct / correct | `"bug"` | wrong / wrong |
| BF16 | `LiquidAI/LFM2.5-1.2B-Instruct` | 334676 | feature-request | `"Feature-request"` | correct / correct | `"feature-request"` | correct / correct |
| BF16 | `Qwen/Qwen3-1.7B` | 334676 | feature-request | `"feature-request"` | correct / correct | `"bug"` | wrong / wrong |
| BF16 | `Qwen/Qwen3.5-2B` | 334676 | feature-request | `"feature-request"` | correct / correct | `"bug"` | wrong / wrong |
| BF16 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 334676 | feature-request | `"feature-request"` | correct / correct | `"bug"` | wrong / wrong |
| BF16 | `Qwen/Qwen3.5-4B` | 334676 | feature-request | `"feature-request"` | correct / correct | `"bug"` | wrong / wrong |

Source: `benchmarks/20260915T112519Z/models/<slug>/{baseline,finetuned}_predictions.csv`.

### Case C — format-invalid / semantic-only (QLoRA track), [microsoft/vscode#335623](https://github.com/microsoft/vscode/issues/335623), gold `bug`

Fine-tuned Ministral-3-8B produced an unparseable completion (`INVALID / INVALID`); fine-tuned
Ministral-3-14B started with `bug` but continued generating a fenced JSON fragment, so it fails
exact match while the forgiving parse still recovers `bug` (semantic-only). The other two QLoRA
configs stayed plain `bug`. The last row is a **cross-track illustration only** (BF16 track, same
issue ID): its fine-tuned 0.8B adapter emitted `bug`, then a `user` marker and an empty
`<think>` block, then `bug` again; it is not evidence about the QLoRA outputs.

| Track | Model (checkpoint) | Issue | Gold | Base raw (escaped) | Base strict / semantic | FT raw (escaped) | FT strict / semantic |
|---|---|---|---|---|---|---|---|
| QLoRA | `Qwen/Qwen3-8B` | 335623 | bug | `"bug"` | correct / correct | `"bug"` | correct / correct |
| QLoRA | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 335623 | bug | `"bug"` | correct / correct | `"The SDK emits successful tool calls without \u0060result_token_count\u0060,"` | INVALID / INVALID |
| QLoRA | `Qwen/Qwen3.5-9B` | 335623 | bug | `"bug"` | correct / correct | `"bug"` | correct / correct |
| QLoRA | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 335623 | bug | `"bug"` | correct / correct | `"bug\n\u0060\u0060\u0060json\n{\n  \"tool_name\": \""` | INVALID / correct (semantic-only) |
| BF16 | `Qwen/Qwen3.5-0.8B` | 335623 | bug | `"feature-request"` | wrong / wrong | `"bug\nuser\n<think>\n\n</think>\n\nbug"` | INVALID / correct (semantic-only) |

Sources: `benchmarks/qlora-large/20260916T185922Z/models/<slug>/workspace/{baseline,finetuned}_predictions.csv`
and `benchmarks/20260915T112519Z/models/01-qwen3.5-0.8b/{baseline,finetuned}_predictions.csv`.
The BF16 row is on the same issue ID but a different track and run; no causal equivalence between
the two tracks is implied.

## 9. Counts, verification and reproduction

- The two runs contain **20 evaluation rows** (**12 BF16** + **8 QLoRA**) and **20 prediction CSVs** (one baseline and one fine-tuned per configuration), each with **200 data rows** and a 100/100 class split.
- The table values, counts, case issue IDs, gold labels and quoted raw outputs in this document
  were verified during authoring against the committed artifacts by a **temporary checker script**
  kept outside the repository and deliberately not published; readers can independently re-derive
  everything from `results.csv`, the metrics JSON and the prediction CSVs cited in each section.
- Exact end-to-end reproduction of the runs additionally needs the uncommitted dataset and follows
  the operational steps in [`docs/reproduction.md`](reproduction.md); weights, adapters and
  checkpoints are not published.
