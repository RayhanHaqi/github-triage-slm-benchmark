# GitHub Issue Triage - SLM Fine-Tuning Benchmark

Case study: fine-tune several small language models (SLMs) on one narrow GitHub
triage task, then compare every fine-tuned adapter against its own base model.

Problem: classify a `microsoft/vscode` issue as `bug` or `feature-request`
(two classes only). This is not a general triage system - no severity,
assignment, multi-label routing or needs-info handling - and not a generic
framework. Internal names are intentionally unchanged: the Python package is
`slm-specialist` and the CLI is `specialist`; only the repository identity moved
to `github-triage-slm-benchmark`.

Research questions:

- Does fine-tuning improve over the base checkpoint, per model? (baseline vs
  fine-tuned, same frozen test split, same evaluator)
- How does the gain relate to nominal parameter count and measured resources
  (training wall time, peak VRAM)?

## Dataset (frozen)

- Source: `microsoft/vscode` issues; classes `bug` and `feature-request`.
- Frozen split: **1593 train / 200 val / 200 test**; temporal per-class by
  `created_at`, seed 42; test balanced 100 bug + 100 feature-request.
- Exact SHA-256 split pins and row counts are recorded in
  [docs/reproduction.md](docs/reproduction.md) and in every run's
  `manifest.json` / `comparison.json`.
- Raw and prepared data files are intentionally not committed; see
  [Reproduction](docs/reproduction.md).

## Two tracks, no unified ranking

Two completed runs, deliberately kept separate (track-local run IDs and `LATEST`
pointers):

| Track | Run ID | Configurations | Precision | Checkpoint note |
|---|---|---|---|---|
| BF16 LoRA | `20260915T112519Z` | 6 (roughly 0.8B - 4B nominal) | full BF16 base + LoRA | adapter = final epoch (3) |
| QLoRA NF4 | `20260916T185922Z` | 4 (roughly 8B - 14B nominal) | 4-bit NF4 base, bf16 compute + LoRA | best checkpoint (epoch 2, lowest val loss) restored |

- The BF16 track is historical; its vision-freeze behavior is **not
  established** (Unsloth's `all-linear` handling may have silently overridden
  the recorded freeze flags).
- In the QLoRA track, Qwen3-8B was **imported byte-identical** from earlier run
  `20260916T120019Z` and was not rerun; models 2-4 executed in this run. Vision
  freeze is established for this run only.
- Different precision, checkpoint-selection policy and provenance prevent a
  controlled cross-track ranking: cross-model tradeoffs may be read
  descriptively, but no causal model-size or precision ranking is claimed.
  Matched comparisons remain baseline vs fine-tuned within one configuration.

## Benchmark results (quick read)

Strict accuracy on the frozen 200-row test split (single runs;
deltas are adapter minus base, percentage points). FT peak VRAM is the
adapter-evaluation process peak **reserved** by the PyTorch allocator (GiB,
counters reset after model load); FT output speed is stored generated tokens
divided by total `model.generate` elapsed, so it includes prefill and is not
pure decode throughput. The QLoRA models 2-4 ran under the documented shared-GPU
(RustDesk) policy while the BF16 track and the imported Qwen3-8B evidence had no
GPU sharing, so these resource numbers are descriptive within each track and not
a controlled efficiency ranking.

BF16 LoRA track:

| Model | Strict base -> FT | Delta strict | FT peak VRAM (GiB) | FT output speed (tok/s) |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.780 -> 0.805 | +2.5 | 1.96 | 27.12 |
| LFM2.5-1.2B-Instruct | 0.625 -> 0.900 | +27.5 | 2.59 | 52.75 |
| Qwen3-1.7B | 0.580 -> 0.890 | +31.0 | 3.94 | 26.96 |
| Qwen3.5-2B | 0.840 -> 0.890 | +5.0 | 4.54 | 22.99 |
| Ministral-3-3B-Instruct | 0.610 -> 0.875 | +26.5 | 7.95 | 18.88 |
| Qwen3.5-4B | 0.885 -> 0.850 | -3.5 | 9.36 | 14.85 |

QLoRA NF4 track:

| Model | Strict base -> FT | Delta strict | FT peak VRAM (GiB) | FT output speed (tok/s) |
|---|---|---|---|---|
| Qwen3-8B (imported) | 0.870 -> 0.890 | +2.0 | 6.69 | 8.18 |
| Ministral-3-8B-Instruct | 0.815 -> 0.875 | +6.0 | 6.93 | 8.20 |
| Qwen3.5-9B | 0.860 -> 0.890 | +3.0 | 8.18 | 9.45 |
| Ministral-3-14B-Instruct | 0.855 -> 0.885 | +3.0 | 9.90 | 5.47 |

Evidence-backed takeaways:

- Strict accuracy improved in **9 of 10** configurations and semantic accuracy
  in **10 of 10**. The single strict regression (Qwen3.5-4B, -3.5 pp) still
  recorded +1.0 pp semantic.
- Gains do **not** order by nominal parameter count, and no "bigger is better"
  claim is made: the largest deltas come from 1.2B-3B configurations
  (+26.5 to +31.0 pp strict) while the smallest include both a 0.8B
  configuration (+2.5 pp) and the 8B-14B configurations (+2.0 to +6.0 pp).
- All 10 configurations trained and evaluated on one RTX 5060 Ti
  (torch-visible total 15.456 GiB, CUDA 12.8, pinned versions). Peak reserved
  VRAM: <= 9.36 GiB in BF16 adapter evaluation and <= 9.94 GiB in QLoRA
  acceptance runs.
- Deltas are single-run descriptive differences on a 200-row holdout; no
  statistical significance or general-quality claim.

## Case analysis and immutable evidence

- Main case analysis: [docs/github-triage-benchmark.md](docs/github-triage-benchmark.md)
  (per-model analysis; separate document).
- BF16 track run `20260915T112519Z`: [report](benchmarks/20260915T112519Z/report.md)
  | [results.csv](benchmarks/20260915T112519Z/results.csv)
  | [manifest.json](benchmarks/20260915T112519Z/manifest.json)
- QLoRA track run `20260916T185922Z`: [report](benchmarks/qlora-large/20260916T185922Z/report.md)
  | [results.csv](benchmarks/qlora-large/20260916T185922Z/results.csv)
  | [manifest.json](benchmarks/qlora-large/20260916T185922Z/manifest.json)
- Reference single-config run: [runs/20260915T093353Z/](runs/20260915T093353Z/)
  (original reference metadata, metrics, predictions).
- `benchmarks/20260915T100825Z` is an aborted earlier attempt; it is not part
  of the committed record.

Per-model metrics, comparisons, prediction CSVs and run info live under the
`models/` directory listed in each manifest.

## Repository map

```
configs/                     BF16 reference config + configs/qlora-large/*.yaml (4 fixed models)
src/specialist/              gather, prepare, model loading, train, evaluate, CLI, comparison
scripts/run_qlora_large.py   guarded sequential runner for the 4-model QLoRA track
benchmarks/                  committed run evidence: two tracks (+ uncommitted aborted attempt)
runs/                        reference run directory (metadata/metrics/predictions committed)
tests/                       cheap regression tests (no GPU, no network)
docs/                        case analysis and reproduction/operations notes
```

## Getting started (tooling)

```bash
git clone https://github.com/RayhanHaqi/github-triage-slm-benchmark.git   # private repository
cd github-triage-slm-benchmark
python -m pip install -e .                 # CLI + PyYAML only; gather/prepare/--help work with this
specialist --help
python -m unittest discover -s tests -v    # no GPU, no network
```

The ML phases (baseline/train/evaluate) need the pinned `.[ml]` extra and the
frozen experiment environment; see [docs/reproduction.md](docs/reproduction.md).

Reproduction constraints:

- The datasets and model weights are not committed, so a standalone clone cannot
  reproduce the frozen results. Exact reproduction requires the sibling
  `/home/tilakoid/github-triage-data` files with the recorded hashes; the QLoRA
  runner pins that machine-specific path.
- A live `gh issue list` gather produces a **new** dataset, not the frozen one.
  The runner `--dry-run` validates the pinned environment, GPU identity and
  >= 80 GiB free disk (not CPU-only).
- Resuming a finished committed run is not a quickstart: its weights were
  deleted at cleanup.

Full instructions: [docs/reproduction.md](docs/reproduction.md).

## Artifact policy (git)

Committed: source, configs, tests, run metadata (`config.yaml`,
`dataset_stats.json`, `run_info.json`, `*_metrics.json`, `comparison.json`),
prediction CSVs, and benchmark `manifest.json` / `report.md` / `results.csv`.
Not committed: dataset copies, model weights and adapters, trainer checkpoints,
caches, and process logs/transcripts. Retained manifests, metrics and
predictions are committed; diagnostic logs remain local and ignored. Only
failed bulk-run cache and model weights were deleted locally with user approval
- no blanket removal of retained artifacts. Details:
[docs/reproduction.md](docs/reproduction.md).

## Tests

```bash
python -m unittest discover -s tests -v
```
