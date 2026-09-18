# Reproduction and Operations

Operational guide for the GitHub issue triage SLM benchmark described in the
[root README](../README.md). It covers setup, the `specialist` pipeline
commands, run-directory artifacts, the two benchmark tracks, data provenance
and the artifact policy.

This is **not** a complete fresh-machine quickstart. The active prepared dataset
is public and pinned (see [Pinned dataset](#pinned-dataset-active-path)), so a
standalone clone can fetch and verify the exact splits, but reproducing the
published numbers still needs the pinned ML environment, a suitable GPU and the
configured base model weights. Raw collection snapshots are not published, and
historical run artifacts are unchanged.

## Setup

```bash
python -m pip install -e .          # CLI + PyYAML; gather/prepare/--help work with this alone
python -m pip install -e '.[ml]'    # torch, transformers, peft, datasets, trl, unsloth, tqdm
```

The ML phases expect the pinned environment of the benchmark: Python 3.11 and
the `.[ml]` extras. `specialist run` spawns each phase with `sys.executable`, so
run it with that environment's python. The `.[ml]` extra pins the exact versions
of the successful benchmark environments
(`benchmarks/20260915T112519Z` and `benchmarks/qlora-large/20260916T185922Z`);
`torch==2.11.0` is a platform/index-specific wheel, so install the build matching
your CUDA/driver when the default wheel does not fit.

## Commands

```bash
specialist gather   <config> [--run-dir DIR]   # legacy configs only: raw/<class>.json (byte copy or live gh)
specialist prepare  <config> [--run-dir DIR]   # pinned: fetch + verify splits; legacy: clean raw issues
specialist baseline <config> [--run-dir DIR]   # evaluate base model -> baseline_*
specialist train    <config> [--run-dir DIR]   # adapter/, trainer/, train_metrics.json
specialist evaluate <config> --checkpoint <path-or-base> [--run-dir DIR]
specialist run      <config> [--run-dir DIR]   # full pipeline in a unique UTC run dir
```

Without `--run-dir`, phases use `paths.data_dir` (staging under `data/`). With
`--run-dir DIR`, the phase reads/writes `DIR`; `run` snapshots the resolved
config to `DIR/config.yaml` and executes every phase as a subprocess in that dir
(fresh process per model load: clean Unsloth/transformers import order, VRAM
released between phases). Pinned configs skip `gather` in `run`.

`--checkpoint` semantics (minimally ambiguous):

- `base` (or empty) - the configured base model, written as `baseline_*`.
- a directory containing `adapter_config.json` - base model + `PeftModel`,
  written as `finetuned_*`. The base model always comes from config.
- anything else (HF model id or model directory without an adapter) - evaluated
  as a base model, written as `baseline_*`.

## Pinned dataset (active path)

Active configs (`configs/vscode-bug-feature.yaml`, `configs/qlora-large/*.yaml`)
declare a `dataset:` block. The trust root lives in
`src/specialist/dataset.py`:

- repo: `https://huggingface.co/datasets/Tilakoid/vscode-bug-feature-triage`
- revision: `15c7d77e083d0cd30ae84cc5de6add1dce6cf950` (full immutable SHA)
- remote -> local names: `train.jsonl` -> `train.jsonl`, `validation.jsonl` ->
  `val.jsonl`, `test.jsonl` -> `test.jsonl`

`specialist prepare configs/vscode-bug-feature.yaml` downloads only those three
files and verifies each before use:

- train: `5697146` bytes, `1593` rows, SHA-256
  `3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b`
- val: `716214` bytes, `200` rows, SHA-256
  `7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348`
- test: `640586` bytes, `200` rows, SHA-256
  `9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025`

The repo/revision are fixed by the module constants and a config must mirror
them exactly or it fails closed. Fetches build only
`https://huggingface.co/datasets/<repo>/resolve/<revision>/<remote>`; there is no
`main`/HEAD/metadata fallback.

Cache and failure behavior:

- A verified local split is used as-is and performs no network access.
  Standalone consumer phases (`baseline`, `train`, `evaluate`) verify the
  configured dataset before use and fail closed on missing, wrong-size,
  wrong-hash or wrong-row splits; they do not fetch.
- `prepare` may fetch missing or invalid splits. It streams into a temp file in
  the destination directory and atomically replaces the destination only after
  exact byte size, SHA-256 and non-empty row count verify. A failed fetch keeps
  any existing destination and removes the temp file.
- `specialist run` skips raw `gather` for pinned configs; calling `specialist
  gather` on one refuses. Raw gather and raw preparation remain available only
  for legacy configs without a `dataset:` block (gathered `raw/` or a configured
  `file`). The HF package ships only the prepared splits: raw collection cannot
  be reconstructed from it, and live `gh issue list` output is a new dataset,
  not reproduction.

CPU-safe verification of the pinned path:

```bash
specialist prepare configs/vscode-bug-feature.yaml --run-dir /tmp/pinned-check
sha256sum /tmp/pinned-check/train.jsonl /tmp/pinned-check/val.jsonl /tmp/pinned-check/test.jsonl
wc -c /tmp/pinned-check/train.jsonl /tmp/pinned-check/val.jsonl /tmp/pinned-check/test.jsonl
python -m unittest tests.test_dataset tests.test_prepare_repro -v
```

The `sha256sum` and `wc -c` outputs must match the three rows above. `wc -c`
prints plain byte counts (`5697146`, `716214`, `640586`).

## Run directory artifacts

`specialist run` creates `runs/<UTC timestamp Z>/` (suffixed on collision)
containing at minimum:

```
config.yaml                resolved config snapshot
raw/                       gathered per-class raw JSON (legacy configs only)
train.jsonl val.jsonl test.jsonl   verified pinned snapshot (or legacy prepare output)
dataset_stats.json         counts, conflicts, dropped records, split sizes
                           (legacy) or pinned dataset identity/sizes (pinned)
baseline_metrics.json / baseline_predictions.csv
train_metrics.json
adapter/                   final LoRA adapter (+ tokenizer/processor)
trainer/                   trainer checkpoints
finetuned_metrics.json / finetuned_predictions.csv
comparison.json            accuracy, delta pp, per-class recall, confusion, validity,
                           latency, issues/sec, peak VRAM, test_sha256 equality check
```

Both metric JSONs carry `test_sha256` computed from the exact split bytes;
`comparison.json` fails loudly if baseline and fine-tuned hashes differ, if the
base model identity/requested/resolved revision differs, or if their effective
quantization settings (4-bit, quant type, compute dtype, double quant,
quantized parameter devices, CPU/disk/meta offload counts) differ. Offload is
compared as counts, not module names, since adapter wrapping changes module-key
prefixes. Metric JSONs written before quantization metadata existed (legacy
BF16 runs) compare as BF16.

## Frozen data and reproduction constraints

- Prepared splits are fetched from the pinned public dataset (see
  [Pinned dataset](#pinned-dataset-active-path)) and are not committed. Raw
  collection snapshots are not published, so raw preparation cannot be
  reconstructed from the HF package.
- Historical preparation configs used sibling raw files outside the repository.
  A legacy config without `dataset:` still resolves gathered `raw/` or a
  configured `file`, and exact reproduction of pre-pin preparation needs those
  original raw snapshots.
- Recorded split evidence (identical to the pinned constants above):
  - train `3d58b700bb165187462986d719edc6b705e612af9aba2bd23ea1e1410f26202b` (1593 rows)
  - val `7423f47f80368404aa0d21e9bb9c19719b624a67519146c8c9d720f42e517348` (200 rows)
  - test `9fc58e7070c327adaa7b522cf1cd530b90c077dbd54513d00ea31dead5712025` (200 rows)
- Legacy live gather: remove `file:` from a source (keeping
  `repo`/`github_label`) to gather with `gh issue list --state all`. That
  produces a **new** dataset, not the prepared one; it is not reproduction.
- QLoRA runner data lives in the project-local, git-ignored cache
  `.cache/datasets/vscode-bug-feature`. New runs and `--dry-run` fetch missing
  or invalid splits and verify them; `--resume` is verify-only and never repairs
  missing or changed data. Run manifests record the HF repo, revision and split
  hashes, not the cache path.
- `--dry-run` for the QLoRA runner is not CPU-only: it fetches/verifies the
  pinned dataset and validates the pinned environment versions, the
  torch-visible GPU identity, >= 80 GiB free disk and a clean git tree/HEAD,
  without downloading model weights or running any phase. A new run additionally
  requires a git tree with no tracked changes and no untracked paths.
- Resuming a finished committed run is **not** a quickstart: run
  `benchmarks/qlora-large/20260916T185922Z` is committed with its completed
  results documents, and its model weights/adapter evidence were deleted at
  cleanup. `--resume` is for a partial, in-progress run on the same machine.

### Historical runs (pre-pin)

Committed benchmark and run artifacts are unchanged and may retain old absolute
source paths as provenance. Runs created before the pinned dataset change
require their recorded checkout and original local data; current strict resume
checks intentionally reject their old manifests and config digests. Do not
rewrite those artifacts.

## BF16 reference track

`configs/vscode-bug-feature.yaml` reproduces the reference experiment
(Qwen3-1.7B baseline/FT) end to end; `specialist prepare` fetches and verifies
the pinned splits first:

```bash
specialist prepare configs/vscode-bug-feature.yaml
specialist run configs/vscode-bug-feature.yaml
```

The committed reference run is `runs/20260915T093353Z/` (metadata, metrics and
predictions; weights, logs and dataset splits are ignored). The six-model BF16
benchmark under `benchmarks/20260915T112519Z/` uses per-model configs derived
from the same recipe (see its `manifest.json`); that track predates the QLoRA
runner and records no GPU-sharing policy.

## QLoRA-large track (runner operations)

`scripts/run_qlora_large.py` runs the fixed four-model 4-bit QLoRA benchmark in
its own namespace, `benchmarks/qlora-large/<UTC run id>/`. It never writes to or
deletes the BF16 track (`benchmarks/20260915T112519Z`, `benchmarks/LATEST`); the
track-local `benchmarks/qlora-large/LATEST` is written only after all four
models complete. Model order, repos, revisions, kinds and batch/GA are
hard-coded and validated against `configs/qlora-large/*.yaml` before any
download or GPU load.

```bash
python scripts/run_qlora_large.py --dry-run          # fetch/verify data; validate env/GPU/disk/git (no model weights)
python scripts/run_qlora_large.py                    # new run (requires a clean git tree)
python scripts/run_qlora_large.py --resume benchmarks/qlora-large/<run-id>
python scripts/run_qlora_large.py --resume benchmarks/qlora-large/<run-id> --retry-failed
```

New runs require the pinned ML environment and a git tree with no tracked
changes and no untracked paths; while a run is in progress or being resumed,
only files under that exact `benchmarks/qlora-large/<run-id>/` directory are
tolerated, so previously retained runs and the track `LATEST` must be committed
before the next run starts. All phases and probes run with the runner's own
resolved `sys.executable` (there is no interpreter override flag); the manifest
records that interpreter and `--resume` refuses a different one. The manifest
records the git commit, config digests, the pinned HF dataset identity
(repo/revision/remote/local names/sizes/rows/SHA-256 per split) and the
PyTorch-visible GPU identity/capacity. New manifests carry per-split size, row
and SHA-256 evidence but never the local cache path. The invariant guard
re-checks clean git, the recorded HEAD/config digests/environment/pinned dataset
identity/GPU identity, >= 80 GiB free disk and no unapproved GPU compute
processes before starting every model (nvidia-smi failures are hard errors, not
"idle"; the one approved RustDesk exception is described under
[GPU sharing policy](#gpu-sharing-policy-implemented-runner)).

`--resume` refuses a different commit/config/environment/dataset/GPU and never
reruns a phase marked successful (baseline included); a completed entry is only
skipped after its retained evidence
(run_info/acceptance/cleanup/selection/preflight/metrics/200-row
predictions/comparison/snapshot digest) validates. Dataset handling on resume is
verify-only: the cache must match the pinned sizes/rows/hashes, and a missing or
changed split is refused rather than repaired. `--retry-failed` allows
exactly one retry of a phase whose failure matches the positive transient
allowlist (HF/network 5xx/timeout/connection, AF_UNIX short-path, explicit
external interruption); everything else, including OOM, quantization, NaN/Inf,
architecture errors, offload and red-VRAM, is a hard stop. A train retry
resumes only from a checkpoint that actually validates (directory step ==
`trainer_state.json.global_step`, non-empty adapter + optimizer + scheduler +
rng states that load, readable safetensors/JSON) via a retained
attempt-specific config; otherwise the partial train output is deleted and the
phase restarts identically.

Per model the runner first performs a preflight (8 train / 2 val / 1 test real
rows with >= 2048 rendered tokens at the resolved pinned SHA, a tiny run
through the production train path, and one quantized baseline plus one
selected-adapter generation, all re-validated for NF4/no-offload/revision/best-
checkpoint) in fresh `preflight/attempt-N/{first,repeat}` workspaces behind a
VRAM gate (<= 13.50 GiB go, 13.50-14.25 repeat once, otherwise stop), then runs
baseline -> train -> adapter eval -> comparison -> acceptance -> cleanup
strictly sequentially.

Retained under `benchmarks/qlora-large/<run-id>/`: `manifest.json` (status
`completed` only after four successes and only then the track-local `LATEST`,
written via sibling temp + fsync + atomic replace; otherwise `partial`/
`failed`), per-model `config.yaml`, `run_info.json` with phase
attempts/timestamps, `acceptance.json` (revision/quantization/hash/epoch/
best-checkpoint/VRAM checks), `cleanup.json` (GPU process-policy state
before/after plus bytes deleted, written atomically), `preflight/{selection.json, preflight.json}`,
`workspace/` metrics, predictions and comparison, and `logs/`. Only scratch is
deleted after acceptance (per-model run-local `.cache/<slug>`, adapter, trainer,
copied JSONLs, the runner's short temp scratch root); a successful preflight
attempt scratch is deleted before the full phases, while failed attempt
diagnostics are kept. Retained run metadata must be committed before a new
clean-tree run can start.

### GPU sharing policy (implemented runner)

The QLoRA runner implements a one-time user-approved exception
(`rustdesk_shared_gpu_v1`): only the exact installed
`/usr/share/rustdesk/rustdesk` executable may keep running during GPU phases,
with an aggregate allowance of 512 MiB; anything else fails closed, and nothing
is ever killed. This is **not** a generic "GPU is idle" claim: budgets are
point-in-time phase-boundary samples (before/after each GPU subprocess plus an
acceptance sample), and VRAM headroom is judged against effective capacity =
`min(torch-visible total, observed free memory - remaining RustDesk allowance)`
with a minimum headroom of 1.0 GiB. In the completed QLoRA run, models 2-4
(NF4) ran under this policy, with reserved peaks staying more than 1 GiB below
their retained budgets; the imported Qwen3-8B evidence was measured without
GPU sharing, so its timing/VRAM numbers are not directly comparable. The
historical BF16 track predates this policy and records no GPU sharing.

### Precision and checkpoint selection

- QLoRA track: NF4 4-bit base weights, bf16 compute, double quantization, no
  offload; best checkpoint (epoch 2, lowest validation loss) restored before
  evaluation.
- BF16 track: full BF16 base weights, no quantization; the final epoch (3)
  adapter is what was evaluated.
- These differences, plus the imported Qwen3-8B evidence, prevent a controlled
  cross-track ranking.
- Vision `all-linear`: at the actual PEFT call site, a vision config whose
  resolved `target_modules` is the literal string `all-linear` is passed as
  `None` so Unsloth honors the explicit `finetune_vision_layers: false` flags
  instead of silently forcing every `finetune_*` flag true. The recorded config
  and metrics keep the requested value unchanged; the language path is
  unaffected.
- Historical BF16 vision runs may have been affected by that Unsloth override:
  their frozen-vision behavior is not established, and that track remains
  unchanged. The QLoRA run's frozen vision is established for that run only.

## Config behavior

- One YAML covers source label order and metadata, max examples per class,
  cleaning rules, temporal split ratios, base model, system prompt, max sequence
  length, LoRA params, trainer params and evaluation settings. The `dataset:`
  block mirrors the pinned public repo/revision; the repo, revision, remote and
  local names, sizes, rows and hashes are fixed by
  `src/specialist/dataset.py`, so editing the YAML cannot repoint the trust
  root. A config without a `dataset:` block keeps the legacy raw source path.
- `~` and `$ENV_VARS` are expanded before path resolution; relative paths
  resolve against the project root (parent of `configs/`).
- Optional model fields (reference defaults unchanged): `model.revision` (passed
  to every tokenizer/processor/model/Unsloth `from_pretrained` call),
  `model.kind` (`language` default; `vision` loads `AutoProcessor` +
  `AutoModelForImageTextToText` and trains with `FastVisionModel`), and
  `model.trust_remote_code` (default `false`).
- `evaluation.chat_template_kwargs` is the exact mapping passed to
  `apply_chat_template(..., add_generation_prompt=True)` by the shared evaluator
  (and used to render training rows): Qwen models `{enable_thinking: false}`,
  LFM/Ministral `{}`. Absent key keeps the frozen `enable_thinking=False`
  default, so the bundled reference config is unchanged.
- `lora.target_modules` may be a YAML list or the string `all-linear`; the
  requested value is recorded as written (see "Precision and checkpoint
  selection" for the vision PEFT-call mapping). For `model.kind: vision`, the
  LoRA config may set `finetune_vision_layers` (default `false`),
  `finetune_language_layers`, `finetune_attention_modules`,
  `finetune_mlp_modules` (default `true`), passed to
  `FastVisionModel.get_peft_model`.
- `training.vision_collator` (default `false`, vision only): `true` trains
  message rows with `UnslothVisionDataCollator` (Ministral path); `false` trains
  rendered `text` rows through the ordinary collator (Qwen text-only path).
- `model.load_in_4bit` (default `false`): loads/evaluates/trains in 4-bit via an
  explicit canonical `transformers.BitsAndBytesConfig` (NF4, double quant, BF16
  compute) passed as `quantization_config` to the model only (never to the
  tokenizer/processor) for both the transformers evaluator and Unsloth training
  (which also quantizes official `*-BF16` repos on the fly). After every load
  (and after PEFT wrapping in training) the harness fails closed unless the
  loaded object really has packed bitsandbytes `Linear4bit`/`Params4bit` weights
  (`bnb_quantized` + quant state), quant type exactly `nf4`, BF16 compute dtype,
  nested double quantization, zero `Linear8bitLt` layers and CUDA-only placement
  (`hf_device_map` or parameter devices on CPU/disk/meta are rejected). It
  records effective metadata (counts, settings, placement) instead of echoing
  the request. Default BF16 behavior is unchanged.
- `training.load_best_model_at_end`, `training.metric_for_best_model`,
  `training.greater_is_better` (optional): passed through to `SFTConfig`. With
  best-checkpoint selection enabled, training fails closed before anything is
  saved unless `Trainer` reported a non-empty, existing `best_model_checkpoint`
  and a finite `best_metric`; the adapter is written only after
  `Trainer.train()` restored that checkpoint. `train_metrics.json` records
  `best_model_checkpoint`, `best_metric`, the best epoch when derivable, the
  final `epoch` and the final-epoch `eval_loss` separately, plus CUDA training
  peak allocated/reserved GiB.
- Legacy sources with `file:` are copied byte-for-byte when they are not larger
  than `cleaning.max_examples_per_class`; larger files are truncated to the
  first N rows in existing order. Removing `file:` from a legacy source (keeping
  `repo` and `github_label`) gathers live with `gh issue list --state all`,
  keeping the order `gh` returns. Pinned configs ignore raw sources entirely and
  refuse `gather`.
- Cleaning regexes, split ratios, prompt and hyper-parameters match the frozen
  experiment exactly; `tests/test_prepare_repro.py` compares legacy raw
  preparation with the frozen files byte-for-byte when they exist, while pinned
  `prepare` is verified against the trusted constants and covered by
  `tests/test_dataset.py`.

## Reproducibility notes

- Legacy raw preparation preserves class order (config order), cross-label
  number conflicts, exact-text duplicate normalization, the effective-empty
  rule, temporal per-class `int(n*0.8)` / `int(n*0.9)` splits, one seeded RNG
  sequence shuffling train then val then test, and JSON field order/
  serialization. The pinned prepared splits were produced with that recipe from
  the original raw snapshots; the pinned path consumes them directly and does
  not reconstruct raw collection.
- Baseline and fine-tuned runs share one evaluator: same test bytes, prompt,
  1800-token issue-only truncation (via the underlying tokenizer for both
  language and vision models), chat template
  (`add_generation_prompt=True` plus `evaluation.chat_template_kwargs`, default
  `enable_thinking=False`), greedy `max_new_tokens=12` decoding with
  `pad_token_id=eos`, strict + semantic parsers, confusion with INVALID,
  per-class recall, latency/throughput/token stats and peak allocated/reserved
  VRAM (counters reset after model load).
- Both `*_metrics.json` record `model_kind`, requested/resolved revision (the
  resolved SHA comes from the loaded `config._commit_hash` when exposed, else
  the requested revision is labelled `resolved_revision_source: requested`)
  and `library_versions` (python/torch/transformers/peft/bitsandbytes/datasets/
  trl/unsloth/unsloth_zoo plus CUDA/GPU when cheap). `train_metrics.json`
  additionally carries sha256 of the train/val/test bytes, seed, requested
  precision (`bf16`/`fp16`/`load_in_4bit`), effective `quantization` metadata,
  the LoRA settings actually used and measured wall training time
  (`wall_train_seconds`, plus the trainer's own metrics).
- `baseline_predictions.csv` keeps the frozen 5-column format;
  `finetuned_predictions.csv` keeps the frozen 8-column format. Both
  `*_metrics.json` contain the full measurement set.
- Training-time validation is loss-only (`SFTConfig.prediction_loss_only=True`
  plus the logits-free `SFTTrainer` subclass in `src/specialist/train.py`), so
  eval uses the fused loss path instead of Unsloth forcing unused
  full-vocabulary logits.

## Artifact policy (git)

Committed: source, configs, tests, run metadata (`config.yaml`,
`dataset_stats.json`, `run_info.json`, `*_metrics.json`, `comparison.json`),
prediction CSVs, and benchmark `manifest.json` / `report.md` / `results.csv`.
The successful reference run `runs/20260915T093353Z/` keeps its metadata,
metrics and predictions; its weights, logs and dataset splits stay ignored.

Ignored everywhere: model weights and adapter binaries (`*.safetensors`,
`*.bin`, `*.pt`, `*.pth`), `adapter/` and `trainer/` checkpoint directories,
Unsloth/HF caches, `__pycache__` and test caches, dataset copies (`raw/`,
`*.jsonl`), process transcripts (`*.log`, `error*.txt`) and temp files. Logs
and transcripts are not published. Historical failed bulk-run scratch was
removed locally with user approval; retained manifests and logs stayed on this
machine only.

Prepared datasets are intentionally not committed. The active path fetches and
verifies the pinned public splits described in
[Pinned dataset](#pinned-dataset-active-path) (cache under
`.cache/datasets/vscode-bug-feature` is ignored); legacy raw snapshots and
historical run inputs are not published.

## Tests

```bash
python -m unittest discover -s tests -v
```
