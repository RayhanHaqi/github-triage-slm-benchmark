# slm-specialist

Config-driven fine-tuning harness for a small language model (SLM) that
classifies GitHub issues. The bundled config targets the frozen VSCode triage
experiment (bug vs feature-request); its reference inputs are the sibling
`../github-triage-data` files, which are intentionally not committed (see
"Artifact policy (git)").

Pipeline: `gather -> prepare -> baseline -> train -> evaluate -> comparison`.

## Layout

```
configs/vscode-bug-feature.yaml   one config: sources, cleaning, split, model, LoRA, trainer, eval
src/specialist/gather.py          copy configured raw JSON or fetch with `gh issue list`
src/specialist/prepare.py         cleaning + temporal split + SFT jsonl (byte-compatible recipe)
src/specialist/model.py           shared model loading (language/vision), prompts, parsers, run metadata
src/specialist/train.py           Unsloth LoRA SFT recipe
src/specialist/evaluate.py        shared baseline/fine-tuned evaluator (metrics + predictions CSV)
src/specialist/cli.py             CLI, config/path resolution, run-dir orchestration, comparison
data/                             default standalone workspace (only .gitkeep committed)
runs/                             UTC run directories from `specialist run`
tests/                            cheap regression tests (frozen split reproduction, parsers,
                                  config/model selection, prompt behavior, metadata)
```

## Setup

```bash
python -m pip install -e .          # CLI + PyYAML; gather/prepare/--help work with this alone
python -m pip install -e '.[ml]'    # torch, transformers, peft, datasets, trl, unsloth, tqdm
```

The ML phases expect the environment used for the frozen experiment
(`/home/tilakoid/miniconda3/envs/github-triage`). `specialist run` spawns each
phase with `sys.executable`, so use that environment's python. The `.[ml]`
extra pins the exact versions of the successful benchmark environment
(`benchmarks/20260915T112519Z`); `torch==2.11.0` is a platform/index-specific
wheel, so install the build matching your CUDA/driver when the default wheel
does not fit.

## Commands

```bash
specialist gather   <config> [--run-dir DIR]   # raw/<class>.json (byte copy or live gh)
specialist prepare  <config> [--run-dir DIR]   # train/val/test.jsonl + dataset_stats.json
specialist baseline <config> [--run-dir DIR]   # evaluate base model -> baseline_*
specialist train    <config> [--run-dir DIR]   # adapter/, trainer/, train_metrics.json
specialist evaluate <config> --checkpoint <path-or-base> [--run-dir DIR]
specialist run      <config> [--run-dir DIR]   # full pipeline in a unique UTC run dir
```

Without `--run-dir`, phases use `paths.data_dir` (staging under `data/`).
With `--run-dir DIR`, the phase reads/writes `DIR`; `run` snapshots the resolved
config to `DIR/config.yaml` and executes every phase as a subprocess in that dir
(fresh process per model load: clean Unsloth/transformers import order, VRAM
released between phases).

`--checkpoint` semantics (minimally ambiguous):

- `base` (or empty) — the configured base model, written as `baseline_*`.
- a directory containing `adapter_config.json` — base model + `PeftModel`, written as
  `finetuned_*`. The existing frozen adapter `../github-triage-data/vscode-triage-lora`
  loads this way; the base model always comes from config.
- anything else (HF model id or model directory without an adapter) — evaluated as a
  base model, written as `baseline_*`.

## Run directory artifacts

`specialist run` creates `runs/<UTC timestamp Z>/` (suffixed on collision)
containing at minimum:

```
config.yaml                resolved config snapshot
raw/                       gathered per-class raw JSON
train.jsonl val.jsonl test.jsonl   frozen input snapshot shared by both evals
dataset_stats.json         counts, conflicts, dropped records, split sizes
baseline_metrics.json / baseline_predictions.csv
train_metrics.json
adapter/                   final LoRA adapter (+ tokenizer/processor)
trainer/                   trainer checkpoints
finetuned_metrics.json / finetuned_predictions.csv
comparison.json            accuracy, delta pp, per-class recall, confusion, validity,
                           latency, issues/sec, peak VRAM, test_sha256 equality check
```

Both metric JSONs carry `test_sha256` computed from the exact split bytes;
`comparison.json` fails loudly if baseline and fine-tuned hashes differ.

## Benchmark results

Cross-model benchmarks live under `benchmarks/<UTC timestamp Z>/`: a root
`manifest.json` (model list, resolved revisions, dataset hashes, run-dir
mapping), optional `attempt_status.json`, and one `models/<NN>-<slug>/`
directory per model holding the same per-model artifacts as a `specialist run`
directory. Aggregated `report.md` / `results.csv` live at the benchmark root
alongside the manifest, and `benchmarks/LATEST` contains the relative run ID of
the newest run (`20260915T112519Z`). The earlier attempt `20260915T100825Z`
aborted on trainer compatibility bugs and is intentionally not committed. The
successful benchmark environment is pinned in the `.[ml]` extra of
`pyproject.toml`.

## Artifact policy (git)

Committed: source, configs, tests, run metadata (`config.yaml`,
`dataset_stats.json`, `run_info.json`, `*_metrics.json`, `comparison.json`),
prediction CSVs, and benchmark `manifest.json` / `report.md` / `results.csv`.
The successful reference run `runs/20260915T093353Z/` keeps its metadata,
metrics and predictions; its weights, logs and dataset splits stay ignored.
Ignored everywhere: model weights and adapter binaries (`*.safetensors`,
`*.bin`, `*.pt`, `*.pth`), `adapter/` and `trainer/` checkpoint directories,
Unsloth/HF caches, `__pycache__` and test caches, dataset copies (`raw/`,
`*.jsonl`), process transcripts (`*.log`, `error*.txt`) and temp files.

Frozen datasets are intentionally not committed: gathered `raw/` JSON and the
prepared `*.jsonl` splits are ignored, and the `../github-triage-data` source
files sit outside the repository, so a standalone clone cannot reconstruct the
frozen data. Exact reproduction requires sibling
`../github-triage-data/bugs.json` and `features.json` matching the dataset
hashes recorded in the run/benchmark metadata (`manifest.json`,
`dataset_stats.json`, `train_metrics.json`). Removing `file:` from a source
(keeping `repo`/`github_label`) switches to a non-frozen live `gh issue list`
gather instead.

## Config behavior

- One YAML covers everything: source repo, class -> GitHub label mapping, local
  source files, max examples per class, cleaning rules, temporal split ratios,
  base model, system prompt, max sequence length, LoRA params, trainer params,
  evaluation settings.
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
- `lora.target_modules` may be a YAML list or the string `all-linear`; the type
  is preserved as written. For `model.kind: vision`, the LoRA config may set
  `finetune_vision_layers` (default `false`), `finetune_language_layers`,
  `finetune_attention_modules`, `finetune_mlp_modules` (default `true`), passed
  to `FastVisionModel.get_peft_model`.
- `training.vision_collator` (default `false`, vision only): `true` trains
  message rows with `UnslothVisionDataCollator` (Ministral path); `false` trains
  rendered `text` rows through the ordinary collator (Qwen3.5 text-only path).
- Sources with `file:` are copied byte-for-byte when they are not larger than
  `cleaning.max_examples_per_class`; larger files are truncated to the first N
  rows in existing order. Remove `file:` from a source (keep `repo` and
  `github_label`) to gather live with `gh issue list --state all`, keeping the
  order `gh` returns.
- Cleaning regexes, split ratios, prompt and hyper-parameters match the frozen
  experiment exactly; `tests/test_prepare_repro.py` compares generated
  `train/val/test.jsonl` with the frozen files byte-for-byte when they exist.

## Reproducibility notes

- Preparation preserves class order (config order), cross-label number
  conflicts, exact-text duplicate normalization, the effective-empty rule,
  temporal per-class `int(n*0.8)` / `int(n*0.9)` splits, one seeded RNG sequence
  shuffling train then val then test, and JSON field order/serialization.
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
  and `library_versions` (python/torch/transformers/peft/datasets/trl/unsloth/
  unsloth_zoo plus CUDA/GPU when cheap). `train_metrics.json` additionally
  carries sha256 of the train/val/test bytes, seed, precision/quantization
  (`bf16`/`fp16`/`load_in_4bit`), the LoRA settings actually used and measured
  wall training time (`wall_train_seconds`, plus the trainer's own metrics).
- `baseline_predictions.csv` keeps the frozen 5-column format;
  `finetuned_predictions.csv` keeps the frozen 8-column format.
  Both `*_metrics.json` contain the full measurement set.
- Training-time validation is loss-only (`SFTConfig.prediction_loss_only=True`
  plus the logits-free `SFTTrainer` subclass in `src/specialist/train.py`), so
  eval uses the fused loss path instead of Unsloth forcing unused
  full-vocabulary logits.

## Tests

```bash
python -m unittest discover -s tests -v
```
