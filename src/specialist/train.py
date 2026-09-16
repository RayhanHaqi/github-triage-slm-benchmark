"""Unsloth LoRA SFT training, reproducing the frozen train_lora.py recipe.

Import order (datasets, unsloth, trl) matches the frozen script; run this phase
in its own process so baseline/eval VRAM is released before Unsloth patches.
Importing unsloth before trl is required on TRL 0.24: Unsloth replaces
`trl.SFTConfig`/`trl.SFTTrainer` with patched classes, and binding the originals
leads to `max_seq_length` TypeErrors plus the TRL config-rebuild path that
redacts `eos_token=None` into the literal `"<EOS_TOKEN>"` via `to_dict()`.

`model.kind: language` (default) keeps the frozen FastLanguageModel path.
`model.kind: vision` uses FastVisionModel on the configured upstream model id;
`training.vision_collator: true` (Ministral path) trains message rows with
UnslothVisionDataCollator, otherwise rows are rendered `text` rows processed by
the ordinary collator (Qwen3.5 text-only path).

Evaluation is loss-only (`SFTConfig.prediction_loss_only=True`); the local
SFTTrainer subclass bypasses Unsloth's logits-forcing global `prediction_step`
so eval uses the fused loss path instead of materialising full logits.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from . import model as model_mod


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize_target_modules(value):
    """Keep YAML strings (e.g. `all-linear`) as strings; copy lists/tuples."""
    if isinstance(value, str):
        return value
    return list(value)


def peft_kwargs(lora_cfg: dict, kind: str) -> dict:
    """Fast*Model.get_peft_model kwargs; vision adds the four layer/module flags."""
    kwargs = {
        "r": int(lora_cfg["r"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
        "target_modules": normalize_target_modules(lora_cfg["target_modules"]),
        "bias": lora_cfg["bias"],
        "use_gradient_checkpointing": lora_cfg["gradient_checkpointing"],
        "random_state": int(lora_cfg["random_state"]),
    }
    if kind == "vision":
        kwargs.update(
            finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
            finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
            finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
            finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
        )
    return kwargs


def resolve_eos_token(processor, model=None) -> str | None:
    """Valid EOS token string for `SFTConfig.eos_token` on processor-backed training.

    Prefers the underlying tokenizer's `eos_token`; falls back to the model
    config's `eos_token_id` (int or list) validated through the tokenizer.
    Returns None when no valid token exists so SFTConfig keeps its own default.
    Never returns a placeholder that is absent from the vocabulary.
    """
    tokenizer = getattr(processor, "tokenizer", processor)

    def _valid(token):
        if not token:
            return None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            return None
        return token

    token = _valid(getattr(tokenizer, "eos_token", None))
    if token is not None:
        return token

    eos_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_ids is None:
        return None
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    for eos_id in eos_ids:
        try:
            token = tokenizer.convert_ids_to_tokens(eos_id)
        except Exception:
            continue
        token = _valid(token)
        if token is not None:
            return token
    return None


def loss_only_sft_trainer_class(base):
    """SFTTrainer subclass with a logits-free `prediction_loss_only` eval step.

    Unsloth replaces `Trainer.prediction_step` process-wide and forces
    `UNSLOTH_RETURN_LOGITS=1` inside it, so `prediction_loss_only=True` eval
    still materialises full logits and can OOM. This override computes the
    loss itself under `UNSLOTH_RETURN_LOGITS=0` and never returns logits; any
    other mode delegates to the Unsloth-patched `super()` implementation.
    """

    class LossOnlySFTTrainer(base):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if not prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )

            import torch

            inputs = self._prepare_inputs(inputs)
            # Save/restore so an explicit user setting is not left clobbered.
            previous_return_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
            os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
            try:
                with torch.no_grad(), self.compute_loss_context_manager():
                    try:
                        num_items_in_batch = self._get_num_items_in_batch(
                            [inputs], self.args.device
                        )
                    except (AttributeError, TypeError):
                        num_items_in_batch = None
                    loss = self.compute_loss(
                        model,
                        inputs,
                        return_outputs=False,
                        num_items_in_batch=num_items_in_batch,
                    )
            finally:
                if previous_return_logits is None:
                    os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
                else:
                    os.environ["UNSLOTH_RETURN_LOGITS"] = previous_return_logits
            return loss.mean().detach(), None, None

    return LossOnlySFTTrainer


def _sha256(path: Path) -> str | None:
    return model_mod.sha256_file(path) if path.is_file() else None


def _reset_cuda_peak() -> None:
    """Zero CUDA peak counters so training peaks cover training only."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - driver/CUDA probing can fail
        pass


def _cuda_peak_gib() -> tuple[float, float]:
    """(peak allocated GiB, peak reserved GiB).

    Returns (0, 0) only when CUDA is genuinely unavailable; probe/inspection
    errors propagate so a silent zero can never pass the VRAM gate.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


# Conservative tokens: any trainable parameter whose name contains one of these
# means a vision-like tower/projector is actually being trained.
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


def trainable_parameter_summary(model, name_limit: int = 200, vision_name_limit: int = 50) -> dict:
    """Actual post-PEFT trainable parameter counts/names (compact, JSON-safe).

    Models that expose no `named_parameters()` (test doubles, exotic wrappers)
    record an explicit `unavailable` status; vision acceptance refuses that.
    """
    if not hasattr(model, "named_parameters"):
        return {
            "status": "unavailable",
            "reason": f"{type(model).__name__} exposes no named_parameters()",
        }
    names: list[str] = []
    sizes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        names.append(name)
        sizes[name] = int(parameter.numel())
    names.sort()
    vision_like = [
        name for name in names
        if any(token in name.lower() for token in VISION_LIKE_PARAMETER_TOKENS)
    ]
    return {
        "status": "available",
        "count": len(names),
        "numel": sum(sizes.values()),
        "names": names[:name_limit],
        "names_truncated": len(names) > name_limit,
        "vision_like_count": len(vision_like),
        "vision_like_numel": sum(sizes[name] for name in vision_like),
        "vision_like_names": vision_like[:vision_name_limit],
    }


def _best_epoch(log_history: list[dict], best_checkpoint: str | None) -> float | None:
    """Epoch of the checkpoint Trainer restored, when identifiable in log_history."""
    if not best_checkpoint:
        return None
    match = re.search(r"checkpoint-(\d+)\s*$", str(best_checkpoint))
    if not match:
        return None
    step = int(match.group(1))
    for entry in reversed(log_history):
        if entry.get("step") == step and entry.get("epoch") is not None:
            return entry["epoch"]
    return None


def _final_eval(log_history: list[dict]) -> tuple[float | None, float | None]:
    """(epoch, eval_loss) of the last recorded evaluation, else (None, None)."""
    for entry in reversed(log_history):
        if "eval_loss" in entry:
            return entry.get("epoch"), entry["eval_loss"]
    return None, None


def _validated_best_checkpoint(state) -> tuple[str, float]:
    """Fail closed on best-checkpoint selection before anything is saved.

    Returns (best_model_checkpoint, best_metric) only when the checkpoint path
    is non-empty, exists on disk and the metric is finite.
    """
    checkpoint = getattr(state, "best_model_checkpoint", None)
    if not checkpoint:
        raise RuntimeError(
            "load_best_model_at_end=True but the trainer reported no "
            "best_model_checkpoint; refusing to save an unvalidated adapter"
        )
    metric = getattr(state, "best_metric", None)
    try:
        metric_value = float(metric)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"load_best_model_at_end=True but best_metric is not numeric: {metric!r}"
        ) from None
    if not math.isfinite(metric_value):
        raise RuntimeError(
            f"load_best_model_at_end=True but best_metric is not finite: {metric_value!r}"
        )
    if not Path(str(checkpoint)).is_dir():
        raise RuntimeError(
            "load_best_model_at_end=True but best_model_checkpoint does not exist: "
            f"{checkpoint}"
        )
    return str(checkpoint), metric_value


def train(config: dict, workspace: str | Path) -> dict:
    from datasets import Dataset

    # Unsloth first: it patches trl so the bound SFTConfig/SFTTrainer are the
    # classes TRL's own isinstance checks expect (see module docstring).
    from unsloth import FastLanguageModel, FastVisionModel
    from trl import SFTConfig, SFTTrainer

    workspace = Path(workspace)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    kind = model_cfg.get("kind", "language")
    if kind not in ("language", "vision"):
        raise ValueError(f"unknown model kind: {kind!r} (use 'language' or 'vision')")

    FastModel = FastVisionModel if kind == "vision" else FastLanguageModel
    if kind == "vision":
        from unsloth.trainer import UnslothVisionDataCollator

    train_rows = load_jsonl(workspace / "train.jsonl")
    val_rows = load_jsonl(workspace / "val.jsonl")

    model_id = model_cfg["base_model"]
    max_seq_length = int(model_cfg["max_seq_length"])
    system_prompt = model_cfg["system_prompt"]
    revision = model_cfg.get("revision")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    use_exact_model_name = bool(model_cfg.get("use_exact_model_name", False))

    # Canonical explicit config. Unsloth 2026.9.4 forwards a user-provided
    # `quantization_config` through **kwargs (loader.py: "Respect a user-provided
    # quantization_config") and still quantizes `*-BF16` repos on the fly, so the
    # request is not left to the `load_in_4bit` flag alone.
    quantization_config = (
        model_mod.bitsandbytes_config() if load_in_4bit else None
    )

    print(f"Loading {model_id}...")

    # Normal BF16/FP16 LoRA, not QLoRA (unless the config asks for 4bit).
    load_kwargs = dict(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_exact_model_name=use_exact_model_name,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        # Embedding offload can leave parameters outside CUDA, which the
        # requested-4-bit placement validation rejects. Explicit for 4-bit
        # only; BF16 loads keep Unsloth's default.
        load_kwargs["offload_embedding"] = False

    model, processor = FastModel.from_pretrained(**load_kwargs)

    quantization_after_load = model_mod.assert_effective_quantization(
        model, load_in_4bit, context=f"Unsloth load of {model_id}"
    )

    model = FastModel.get_peft_model(model, **peft_kwargs(lora_cfg, kind))

    # Actual post-PEFT trainable parameter summary (vision-like names prove the
    # vision tower/projector really stayed frozen).
    trainable_parameters = trainable_parameter_summary(model)

    # Assert again on the PEFT-wrapped object: counting modules on the final
    # object is what makes the recorded metadata meaningful.
    quantization = model_mod.assert_effective_quantization(
        model, load_in_4bit, context=f"PEFT-wrapped {model_id}"
    )
    if load_in_4bit:
        print(
            "Effective 4-bit: "
            f"{quantization['quantized_module_count']} Linear4bit modules, "
            f"{quantization['quantized_parameter_count']} Params4bit parameters, "
            f"quant_type={quantization['quant_type']}, "
            f"compute_dtype={quantization['compute_dtype']}, "
            f"double_quant={quantization['double_quant']}"
        )

    # Render train/eval rows with the same prompt kwargs the evaluator uses.
    chat_template_kwargs = model_mod.configured_chat_template_kwargs(
        config.get("evaluation")
    )

    def messages_for(row: dict) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["label"]},
        ]

    def format_example(row: dict) -> str:
        return processor.apply_chat_template(
            messages_for(row),
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )

    vision_collator = kind == "vision" and bool(train_cfg.get("vision_collator", False))

    if vision_collator:
        train_dataset = Dataset.from_list(
            [{"messages": messages_for(row)} for row in train_rows]
        )
        val_dataset = Dataset.from_list(
            [{"messages": messages_for(row)} for row in val_rows]
        )
    else:
        train_dataset = Dataset.from_list(
            [{"text": format_example(row)} for row in train_rows]
        )
        val_dataset = Dataset.from_list(
            [{"text": format_example(row)} for row in val_rows]
        )

    print("Train:", len(train_dataset))
    print("Val  :", len(val_dataset))

    adapter_dir = workspace / train_cfg["adapter_subdir"]
    trainer_dir = workspace / train_cfg["output_subdir"]

    sft_kwargs = dict(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy=train_cfg["eval_strategy"],
        prediction_loss_only=True,
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=int(train_cfg["save_total_limit"]),
        bf16=bool(train_cfg["bf16"]),
        fp16=bool(train_cfg["fp16"]),
        optim=train_cfg["optim"],
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=int(train_cfg["seed"]),
        report_to=train_cfg["report_to"],
    )
    if kind == "vision":
        # TRL 0.24 validates `SFTConfig.eos_token` against the processing class;
        # pin a valid token from the underlying tokenizer/model config instead of
        # relying on the processor exposing one (generic, no model-name branching).
        eos_token = resolve_eos_token(processor, model)
        sft_kwargs["max_length"] = max_seq_length
        sft_kwargs["dataset_text_field"] = "text"
        if eos_token is not None:
            sft_kwargs["eos_token"] = eos_token
            print(f"EOS token: {eos_token}")
        else:
            print("EOS token: none resolvable; leaving SFTConfig.eos_token unset")
        if vision_collator:
            sft_kwargs.update(
                dataset_text_field="",
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
            )
    else:
        sft_kwargs["max_length"] = max_seq_length
        sft_kwargs["dataset_text_field"] = "text"

    # Optional best-checkpoint selection pass-through (e.g. eval_loss/epoch
    # configs): Trainer then restores the best checkpoint at the end of train().
    for key in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
        if train_cfg.get(key) is not None:
            sft_kwargs[key] = train_cfg[key]

    trainer_kwargs = dict(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(**sft_kwargs),
    )
    if vision_collator:
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(
            model, processor, max_seq_length=max_seq_length
        )

    # Unsloth's process-wide Trainer.prediction_step forces full logits during
    # eval; this subclass keeps `prediction_loss_only=True` evaluation
    # logits-free (see loss_only_sft_trainer_class).
    trainer = loss_only_sft_trainer_class(SFTTrainer)(**trainer_kwargs)

    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_from_checkpoint = str(Path(resume_from_checkpoint).expanduser().resolve())
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # Training-only CUDA peaks: reset after load/PEFT wrapping. The pre-reset
    # figures cover model setup (from_pretrained + get_peft_model).
    setup_peak_allocated_gib, setup_peak_reserved_gib = _cuda_peak_gib()
    _reset_cuda_peak()

    started = time.perf_counter()
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    wall_train_seconds = time.perf_counter() - started
    print("\nTraining complete.")
    print(trainer_stats)

    # Fail closed before any adapter bytes are written: with best-checkpoint
    # selection enabled, the trainer must have restored a real checkpoint.
    # Trainer.train() ends with `_load_best_model()` when the flag is set, so
    # saving here writes the restored best model.
    best_checkpoint = None
    best_metric = None
    if bool(train_cfg.get("load_best_model_at_end")):
        best_checkpoint, best_metric = _validated_best_checkpoint(trainer.state)

    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter: {adapter_dir}")

    log_history = trainer.state.log_history
    final_epoch = getattr(trainer.state, "epoch", None)
    if final_epoch is None:
        final_epoch = trainer_stats.metrics.get("epoch")
    final_eval_epoch, final_eval_loss = _final_eval(log_history)
    train_peak_allocated_gib, train_peak_reserved_gib = _cuda_peak_gib()

    metrics = {
        "base_model": model_id,
        "max_seq_length": max_seq_length,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "adapter_dir": str(adapter_dir),
        "trainer_dir": str(trainer_dir),
        "train_metrics": trainer_stats.metrics,
        "log_history": log_history,
        "model_kind": kind,
        "use_exact_model_name": use_exact_model_name,
        **model_mod.revision_metadata(revision, model, processor),
        "library_versions": model_mod.library_versions(),
        "sha256": {
            "train": _sha256(workspace / "train.jsonl"),
            "val": _sha256(workspace / "val.jsonl"),
            "test": _sha256(workspace / "test.jsonl"),
        },
        "seed": int(train_cfg["seed"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "best_model_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "best_epoch": _best_epoch(log_history, best_checkpoint),
        "final_epoch": final_epoch,
        "final_eval_epoch": final_eval_epoch,
        "final_eval_loss": final_eval_loss,
        "train_peak_allocated_gib": train_peak_allocated_gib,
        "train_peak_reserved_gib": train_peak_reserved_gib,
        "setup_peak_allocated_gib": setup_peak_allocated_gib,
        "setup_peak_reserved_gib": setup_peak_reserved_gib,
        "precision": {
            "bf16": bool(train_cfg["bf16"]),
            "fp16": bool(train_cfg["fp16"]),
            "load_in_4bit": load_in_4bit,
        },
        # Effective settings from the loaded modules, not the requested flags.
        "quantization": quantization,
        "quantization_after_load": quantization_after_load,
        "lora": peft_kwargs(lora_cfg, kind),
        "trainable_parameters": trainable_parameters,
        "wall_train_seconds": wall_train_seconds,
        "resume_from_checkpoint": resume_from_checkpoint,
        "prediction_loss_only": True,
    }
    (workspace / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("Saved: train_metrics.json")
    return metrics
