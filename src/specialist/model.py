"""Shared model loading, prompt construction, output parsers and run metadata.

Used by both baseline and fine-tuned evaluation so test file, prompt,
truncation, decoding, parsing and metrics stay identical. Heavy ML imports are
done lazily inside functions so CLI/gather/prepare stay lightweight.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Frozen reference default: the bundled config predates
# `evaluation.chat_template_kwargs`, so an absent field keeps the frozen
# `enable_thinking=False`. Per-run configs set the exact mapping instead
# (Qwen models {enable_thinking: false}, LFM/Ministral {}).
REFERENCE_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# Reported by `library_versions()` for run metadata; missing packages -> None.
_VERSION_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "trl",
    "unsloth",
    "unsloth_zoo",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_chat_template_kwargs(config: dict | None) -> dict:
    """Chat-template kwargs from `evaluation.chat_template_kwargs`.

    Missing key -> frozen reference default (`enable_thinking=False`); an
    explicit `{}` means no extra kwargs at all.
    """
    if not config or "chat_template_kwargs" not in config:
        return dict(REFERENCE_CHAT_TEMPLATE_KWARGS)
    return dict(config.get("chat_template_kwargs") or {})


def _prompt_tokenizer(prompt_source):
    """Underlying tokenizer: `processor.tokenizer` when present, else the object."""
    return getattr(prompt_source, "tokenizer", prompt_source)


def build_prompt(
    prompt_source,
    issue_text: str,
    system_prompt: str,
    max_user_tokens: int,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, int]:
    """Truncate the issue text alone, then apply the chat template.

    `prompt_source` is a tokenizer (language) or a processor (vision); issue
    truncation/decoding uses the underlying tokenizer in both cases. Returns
    (prompt, user_token_count); `chat_template_kwargs=None` keeps the frozen
    reference kwargs.
    """
    tokenizer = _prompt_tokenizer(prompt_source)

    user_ids = tokenizer(
        issue_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_user_tokens,
    )["input_ids"]

    user_text = tokenizer.decode(user_ids, skip_special_tokens=True)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    if chat_template_kwargs is None:
        chat_template_kwargs = REFERENCE_CHAT_TEMPLATE_KWARGS

    prompt = prompt_source.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    return prompt, len(user_ids)


def _label_pattern(label: str) -> str:
    """Verbatim-equivalent regex for a class name.

    "bug" -> \\bbug\\b, "feature-request" -> \\bfeature[\\s-]+request\\b, which is
    exactly the frozen parser's pattern for the two frozen classes.
    """
    parts = [re.escape(part) for part in re.split(r"[^A-Za-z0-9]+", label) if part]
    return r"\b" + r"[\s-]+".join(parts) + r"\b"


def strict_prediction(raw: str, labels: list[str]) -> str | None:
    """Accept only exactly one of the class names (case-insensitive, stripped)."""
    text = raw.strip().lower()
    if text in {label.lower() for label in labels}:
        return text
    return None


def parse_prediction(raw: str, labels: list[str]) -> str | None:
    """Forgiving semantic parser; both/neither class mentioned -> None (INVALID)."""
    text = raw.strip().lower()
    matches = [label for label in labels if re.search(_label_pattern(label), text)]
    if len(matches) == 1:
        return matches[0]
    return None


def library_versions() -> dict:
    """Installed versions + CUDA/GPU names for run metadata (cheap, JSON-safe)."""
    from importlib import metadata

    versions: dict = {"python": sys.version.split()[0]}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    try:
        import torch
    except ImportError:  # pragma: no cover - torch missing outside ML envs
        return versions

    versions["cuda"] = getattr(torch.version, "cuda", None)
    try:
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if versions["cuda_available"]:
            versions["gpu"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:  # pragma: no cover - driver/CUDA probing can fail
        versions["cuda_available"] = False
    return versions


def resolved_commit_hash(*objects) -> str | None:
    """Best-effort resolved hub SHA (`_commit_hash`) from model/tokenizer/processor."""
    for obj in objects:
        if obj is None:
            continue
        for holder in (getattr(obj, "config", None), getattr(obj, "init_kwargs", None)):
            if holder is None:
                continue
            value = (
                holder.get("_commit_hash")
                if isinstance(holder, dict)
                else getattr(holder, "_commit_hash", None)
            )
            if value:
                return str(value)
    return None


def revision_metadata(requested: str | None, *objects) -> dict:
    """Requested revision + resolved SHA; accurately labelled when only requested exists."""
    resolved = resolved_commit_hash(*objects)
    return {
        "requested_revision": requested,
        "resolved_revision": resolved or requested,
        "resolved_revision_source": (
            "model_config" if resolved else ("requested" if requested else None)
        ),
    }


@dataclass
class LoadedModel:
    """A loaded base model (optionally + adapter) with its tokenizer/processor.

    `tokenizer` is always the underlying tokenizer (truncation, decode, EOS);
    `processor` is set only for vision models (prompt rendering + tokenization).
    """

    model: object
    tokenizer: object
    processor: object | None
    kind: str
    requested_revision: str | None = None

    @property
    def prompt_source(self):
        """Chat-template source: the processor for vision, else the tokenizer."""
        return self.processor if self.processor is not None else self.tokenizer

    def metadata(self) -> dict:
        return {
            "model_kind": self.kind,
            **revision_metadata(
                self.requested_revision, self.model, self.tokenizer, self.processor
            ),
        }


def _from_pretrained_kwargs(revision: str | None, trust_remote_code: bool) -> dict:
    kwargs = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    return kwargs


def load_model(
    base_model: str,
    adapter: str | None = None,
    *,
    revision: str | None = None,
    kind: str = "language",
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a language or vision-conditional base model, optionally + PeftModel adapter.

    `model.kind: language` keeps AutoTokenizer + AutoModelForCausalLM;
    `vision` uses AutoProcessor + AutoModelForImageTextToText.
    """
    if kind not in ("language", "vision"):
        raise ValueError(f"unknown model kind: {kind!r} (use 'language' or 'vision')")

    load_kwargs = _from_pretrained_kwargs(revision, trust_remote_code)

    if kind == "vision":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(base_model, **load_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"processor for {base_model} exposes no tokenizer")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            **load_kwargs,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        processor = None
        tokenizer = AutoTokenizer.from_pretrained(base_model, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            **load_kwargs,
        )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        kind=kind,
        requested_revision=revision,
    )


def tokenize_prompt(loaded: LoadedModel, prompt: str) -> dict:
    """Tokenize a rendered prompt: `processor(text=...)` for vision, tokenizer otherwise."""
    if loaded.processor is not None:
        return loaded.processor(text=prompt, return_tensors="pt")
    return loaded.tokenizer(prompt, return_tensors="pt")
