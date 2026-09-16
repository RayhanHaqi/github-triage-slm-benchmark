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
    "bitsandbytes",
    "datasets",
    "trl",
    "unsloth",
    "unsloth_zoo",
)

# Canonical 4-bit recipe used everywhere the harness supports quantization:
# NF4 weights, double quantization, BF16 compute. Only `load_in_4bit` is a
# config toggle; the recipe itself is fixed.
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_DOUBLE_QUANT = True
# `str(torch.bfloat16)`; kept as a string so inspection stays JSON-safe and
# comparable without importing torch.
BF16_DTYPE_NAME = "torch.bfloat16"


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


def bitsandbytes_config(
    *,
    quant_type: str = BNB_4BIT_QUANT_TYPE,
    compute_dtype: str = BNB_4BIT_COMPUTE_DTYPE,
    use_double_quant: bool = BNB_4BIT_DOUBLE_QUANT,
):
    """Explicit `transformers.BitsAndBytesConfig` for the canonical 4-bit load.

    Callers always pass this object as `quantization_config` (never just a
    `load_in_4bit`/`dtype` flag) so transformers and Unsloth both quantize on
    the fly, including `*-BF16` checkpoints that carry no quantized weights.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def _sorted_unique(values) -> list[str] | None:
    unique = sorted({str(value) for value in values if value is not None})
    return unique or None


def _bitsandbytes():
    try:
        import bitsandbytes as bnb
    except Exception:
        return None
    return bnb


def _normalized(value) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _packed_quant_state(weight, params4_cls):
    """Real packed-state evidence for a `Params4bit` weight (bnb 0.50.2), else None.

    A packed weight carries `bnb_quantized=True` and a `QuantState` with
    `absmax`/`code`; anything less is a partially materialised layout.
    """
    if params4_cls is None or not isinstance(weight, params4_cls):
        return None
    if getattr(weight, "bnb_quantized", False) is not True:
        return None
    state = getattr(weight, "quant_state", None)
    if state is None:
        return None
    if getattr(state, "absmax", None) is None or getattr(state, "code", None) is None:
        return None
    return state


def quantization_stats(model, requested_load_in_4bit: bool = False) -> dict:
    """Effective quantization of a loaded model: actual modules, not the request.

    Counts bitsandbytes `Linear4bit` modules / `Params4bit` parameters
    (deduplicated by weight identity for tied weights), the packed subset, the
    settings those objects carry (quant type, compute dtype, nested double
    quant) plus `hf_device_map`, quantized-parameter devices and the devices of
    every model parameter. Traversal errors propagate: swallowing one would
    read as "not 4-bit" after a requested 4-bit load.
    """
    bnb = _bitsandbytes()
    if requested_load_in_4bit and bnb is None:
        raise RuntimeError(
            "requested a 4-bit load but bitsandbytes is not importable; "
            "install the pinned bitsandbytes==0.50.2"
        )

    has_modules = hasattr(model, "modules")
    has_parameters = hasattr(model, "parameters")
    if requested_load_in_4bit and not (has_modules and has_parameters):
        raise RuntimeError(
            f"cannot inspect requested 4-bit placement: {type(model).__name__} "
            "exposes no modules()/parameters() (not a torch Module)"
        )
    modules = list(model.modules()) if has_modules else []
    parameters = list(model.parameters()) if has_parameters else []

    nn = getattr(bnb, "nn", None) if bnb is not None else None
    linear4_cls = getattr(nn, "Linear4bit", None)
    linear8_cls = getattr(nn, "Linear8bitLt", None)
    params4_cls = getattr(nn, "Params4bit", None)

    linear4 = [m for m in modules if linear4_cls is not None and isinstance(m, linear4_cls)]
    linear8 = [m for m in modules if linear8_cls is not None and isinstance(m, linear8_cls)]
    params4 = [p for p in parameters if params4_cls is not None and isinstance(p, params4_cls)]

    # Deduplicate by object identity: tied/shared Linear4bit weights appear once
    # per module but are a single parameter, and counts must never go negative.
    unique_weights: list = []
    seen_weights: set[int] = set()
    for module in linear4:
        weight = getattr(module, "weight", None)
        if id(weight) in seen_weights:
            continue
        seen_weights.add(id(weight))
        unique_weights.append(weight)

    packed_states = [
        _packed_quant_state(weight, params4_cls) for weight in unique_weights
    ]
    packed_weights = [
        weight
        for weight, state in zip(unique_weights, packed_states)
        if state is not None
    ]
    # Direct count: an unpacked parameter is one whose packed state is absent.
    unpacked_parameter_count = sum(1 for state in packed_states if state is None)
    params4_weight_count = sum(
        1
        for weight in unique_weights
        if params4_cls is not None and isinstance(weight, params4_cls)
    )
    non_params4_weights = len(unique_weights) - params4_weight_count

    quant_types: list[str] = []
    for weight in unique_weights:
        for value in (
            getattr(weight, "quant_type", None),
            getattr(getattr(weight, "quant_state", None), "quant_type", None),
        ):
            normalized = _normalized(value)
            if normalized is not None:
                quant_types.append(normalized)

    # Double quant must be requested AND structurally present per weight.
    double_flags = [
        bool(
            getattr(weight, "compress_statistics", False)
            and state is not None
            and getattr(state, "nested", False)
            and getattr(state, "state2", None) is not None
        )
        for weight, state in zip(unique_weights, packed_states)
    ]

    # Placement is a whole-model property: an ordinary (non-quantized)
    # parameter on CPU/disk/meta must fail even without an hf_device_map.
    parameter_devices = [
        str(getattr(parameter, "device", None)) for parameter in parameters
    ]
    non_cuda_parameter_devices = sorted(
        {
            device
            for device in parameter_devices
            if device.strip().lower().split(":", 1)[0] != "cuda"
        }
    )
    non_cuda_parameter_count = sum(
        1
        for device in parameter_devices
        if device.strip().lower().split(":", 1)[0] != "cuda"
    )

    device_map = getattr(model, "hf_device_map", None)
    normalized_map = None
    offload = {"cpu": [], "disk": [], "meta": []}
    if isinstance(device_map, dict):
        normalized_map = {str(name): str(device) for name, device in device_map.items()}
        for name, device in normalized_map.items():
            kind = device.strip().lower().split(":", 1)[0]
            if kind in offload:
                offload[kind].append(name)

    return {
        "requested_load_in_4bit": bool(requested_load_in_4bit),
        # Effective means real Linear4bit modules with packed Params4bit weights.
        "effective_load_in_4bit": bool(linear4 and packed_weights),
        "quant_type": _sorted_unique(quant_types),
        # module.compute_dtype only: the loaded Linear4bit carries the real one.
        "compute_dtype": _sorted_unique(
            str(getattr(module, "compute_dtype", None)) for module in linear4
        ),
        "double_quant": all(double_flags) if double_flags else None,
        "quantized_module_count": len(linear4),
        "quantized_parameter_count": params4_weight_count,
        "packed_parameter_count": len(packed_weights),
        "unpacked_parameter_count": unpacked_parameter_count,
        "non_params4_weight_count": non_params4_weights,
        "quantized_parameter_devices": _sorted_unique(
            getattr(parameter, "device", None) for parameter in params4
        )
        or [],
        "all_parameter_devices": _sorted_unique(parameter_devices) or [],
        "non_cuda_parameter_count": non_cuda_parameter_count,
        "non_cuda_parameter_devices": non_cuda_parameter_devices,
        "bitsandbytes_8bit_module_count": len(linear8),
        "bitsandbytes_available": bnb is not None,
        "bitsandbytes_version": getattr(bnb, "__version__", None),
        "hf_device_map": normalized_map,
        "offload": {key: sorted(names) for key, names in offload.items()},
    }


def assert_effective_quantization(
    model, requested_load_in_4bit: bool, context: str = "model"
) -> dict:
    """Raise when 4-bit was requested but the loaded model is not actually 4-bit.

    Fail-closed: every material setting (packed NF4 weights, BF16 compute,
    nested double quant, no 8-bit layers, CUDA-only placement) must be present
    and consistent. Returns the effective `quantization_stats` so callers
    record them instead of echoing the request. `requested_load_in_4bit=False`
    returns the stats unchanged, keeping the BF16 path as before.
    """
    stats = quantization_stats(model, requested_load_in_4bit)
    if not requested_load_in_4bit:
        return stats

    problems: list[str] = []
    if stats["quantized_module_count"] < 1:
        problems.append("no bitsandbytes Linear4bit modules found")
    if stats["quantized_parameter_count"] < 1:
        problems.append("no bitsandbytes Params4bit parameters found")
    if stats["non_params4_weight_count"]:
        problems.append(
            f"{stats['non_params4_weight_count']} Linear4bit weights are not Params4bit"
        )
    if stats["unpacked_parameter_count"]:
        problems.append(
            f"{stats['unpacked_parameter_count']} Params4bit parameters are not "
            "packed (bnb_quantized / quant_state missing)"
        )
    if stats["quant_type"] != [BNB_4BIT_QUANT_TYPE]:
        problems.append(
            f"quant type must be exactly {BNB_4BIT_QUANT_TYPE!r}, "
            f"found {stats['quant_type']}"
        )
    if stats["compute_dtype"] != [BF16_DTYPE_NAME]:
        problems.append(
            f"Linear4bit compute dtype must be exactly {BF16_DTYPE_NAME!r}, "
            f"found {stats['compute_dtype']}"
        )
    if stats["double_quant"] is not True:
        problems.append(
            "nested/double quantization must be present and consistent, "
            f"found {stats['double_quant']!r}"
        )
    if stats["bitsandbytes_8bit_module_count"]:
        problems.append(
            f"{stats['bitsandbytes_8bit_module_count']} Linear8bitLt modules present"
        )
    if stats["non_cuda_parameter_count"]:
        problems.append(
            f"{stats['non_cuda_parameter_count']} parameters are not on CUDA: "
            f"{stats['non_cuda_parameter_devices']}"
        )
    if not stats["effective_load_in_4bit"]:
        problems.append("no packed 4-bit weights detected")
    offloaded = {
        key: names for key, names in stats["offload"].items() if names
    }
    if offloaded:
        problems.append(
            f"device_map places weights on {sorted(offloaded)} "
            "(CPU/disk/meta fallback is not allowed for 4-bit runs)"
        )

    if problems:
        raise RuntimeError(
            f"requested a 4-bit load for {context}, but effective quantization "
            "validation failed: " + "; ".join(problems)
        )
    return stats


@dataclass
class LoadedModel:
    """A loaded base model (optionally + adapter) with its tokenizer/processor.

    `tokenizer` is always the underlying tokenizer (truncation, decode, EOS);
    `processor` is set only for vision models (prompt rendering + tokenization).
    `quantization` is the effective inspection metadata from `load_model`.
    """

    model: object
    tokenizer: object
    processor: object | None
    kind: str
    requested_revision: str | None = None
    quantization: dict | None = None

    @property
    def prompt_source(self):
        """Chat-template source: the processor for vision, else the tokenizer."""
        return self.processor if self.processor is not None else self.tokenizer

    def metadata(self) -> dict:
        metadata = {
            "model_kind": self.kind,
            **revision_metadata(
                self.requested_revision, self.model, self.tokenizer, self.processor
            ),
        }
        if self.quantization is not None:
            metadata["quantization"] = self.quantization
        return metadata


def _repo_kwargs(revision: str | None, trust_remote_code: bool) -> dict:
    """Repository/artifact kwargs for tokenizer, processor and model loads."""
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
    load_in_4bit: bool = False,
    bnb_quant_type: str = BNB_4BIT_QUANT_TYPE,
    bnb_compute_dtype: str = BNB_4BIT_COMPUTE_DTYPE,
    bnb_4bit_use_double_quant: bool = BNB_4BIT_DOUBLE_QUANT,
) -> LoadedModel:
    """Load a language or vision-conditional base model, optionally + PeftModel adapter.

    `model.kind: language` keeps AutoTokenizer + AutoModelForCausalLM;
    `vision` uses AutoProcessor + AutoModelForImageTextToText. When
    `load_in_4bit`, an explicit canonical `BitsAndBytesConfig` is passed to
    `from_pretrained` and the loaded modules are checked to really be 4-bit.

    Repository kwargs (revision/trust_remote_code) go to the tokenizer/processor
    as well; `quantization_config` is model-only and never reaches them.

    Default `load_in_4bit=False` keeps the previous BF16/`torch_dtype="auto"`
    behavior byte-for-byte.
    """
    if kind not in ("language", "vision"):
        raise ValueError(f"unknown model kind: {kind!r} (use 'language' or 'vision')")

    repo_kwargs = _repo_kwargs(revision, trust_remote_code)
    model_kwargs = {"torch_dtype": "auto", "device_map": "auto", **repo_kwargs}
    if load_in_4bit:
        model_kwargs["quantization_config"] = bitsandbytes_config(
            quant_type=bnb_quant_type,
            compute_dtype=bnb_compute_dtype,
            use_double_quant=bnb_4bit_use_double_quant,
        )

    if kind == "vision":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(base_model, **repo_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"processor for {base_model} exposes no tokenizer")
        model = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        processor = None
        tokenizer = AutoTokenizer.from_pretrained(base_model, **repo_kwargs)
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    # Assert on the final object (adapter included) and record what was found.
    quantization = assert_effective_quantization(
        model, load_in_4bit, context=f"{'adapter' if adapter else 'base'} {base_model}"
    )
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        kind=kind,
        requested_revision=revision,
        quantization=quantization,
    )


def tokenize_prompt(loaded: LoadedModel, prompt: str) -> dict:
    """Tokenize a rendered prompt: `processor(text=...)` for vision, tokenizer otherwise."""
    if loaded.processor is not None:
        return loaded.processor(text=prompt, return_tensors="pt")
    return loaded.tokenizer(prompt, return_tensors="pt")
