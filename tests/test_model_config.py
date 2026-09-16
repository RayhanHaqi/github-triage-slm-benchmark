"""Cheap tests for config-driven model selection, prompts and run metadata.

Mock-based: no torch/transformers/unsloth needed. Run with
`python -m unittest discover -s tests`.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specialist import cli  # noqa: E402
from specialist import model as model_mod  # noqa: E402
from specialist import train as train_mod  # noqa: E402

# Pin torch in sys.modules before any `mock.patch.dict(sys.modules, ...)` window:
# patch.dict restores by clearing/re-adding its snapshot, so a real module first
# imported inside a window would be dropped while its C state persists, making
# a later `import torch` re-execute module code and fail. Fake ML modules cover
# the imports inside the windows; this covers torch itself.
try:  # pragma: no cover - absent/broken ML stack just skips torch tests
    import torch  # noqa: F401
except Exception:
    pass

LABELS = ["bug", "feature-request"]


class FakeTokenizer:
    def __init__(self, eos_token="<|im_end|>"):
        self.eos_token = eos_token
        self.unk_token_id = 0
        self.encode_calls = []
        self.decode_calls = []
        self.template_call = None
        self._token_to_id = {"<|im_end|>": 100, "<|endoftext|>": 101}

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token)

    def convert_ids_to_tokens(self, token_id):
        return {100: "<|im_end|>", 101: "<|endoftext|>"}.get(token_id)

    def __call__(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        if kwargs.get("return_tensors"):
            return {"input_ids": [[10, 11, 12]]}
        return {"input_ids": [10, 11, 12]}

    def decode(self, ids, **kwargs):
        self.decode_calls.append((list(ids), kwargs))
        return "ISSUE"

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return "RENDERED"

    def save_pretrained(self, path, **kwargs):
        self.saved = path


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.template_call = None
        self.text_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return "RENDERED-P"

    def __call__(self, text=None, **kwargs):
        self.text_call = (text, kwargs)
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def save_pretrained(self, path, **kwargs):
        self.saved = path


class FakeQuantState:
    """Stand-in for `bitsandbytes.functional.QuantState` (packed 4-bit state)."""

    def __init__(self, nested=True, absmax=object(), code=object()):
        self.absmax = absmax
        self.code = code
        self.nested = nested
        self.state2 = FakeQuantState(nested=False) if nested else None
        self.quant_type = "nf4"
        self.dtype = "torch.bfloat16"


class FakeParams4bit:
    """Stand-in for `bitsandbytes.nn.Params4bit` (bnb 0.50.2 invariants)."""

    def __init__(
        self,
        device="cuda:0",
        compress_statistics=True,
        quant_type="nf4",
        bnb_quantized=True,
        quant_state=True,
    ):
        self.device = device
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.bnb_quantized = bnb_quantized
        if quant_state is True:
            self.quant_state = FakeQuantState(nested=compress_statistics)
        elif quant_state is False:
            self.quant_state = None
        else:
            self.quant_state = quant_state


class FakeLinear4bit:
    """Stand-in for `bitsandbytes.nn.Linear4bit`."""

    def __init__(self, weight=None, compute_dtype="torch.bfloat16"):
        self.weight = weight if weight is not None else FakeParams4bit()
        self.compute_dtype = compute_dtype
        self.quant_state = getattr(self.weight, "quant_state", None)


class FakeLinear8bitLt:
    """Stand-in for `bitsandbytes.nn.Linear8bitLt` (must not appear in 4-bit)."""


class QuantHolder:
    """Minimal Module-like holder for inspection tests."""

    def __init__(self, modules=(), parameters=(), device_map=None):
        self._modules = list(modules)
        self._parameters = list(parameters)
        self.hf_device_map = device_map

    def modules(self):
        return iter(self._modules)

    def parameters(self):
        return iter(self._parameters)


class FakeModel:
    def __init__(
        self,
        commit_hash=None,
        eos_token_id=None,
        quantized=False,
        device_map=None,
        also_8bit=False,
    ):
        self.config = types.SimpleNamespace(
            _commit_hash=commit_hash, eos_token_id=eos_token_id
        )
        self.eval_called = False
        self.saved = None
        quant_modules = [FakeLinear4bit()] if quantized else []
        if also_8bit:
            quant_modules.append(FakeLinear8bitLt())
        self._quant_modules = quant_modules
        self._quant_params = [
            module.weight for module in quant_modules if isinstance(module, FakeLinear4bit)
        ]
        self.hf_device_map = device_map

    def modules(self):
        return iter(self._quant_modules)

    def parameters(self):
        return iter(self._quant_params)

    def eval(self):
        self.eval_called = True
        return self

    def save_pretrained(self, path, **kwargs):
        self.saved = path


class FakeLoader:
    """Stand-in for an Auto* class: records from_pretrained calls."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def from_pretrained(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.result


class FakeBitsAndBytesConfig:
    """Stand-in for `transformers.BitsAndBytesConfig`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_transformers(
    commit_hash=None, quantized=False, device_map=None, also_8bit=False
):
    model = FakeModel(
        commit_hash, quantized=quantized, device_map=device_map, also_8bit=also_8bit
    )
    tokenizer = FakeTokenizer()
    processor = FakeProcessor()
    module = types.ModuleType("transformers")
    module.AutoTokenizer = FakeLoader(tokenizer)
    module.AutoModelForCausalLM = FakeLoader(model)
    module.AutoProcessor = FakeLoader(processor)
    module.AutoModelForImageTextToText = FakeLoader(model)
    module.BitsAndBytesConfig = FakeBitsAndBytesConfig
    return module, {"tokenizer": tokenizer, "processor": processor, "model": model}


def fake_bitsandbytes(version="0.50.2"):
    module = types.ModuleType("bitsandbytes")
    module.__version__ = version
    nn = types.ModuleType("bitsandbytes.nn")
    nn.Linear4bit = FakeLinear4bit
    nn.Params4bit = FakeParams4bit
    nn.Linear8bitLt = FakeLinear8bitLt
    module.nn = nn
    return module


def fake_peft():
    module = types.ModuleType("peft")

    class PeftModel:
        last_call = None

        @classmethod
        def from_pretrained(cls, model, adapter):
            cls.last_call = (model, adapter)
            return model

    module.PeftModel = PeftModel
    return module, PeftModel


class ChatTemplateKwargsTest(unittest.TestCase):
    def test_reference_default_when_key_absent(self):
        self.assertEqual(
            model_mod.configured_chat_template_kwargs(None),
            {"enable_thinking": False},
        )
        self.assertEqual(
            model_mod.configured_chat_template_kwargs({}),
            {"enable_thinking": False},
        )

    def test_explicit_mapping_wins(self):
        self.assertEqual(
            model_mod.configured_chat_template_kwargs({"chat_template_kwargs": {}}), {}
        )
        self.assertEqual(
            model_mod.configured_chat_template_kwargs(
                {"chat_template_kwargs": {"enable_thinking": True}}
            ),
            {"enable_thinking": True},
        )


class BuildPromptTest(unittest.TestCase):
    def test_language_uses_tokenizer_and_reference_kwargs(self):
        tokenizer = FakeTokenizer()
        prompt, count = model_mod.build_prompt(tokenizer, "issue text", "SYS", 123)

        self.assertEqual((prompt, count), ("RENDERED", 3))
        self.assertEqual(
            tokenizer.encode_calls[0],
            (
                "issue text",
                {"add_special_tokens": False, "truncation": True, "max_length": 123},
            ),
        )
        messages, kwargs = tokenizer.template_call
        self.assertEqual(messages[0], {"role": "system", "content": "SYS"})
        self.assertEqual(messages[1], {"role": "user", "content": "ISSUE"})
        self.assertEqual(
            kwargs,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_vision_truncates_via_processor_tokenizer_and_passes_kwargs(self):
        processor = FakeProcessor()
        prompt, _ = model_mod.build_prompt(
            processor, "issue text", "SYS", 99, chat_template_kwargs={}
        )

        self.assertEqual(prompt, "RENDERED-P")
        self.assertEqual(processor.tokenizer.encode_calls[0][1]["max_length"], 99)
        self.assertEqual(processor.tokenizer.decode_calls[0][0], [10, 11, 12])
        self.assertEqual(
            processor.template_call[1],
            {"tokenize": False, "add_generation_prompt": True},
        )

    def test_tokenize_prompt_language_vs_vision(self):
        tokenizer = FakeTokenizer()
        processor = FakeProcessor()
        language = model_mod.LoadedModel(tokenizer, tokenizer, None, "language")
        vision = model_mod.LoadedModel(processor, processor.tokenizer, processor, "vision")

        self.assertIn("input_ids", model_mod.tokenize_prompt(language, "P"))
        self.assertEqual(tokenizer.encode_calls[-1][0], "P")
        self.assertIn("input_ids", model_mod.tokenize_prompt(vision, "P"))
        self.assertEqual(processor.text_call, ("P", {"return_tensors": "pt"}))


def patched_ml_modules(transformers_mod, **extra):
    """Fake transformers/bitsandbytes for a mocked load window.

    The fake bitsandbytes keeps the real package (and torch state) from being
    imported inside a `mock.patch.dict(sys.modules, ...)` window.
    """
    modules = {"transformers": transformers_mod, "bitsandbytes": fake_bitsandbytes()}
    modules.update(extra)
    return mock.patch.dict(sys.modules, modules)


class LoadModelTest(unittest.TestCase):
    def test_language_selects_causal_auto_classes(self):
        transformers_mod, fakes = fake_transformers()
        with patched_ml_modules(transformers_mod):
            loaded = model_mod.load_model(
                "Qwen/Qwen3-1.7B",
                revision="abc123",
                kind="language",
                trust_remote_code=True,
            )

        tokenizer_kwargs = {"trust_remote_code": True, "revision": "abc123"}
        self.assertEqual(
            transformers_mod.AutoTokenizer.calls, [("Qwen/Qwen3-1.7B", tokenizer_kwargs)]
        )
        model_call = transformers_mod.AutoModelForCausalLM.calls[0]
        self.assertEqual(model_call[0], "Qwen/Qwen3-1.7B")
        self.assertEqual(
            model_call[1],
            {
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": True,
                "revision": "abc123",
            },
        )
        self.assertEqual(transformers_mod.AutoProcessor.calls, [])
        self.assertIsNone(loaded.processor)
        self.assertTrue(loaded.model.eval_called)
        meta = loaded.metadata()
        self.assertEqual(meta["model_kind"], "language")
        self.assertEqual(meta["resolved_revision"], "abc123")
        self.assertEqual(meta["resolved_revision_source"], "requested")

    def test_resolved_revision_from_model_config(self):
        transformers_mod, _ = fake_transformers(commit_hash="deadbeef")
        with patched_ml_modules(transformers_mod):
            loaded = model_mod.load_model("m", revision="abc123", kind="language")

        meta = loaded.metadata()
        self.assertEqual(meta["resolved_revision"], "deadbeef")
        self.assertEqual(meta["resolved_revision_source"], "model_config")

    def test_vision_selects_processor_and_image_text_class(self):
        transformers_mod, fakes = fake_transformers()
        with patched_ml_modules(transformers_mod):
            loaded = model_mod.load_model("LiquidAI/LFM2.5-1.2B-Instruct", kind="vision")

        self.assertEqual(
            transformers_mod.AutoProcessor.calls[0][0], "LiquidAI/LFM2.5-1.2B-Instruct"
        )
        self.assertEqual(
            transformers_mod.AutoModelForImageTextToText.calls[0][0],
            "LiquidAI/LFM2.5-1.2B-Instruct",
        )
        self.assertEqual(transformers_mod.AutoModelForCausalLM.calls, [])
        self.assertIs(loaded.tokenizer, fakes["processor"].tokenizer)
        self.assertIs(loaded.prompt_source, fakes["processor"])
        self.assertEqual(loaded.metadata()["model_kind"], "vision")

    def test_adapter_wraps_peft_model(self):
        transformers_mod, _ = fake_transformers()
        peft_mod, peft_cls = fake_peft()
        with patched_ml_modules(transformers_mod, peft=peft_mod):
            loaded = model_mod.load_model("m", adapter="/tmp/adapter", kind="language")

        self.assertEqual(peft_cls.last_call[1], "/tmp/adapter")
        self.assertIs(loaded.model, peft_cls.last_call[0])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            model_mod.load_model("m", kind="audio")

    def test_4bit_language_passes_config_to_model_only(self):
        transformers_mod, _ = fake_transformers(
            quantized=True, device_map={"a": "cuda:0", "b": "cuda:1"}
        )
        with patched_ml_modules(transformers_mod):
            loaded = model_mod.load_model(
                "m", revision="rev1", kind="language", load_in_4bit=True
            )

        repo_kwargs = {"trust_remote_code": False, "revision": "rev1"}
        self.assertEqual(
            transformers_mod.AutoTokenizer.calls, [("m", repo_kwargs)]
        )
        self.assertNotIn(
            "quantization_config", transformers_mod.AutoTokenizer.calls[0][1]
        )
        model_call_kwargs = transformers_mod.AutoModelForCausalLM.calls[0][1]
        qconfig = model_call_kwargs["quantization_config"]
        self.assertIsInstance(qconfig, transformers_mod.BitsAndBytesConfig)
        self.assertEqual(
            qconfig.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
        )
        self.assertNotIn("load_in_4bit", model_call_kwargs)

        quantization = loaded.metadata()["quantization"]
        self.assertTrue(quantization["requested_load_in_4bit"])
        self.assertTrue(quantization["effective_load_in_4bit"])
        self.assertEqual(quantization["quantized_module_count"], 1)
        self.assertEqual(quantization["quantized_parameter_count"], 1)
        self.assertEqual(quantization["packed_parameter_count"], 1)
        self.assertEqual(quantization["unpacked_parameter_count"], 0)
        self.assertEqual(quantization["quant_type"], ["nf4"])
        self.assertEqual(quantization["compute_dtype"], ["torch.bfloat16"])
        self.assertTrue(quantization["double_quant"])
        self.assertEqual(quantization["quantized_parameter_devices"], ["cuda:0"])
        self.assertEqual(
            quantization["offload"], {"cpu": [], "disk": [], "meta": []}
        )
        self.assertEqual(quantization["bitsandbytes_version"], "0.50.2")

    def test_4bit_vision_passes_config_to_model_only(self):
        transformers_mod, fakes = fake_transformers(
            quantized=True, device_map={"a": "cuda:0"}
        )
        with patched_ml_modules(transformers_mod):
            loaded = model_mod.load_model("m", kind="vision", load_in_4bit=True)

        processor_kwargs = transformers_mod.AutoProcessor.calls[0][1]
        self.assertNotIn("quantization_config", processor_kwargs)
        model_call_kwargs = transformers_mod.AutoModelForImageTextToText.calls[0][1]
        self.assertIn("quantization_config", model_call_kwargs)
        self.assertNotIn(
            "quantization_config", transformers_mod.AutoModelForCausalLM.calls
        )
        self.assertTrue(loaded.metadata()["quantization"]["effective_load_in_4bit"])
        self.assertIs(loaded.processor, fakes["processor"])

    def test_4bit_load_without_quantized_modules_raises(self):
        transformers_mod, _ = fake_transformers(quantized=False)
        with patched_ml_modules(transformers_mod):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.load_model("Qwen/Qwen3-1.7B", kind="language", load_in_4bit=True)

        message = str(ctx.exception)
        self.assertIn("no bitsandbytes Linear4bit modules", message)
        self.assertIn("Qwen/Qwen3-1.7B", message)

    def test_4bit_offloaded_device_map_raises(self):
        for device in ("cpu", "disk", "meta"):
            transformers_mod, _ = fake_transformers(
                quantized=True, device_map={"layer": device}
            )
            with self.subTest(device=device), patched_ml_modules(transformers_mod):
                with self.assertRaises(RuntimeError) as ctx:
                    model_mod.load_model("m", kind="language", load_in_4bit=True)
                self.assertIn("CPU/disk/meta fallback", str(ctx.exception))

    def test_4bit_with_eight_bit_layers_raises(self):
        transformers_mod, _ = fake_transformers(quantized=True, also_8bit=True)
        with patched_ml_modules(transformers_mod):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.load_model("m", kind="language", load_in_4bit=True)

        self.assertIn("Linear8bitLt", str(ctx.exception))

    def test_4bit_adapter_path_asserts_after_peft_wrap(self):
        transformers_mod, _ = fake_transformers(quantized=True)
        peft_mod, peft_cls = fake_peft()
        with patched_ml_modules(transformers_mod, peft=peft_mod):
            loaded = model_mod.load_model(
                "m", adapter="/tmp/adapter", kind="language", load_in_4bit=True
            )

        self.assertTrue(loaded.metadata()["quantization"]["effective_load_in_4bit"])
        self.assertIs(loaded.model, peft_cls.last_call[0])


class BitsAndBytesConfigTest(unittest.TestCase):
    def test_canonical_arguments(self):
        transformers_mod, _ = fake_transformers()
        with patched_ml_modules(transformers_mod):
            config = model_mod.bitsandbytes_config()

        self.assertIsInstance(config, transformers_mod.BitsAndBytesConfig)
        self.assertEqual(
            config.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
        )


def fake_quantized_holder(**weight_kwargs):
    """One valid packed Linear4bit module/parameter pair on CUDA."""
    weight = FakeParams4bit(**weight_kwargs)
    return QuantHolder([FakeLinear4bit(weight=weight)], [weight], {"layer": "cuda:0"})


class QuantizationInspectionTest(unittest.TestCase):
    def _assert_fails(self, holder, *needles):
        with mock.patch.dict(sys.modules, {"bitsandbytes": fake_bitsandbytes()}):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.assert_effective_quantization(holder, True, context="test model")
        message = str(ctx.exception)
        self.assertIn("test model", message)
        for needle in needles:
            self.assertIn(needle, message)
        return message

    def test_effective_stats_from_packed_model(self):
        holder = fake_quantized_holder()
        with mock.patch.dict(sys.modules, {"bitsandbytes": fake_bitsandbytes()}):
            stats = model_mod.assert_effective_quantization(holder, True, context="test")

        self.assertTrue(stats["effective_load_in_4bit"])
        self.assertEqual(stats["quantized_module_count"], 1)
        self.assertEqual(stats["quantized_parameter_count"], 1)
        self.assertEqual(stats["packed_parameter_count"], 1)
        self.assertEqual(stats["unpacked_parameter_count"], 0)
        self.assertEqual(stats["quant_type"], ["nf4"])
        self.assertEqual(stats["compute_dtype"], ["torch.bfloat16"])
        self.assertTrue(stats["double_quant"])
        self.assertEqual(stats["quantized_parameter_devices"], ["cuda:0"])
        self.assertEqual(stats["all_parameter_devices"], ["cuda:0"])
        self.assertEqual(stats["non_cuda_parameter_count"], 0)
        self.assertEqual(stats["offload"], {"cpu": [], "disk": [], "meta": []})

    def test_tied_weights_counted_once(self):
        shared = FakeParams4bit()
        holder = QuantHolder(
            [FakeLinear4bit(weight=shared), FakeLinear4bit(weight=shared)],
            [shared],
            {"l": "cuda:0"},
        )
        with mock.patch.dict(sys.modules, {"bitsandbytes": fake_bitsandbytes()}):
            stats = model_mod.assert_effective_quantization(holder, True, context="tied")

        self.assertEqual(stats["quantized_module_count"], 2)
        self.assertEqual(stats["quantized_parameter_count"], 1)
        self.assertEqual(stats["packed_parameter_count"], 1)
        self.assertEqual(stats["unpacked_parameter_count"], 0)
        self.assertEqual(stats["non_params4_weight_count"], 0)

    def test_requested_4bit_without_quantized_modules_raises(self):
        message = self._assert_fails(QuantHolder(), "no bitsandbytes Linear4bit modules")
        self.assertIn("no bitsandbytes Params4bit parameters", message)

    def test_bitsandbytes_unavailable_raises(self):
        with mock.patch.dict(sys.modules, {"bitsandbytes": None}):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.assert_effective_quantization(
                    fake_quantized_holder(), True, context="baseline m"
                )
        self.assertIn("bitsandbytes is not importable", str(ctx.exception))

    def test_offloaded_device_map_rejected(self):
        for device in ("cpu", "disk", "meta"):
            holder = fake_quantized_holder()
            holder.hf_device_map = {"layer": device}
            with self.subTest(device=device):
                self._assert_fails(holder, "CPU/disk/meta fallback")

    def test_non_cuda_parameter_device_rejected(self):
        for device in ("cpu", "meta"):
            holder = fake_quantized_holder(device=device)
            with self.subTest(device=device):
                self._assert_fails(holder, "not on CUDA")

    def test_ordinary_non_cuda_parameter_rejected_without_device_map(self):
        # Valid packed CUDA NF4 plus one ordinary CPU/meta parameter and no
        # hf_device_map: whole-model placement must still fail closed.
        weight = FakeParams4bit()
        for device in ("cpu", "meta", "disk"):
            ordinary = types.SimpleNamespace(device=device)
            holder = QuantHolder(
                [FakeLinear4bit(weight=weight)], [weight, ordinary], None
            )
            with self.subTest(device=device):
                self.assertIsNone(holder.hf_device_map)
                self._assert_fails(holder, "not on CUDA", "'" + device + "'")

    def test_wrong_quant_type_rejected(self):
        weight = FakeParams4bit(quant_type="fp4")
        weight.quant_state.quant_type = "fp4"
        holder = QuantHolder([FakeLinear4bit(weight=weight)], [weight], {"l": "cuda:0"})
        self._assert_fails(holder, "quant type must be exactly 'nf4'", "'fp4'")

    def test_wrong_compute_dtype_rejected(self):
        for dtype in ("torch.float16", None):
            weight = FakeParams4bit()
            holder = QuantHolder(
                [FakeLinear4bit(weight=weight, compute_dtype=dtype)],
                [weight],
                {"l": "cuda:0"},
            )
            with self.subTest(dtype=dtype):
                self._assert_fails(holder, "compute dtype must be exactly")

    def test_missing_double_quant_rejected(self):
        # compress_statistics off => nested False; then a state without state2.
        weight = FakeParams4bit(compress_statistics=False)
        holder = QuantHolder([FakeLinear4bit(weight=weight)], [weight], {"l": "cuda:0"})
        self._assert_fails(holder, "double quantization must be present")

        nested_missing = FakeParams4bit()
        nested_missing.quant_state.state2 = None
        nested_missing.quant_state.nested = True
        holder = QuantHolder(
            [FakeLinear4bit(weight=nested_missing)], [nested_missing], {"l": "cuda:0"}
        )
        self._assert_fails(holder, "double quantization must be present")

    def test_unpacked_params_rejected(self):
        for kwargs in (
            {"bnb_quantized": False},
            {"quant_state": False},
        ):
            weight = FakeParams4bit(**kwargs)
            holder = QuantHolder(
                [FakeLinear4bit(weight=weight)], [weight], {"l": "cuda:0"}
            )
            with self.subTest(**kwargs):
                self._assert_fails(holder, "not packed")

        no_state = FakeParams4bit(quant_state=False)
        no_state.quant_state = FakeQuantState(absmax=None)
        holder = QuantHolder(
            [FakeLinear4bit(weight=no_state)], [no_state], {"l": "cuda:0"}
        )
        self._assert_fails(holder, "not packed")

    def test_partial_evidence_rejected(self):
        # Linear4bit weights that are not Params4bit at all.
        holder = QuantHolder(
            [FakeLinear4bit(weight=object())], [FakeParams4bit()], {"l": "cuda:0"}
        )
        self._assert_fails(holder, "are not Params4bit")

        # Mixed packed and unpacked weights: identity-deduplicated counts must
        # still expose the unpacked one.
        packed = FakeParams4bit()
        unpacked = FakeParams4bit(bnb_quantized=False)
        holder = QuantHolder(
            [FakeLinear4bit(weight=packed), FakeLinear4bit(weight=unpacked)],
            [packed, unpacked],
            {"l": "cuda:0"},
        )
        self._assert_fails(holder, "not packed")

    def test_eight_bit_layers_rejected(self):
        holder = QuantHolder(
            [FakeLinear4bit(), FakeLinear8bitLt()],
            [FakeParams4bit()],
            {"l": "cuda:0"},
        )
        self._assert_fails(holder, "Linear8bitLt")

    def test_traversal_error_propagates(self):
        class BrokenHolder(QuantHolder):
            def modules(self):
                raise RuntimeError("boom during traversal")

        with mock.patch.dict(sys.modules, {"bitsandbytes": fake_bitsandbytes()}):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.assert_effective_quantization(
                    BrokenHolder(), True, context="test model"
                )
        self.assertIn("boom during traversal", str(ctx.exception))

    def test_non_module_requested_4bit_raises(self):
        with mock.patch.dict(sys.modules, {"bitsandbytes": fake_bitsandbytes()}):
            with self.assertRaises(RuntimeError) as ctx:
                model_mod.assert_effective_quantization(
                    types.SimpleNamespace(), True, context="test model"
                )
        self.assertIn("modules()/parameters()", str(ctx.exception))

    def test_unrequested_4bit_does_not_raise(self):
        stats = model_mod.assert_effective_quantization(FakeModel(), False)
        self.assertFalse(stats["requested_load_in_4bit"])
        self.assertFalse(stats["effective_load_in_4bit"])

        # Non-Module objects keep the legacy permissive BF16 path.
        stats = model_mod.assert_effective_quantization(types.SimpleNamespace(), False)
        self.assertFalse(stats["effective_load_in_4bit"])


class TrainHelpersTest(unittest.TestCase):
    LORA = {
        "r": 16,
        "alpha": 32,
        "dropout": 0,
        "target_modules": "all-linear",
        "bias": "none",
        "gradient_checkpointing": "unsloth",
        "random_state": 42,
    }

    def test_target_module_type_preserved(self):
        self.assertEqual(train_mod.normalize_target_modules("all-linear"), "all-linear")
        self.assertEqual(
            train_mod.normalize_target_modules(["q_proj", "k_proj"]),
            ["q_proj", "k_proj"],
        )

    def test_peft_kwargs_vision_flags_defaults(self):
        kwargs = train_mod.peft_kwargs(self.LORA, "vision")
        self.assertEqual(kwargs["target_modules"], "all-linear")
        self.assertFalse(kwargs["finetune_vision_layers"])
        self.assertTrue(kwargs["finetune_language_layers"])
        self.assertTrue(kwargs["finetune_attention_modules"])
        self.assertTrue(kwargs["finetune_mlp_modules"])

    def test_peft_kwargs_language_has_no_vision_flags(self):
        kwargs = train_mod.peft_kwargs(self.LORA, "language")
        self.assertNotIn("finetune_vision_layers", kwargs)
        self.assertEqual(kwargs["lora_alpha"], 32)

    def test_peft_kwargs_for_peft_call_maps_only_vision_all_linear(self):
        call_kwargs = train_mod.peft_kwargs(self.LORA, "vision", for_peft_call=True)
        self.assertIsNone(call_kwargs["target_modules"])
        self.assertFalse(call_kwargs["finetune_vision_layers"])
        self.assertTrue(call_kwargs["finetune_language_layers"])
        self.assertTrue(call_kwargs["finetune_attention_modules"])
        self.assertTrue(call_kwargs["finetune_mlp_modules"])
        # requested value stays in the default (metrics) form
        self.assertEqual(
            train_mod.peft_kwargs(self.LORA, "vision")["target_modules"], "all-linear"
        )
        # language path is never remapped
        self.assertEqual(
            train_mod.peft_kwargs(
                self.LORA, "language", for_peft_call=True
            )["target_modules"],
            "all-linear",
        )
        # explicit module lists are passed through untouched
        explicit = dict(self.LORA, target_modules=["q_proj", "v_proj"])
        self.assertEqual(
            train_mod.peft_kwargs(
                explicit, "vision", for_peft_call=True
            )["target_modules"],
            ["q_proj", "v_proj"],
        )

    def test_library_versions_json_serializable(self):
        versions = model_mod.library_versions()
        self.assertIn("python", versions)
        self.assertIn("bitsandbytes", versions)
        json.dumps(versions)


class ResolveEosTokenTest(unittest.TestCase):
    def test_valid_tokenizer_eos_returned(self):
        self.assertEqual(train_mod.resolve_eos_token(FakeTokenizer()), "<|im_end|>")

    def test_redaction_placeholder_never_returned(self):
        # transformers 5.x to_dict() rewrites eos_token=None to "<EOS_TOKEN>".
        tokenizer = FakeTokenizer(eos_token="<EOS_TOKEN>")
        self.assertIsNone(train_mod.resolve_eos_token(tokenizer))
        processor = FakeProcessor()
        processor.tokenizer = FakeTokenizer(eos_token="<EOS_TOKEN>")
        self.assertIsNone(train_mod.resolve_eos_token(processor))

    def test_model_config_eos_id_fallback(self):
        tokenizer = FakeTokenizer(eos_token=None)
        self.assertEqual(
            train_mod.resolve_eos_token(tokenizer, FakeModel(eos_token_id=101)),
            "<|endoftext|>",
        )
        self.assertEqual(
            train_mod.resolve_eos_token(tokenizer, FakeModel(eos_token_id=[999, 100])),
            "<|im_end|>",
        )
        self.assertIsNone(train_mod.resolve_eos_token(tokenizer, FakeModel(eos_token_id=999)))
        self.assertIsNone(train_mod.resolve_eos_token(tokenizer, None))

    def test_processor_uses_underlying_tokenizer(self):
        self.assertEqual(
            train_mod.resolve_eos_token(FakeProcessor(), FakeModel()), "<|im_end|>"
        )


class TrainImportOrderTest(unittest.TestCase):
    def test_unsloth_bound_before_trl(self):
        """Unsloth must patch trl first, else TRL 0.24 SFTConfig/SFTTrainer checks break."""
        source = inspect.getsource(train_mod.train)
        self.assertLess(source.index("from unsloth"), source.index("from trl"))


class TrainPathSelectionTest(unittest.TestCase):
    LORA = {
        "r": 16,
        "alpha": 32,
        "dropout": 0,
        "target_modules": "all-linear",
        "bias": "none",
        "gradient_checkpointing": "unsloth",
        "random_state": 42,
    }
    TRAINING = {
        "output_subdir": "trainer",
        "adapter_subdir": "adapter",
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "report_to": "none",
    }

    def _workspace(self):
        tmp = tempfile.TemporaryDirectory()
        ws = Path(tmp.name)
        for name in ("train", "val", "test"):
            row = {"input": "issue text", "label": "bug", "issue_number": 1}
            (ws / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return tmp, ws

    def _config(self, kind="language", **model_overrides):
        return {
            "model": {
                "base_model": "m",
                "system_prompt": "SYS",
                "max_seq_length": 2048,
                "load_in_4bit": False,
                "kind": kind,
                **model_overrides,
            },
            "lora": dict(self.LORA),
            "training": dict(self.TRAINING),
            "evaluation": {},
        }

    def _run(self, config, workspace, state_overrides=None, events=None):
        dataset_calls = []
        trainer_holders = {}
        events = events if events is not None else []
        canonical_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }

        def quantization_requested(kwargs):
            """Unsloth mock only turns 4-bit when the explicit config is valid."""
            qconfig = kwargs.get("quantization_config")
            return bool(
                kwargs.get("load_in_4bit")
                and isinstance(qconfig, FakeBitsAndBytesConfig)
                and qconfig.kwargs == canonical_config
            )

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSFTTrainer:
            def __init__(self, **kwargs):
                trainer_holders["kwargs"] = kwargs
                state = {
                    "log_history": [],
                    "best_model_checkpoint": None,
                    "best_metric": None,
                }
                state.update(state_overrides or {})
                self.state = types.SimpleNamespace(**state)

            def train(self, **kwargs):
                events.append("train")
                return types.SimpleNamespace(
                    metrics={"train_runtime": 1.0, "epoch": 3.0}
                )

        class FakeCollator:
            def __init__(self, model, processor, **kwargs):
                self.model = model
                self.processor = processor
                self.kwargs = kwargs

        tokenizer = FakeTokenizer()
        processor = FakeProcessor()

        class RecordingModel(FakeModel):
            def save_pretrained(self, path, **kwargs):
                events.append("save")
                return super().save_pretrained(path, **kwargs)

        def make_fast_model(processor_result):
            return type(
                "FakeFastModel",
                (),
                {
                    "from_pretrained": classmethod(
                        lambda c, **kw: (
                            setattr(c, "last_load", kw),
                            (
                                RecordingModel(
                                    "sha-abc",
                                    quantized=quantization_requested(kw),
                                ),
                                processor_result,
                            ),
                        )[1]
                    ),
                    "get_peft_model": classmethod(
                        lambda c, model, **kw: (setattr(c, "last_peft", kw), model)[1]
                    ),
                },
            )

        unsloth_mod = types.ModuleType("unsloth")
        unsloth_mod.FastLanguageModel = make_fast_model(tokenizer)
        unsloth_mod.FastVisionModel = make_fast_model(processor)
        unsloth_trainer_mod = types.ModuleType("unsloth.trainer")
        collators = []

        def make_collator(model, proc, **kwargs):
            collator = FakeCollator(model, proc, **kwargs)
            collators.append(collator)
            return collator

        unsloth_trainer_mod.UnslothVisionDataCollator = make_collator
        unsloth_mod.trainer = unsloth_trainer_mod

        datasets_mod = types.ModuleType("datasets")
        datasets_mod.Dataset = types.SimpleNamespace(
            from_list=lambda rows: (dataset_calls.append(rows), rows)[1]
        )
        trl_mod = types.ModuleType("trl")
        trl_mod.SFTConfig = FakeSFTConfig
        trl_mod.SFTTrainer = FakeSFTTrainer

        modules = {
            "datasets": datasets_mod,
            "trl": trl_mod,
            "unsloth": unsloth_mod,
            "unsloth.trainer": unsloth_trainer_mod,
            # Explicit config construction + effective inspection touch these.
            "transformers": fake_transformers()[0],
            "bitsandbytes": fake_bitsandbytes(),
        }
        with mock.patch.dict(sys.modules, modules):
            metrics = train_mod.train(config, workspace)

        return (
            metrics,
            trainer_holders["kwargs"],
            dataset_calls,
            collators,
            (tokenizer, processor),
            (unsloth_mod.FastLanguageModel, unsloth_mod.FastVisionModel),
        )

    def test_language_path_uses_trl24_max_length_and_no_eos_override(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, kwargs, dataset_calls, collators, (tokenizer, _), _ = self._run(
                self._config(), ws
            )

        self.assertIs(kwargs["processing_class"], tokenizer)
        self.assertNotIn("data_collator", kwargs)
        self.assertEqual(kwargs["args"].kwargs["max_length"], 2048)
        self.assertEqual(kwargs["args"].kwargs["dataset_text_field"], "text")
        self.assertEqual(kwargs["args"].kwargs["seed"], 42)
        self.assertNotIn("max_seq_length", kwargs["args"].kwargs)
        self.assertNotIn("eos_token", kwargs["args"].kwargs)
        self.assertIn("text", dataset_calls[0][0])
        self.assertEqual(collators, [])

    def test_vision_text_only_path_uses_max_length_and_tokenizer_eos(self):
        tmp, ws = self._workspace()
        with tmp:
            _, kwargs, dataset_calls, collators, (_, processor), _ = self._run(
                self._config(kind="vision", revision="9c1f0d2e"), ws
            )

        self.assertIs(kwargs["processing_class"], processor)
        self.assertNotIn("data_collator", kwargs)
        self.assertEqual(kwargs["args"].kwargs["max_length"], 2048)
        self.assertEqual(kwargs["args"].kwargs["dataset_text_field"], "text")
        self.assertNotIn("max_seq_length", kwargs["args"].kwargs)
        self.assertEqual(kwargs["args"].kwargs["eos_token"], "<|im_end|>")
        self.assertIn("text", dataset_calls[0][0])
        self.assertEqual(collators, [])

    def test_vision_all_linear_call_site_passes_none_with_explicit_flags(self):
        tmp, ws = self._workspace()
        with tmp:
            _, _, _, _, _, (flm, fvm) = self._run(
                self._config(kind="vision", revision="9c1f0d2e"), ws
            )

        # The literal "all-linear" makes Unsloth 2026.9.4 FastVisionModel
        # force every finetune_* flag True; the call site must pass None so the
        # explicit flags below are honored through get_peft_regex.
        self.assertIsNone(fvm.last_peft["target_modules"])
        self.assertFalse(fvm.last_peft["finetune_vision_layers"])
        self.assertTrue(fvm.last_peft["finetune_language_layers"])
        self.assertTrue(fvm.last_peft["finetune_attention_modules"])
        self.assertTrue(fvm.last_peft["finetune_mlp_modules"])
        self.assertFalse(hasattr(flm, "last_peft"))

    def test_language_all_linear_call_site_keeps_literal(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, _, _, _, _, (flm, fvm) = self._run(self._config(), ws)

        self.assertEqual(flm.last_peft["target_modules"], "all-linear")
        self.assertNotIn("finetune_vision_layers", flm.last_peft)
        self.assertFalse(hasattr(fvm, "last_peft"))
        # recorded metrics keep the requested value
        self.assertEqual(metrics["lora"]["target_modules"], "all-linear")

    def test_vision_collator_path_uses_message_rows(self):
        config = self._config(kind="vision")
        config["training"]["vision_collator"] = True
        tmp, ws = self._workspace()
        with tmp:
            metrics, kwargs, dataset_calls, collators, (_, processor), _ = self._run(
                config, ws
            )

        self.assertIs(kwargs["processing_class"], processor)
        args = kwargs["args"].kwargs
        self.assertEqual(args["max_length"], 2048)
        self.assertEqual(args["dataset_text_field"], "")
        self.assertFalse(args["remove_unused_columns"])
        self.assertEqual(args["dataset_kwargs"], {"skip_prepare_dataset": True})
        self.assertEqual(args["eos_token"], "<|im_end|>")
        for key in ("packing", "padding_free", "assistant_only_loss"):
            self.assertNotIn(key, args)
        self.assertIn("messages", dataset_calls[0][0])
        self.assertEqual(len(collators), 1)
        self.assertEqual(collators[0].kwargs["max_seq_length"], 2048)

    def test_metrics_are_json_serializable_and_complete(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, _, _, _, _, _ = self._run(
                self._config(kind="vision", revision="9c1f0d2e"), ws
            )
            train_sha = model_mod.sha256_file(ws / "train.jsonl")

        json.dumps(metrics)
        self.assertEqual(metrics["model_kind"], "vision")
        self.assertEqual(metrics["requested_revision"], "9c1f0d2e")
        self.assertEqual(metrics["resolved_revision"], "sha-abc")
        self.assertEqual(metrics["resolved_revision_source"], "model_config")
        self.assertEqual(metrics["sha256"]["train"], train_sha)
        self.assertEqual(metrics["seed"], 42)
        self.assertEqual(
            metrics["precision"], {"bf16": True, "fp16": False, "load_in_4bit": False}
        )
        self.assertEqual(metrics["lora"]["target_modules"], "all-linear")
        self.assertFalse(metrics["lora"]["finetune_vision_layers"])
        self.assertIsInstance(metrics["wall_train_seconds"], float)
        self.assertIn("train_metrics", metrics)
        # Effective quantization is inspected, not echoed from the config.
        self.assertFalse(metrics["quantization"]["effective_load_in_4bit"])
        self.assertFalse(metrics["quantization_after_load"]["effective_load_in_4bit"])
        self.assertEqual(metrics["num_train_epochs"], 3.0)
        self.assertIsNone(metrics["best_model_checkpoint"])
        self.assertIsNone(metrics["best_metric"])
        self.assertIsNone(metrics["best_epoch"])
        self.assertIsNone(metrics["final_eval_loss"])
        self.assertEqual(metrics["final_epoch"], 3.0)
        self.assertIsNone(metrics["final_eval_epoch"])
        self.assertGreaterEqual(metrics["train_peak_allocated_gib"], 0.0)
        self.assertGreaterEqual(metrics["train_peak_reserved_gib"], 0.0)

    def test_non_4bit_training_never_passes_quantization_config(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, _, _, _, _, (flm, _) = self._run(self._config(), ws)

        self.assertFalse(flm.last_load["load_in_4bit"])
        self.assertNotIn("quantization_config", flm.last_load)
        self.assertNotIn("offload_embedding", flm.last_load)
        self.assertFalse(metrics["quantization"]["effective_load_in_4bit"])

    def test_best_checkpoint_fields_passed_and_recorded(self):
        config = self._config()
        config["training"].update(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        log_history = [
            {"step": 100, "epoch": 1.0, "eval_loss": 0.5},
            {"step": 200, "epoch": 2.0, "eval_loss": 0.4},
        ]
        events = []
        tmp, ws = self._workspace()
        with tmp:
            checkpoint_dir = ws / "trainer" / "checkpoint-200"
            checkpoint_dir.mkdir(parents=True)
            metrics, kwargs, _, _, _, _ = self._run(
                config,
                ws,
                state_overrides={
                    "log_history": log_history,
                    "best_model_checkpoint": str(checkpoint_dir),
                    "best_metric": 0.4,
                },
                events=events,
            )

        args = kwargs["args"].kwargs
        self.assertTrue(args["load_best_model_at_end"])
        self.assertEqual(args["metric_for_best_model"], "eval_loss")
        self.assertFalse(args["greater_is_better"])
        self.assertEqual(metrics["num_train_epochs"], 3.0)
        self.assertEqual(metrics["best_model_checkpoint"], str(checkpoint_dir))
        self.assertEqual(metrics["best_metric"], 0.4)
        self.assertEqual(metrics["best_epoch"], 2.0)
        self.assertEqual(metrics["final_epoch"], 3.0)
        self.assertEqual(metrics["final_eval_epoch"], 2.0)
        self.assertEqual(metrics["final_eval_loss"], 0.4)
        # Validation and save happen after the trainer restored the best model.
        self.assertEqual(events, ["train", "save"])

    def test_best_checkpoint_validation_fails_before_save(self):
        config = self._config()
        config["training"].update(load_best_model_at_end=True)

        tmp, ws = self._workspace()
        with tmp:
            checkpoint_dir = ws / "trainer" / "checkpoint-200"
            checkpoint_dir.mkdir(parents=True)
            cases = {
                "missing checkpoint": (
                    {"best_model_checkpoint": None, "best_metric": 0.4},
                    "no best_model_checkpoint",
                ),
                "missing metric": (
                    {"best_model_checkpoint": str(checkpoint_dir), "best_metric": None},
                    "best_metric is not numeric",
                ),
                "non-finite metric": (
                    {
                        "best_model_checkpoint": str(checkpoint_dir),
                        "best_metric": float("nan"),
                    },
                    "best_metric is not finite",
                ),
                "missing directory": (
                    {"best_model_checkpoint": str(checkpoint_dir) + "-gone", "best_metric": 0.4},
                    "does not exist",
                ),
            }
            for label, (overrides, needle) in cases.items():
                events = []
                with self.subTest(label=label):
                    with self.assertRaises(RuntimeError) as ctx:
                        self._run(
                            config, ws, state_overrides=overrides, events=events
                        )
                    self.assertIn(needle, str(ctx.exception))
                    # No adapter bytes written on failure.
                    self.assertEqual(events, ["train"])

    def test_4bit_training_passes_explicit_config_and_asserts_effective(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, _, _, _, _, (flm, _) = self._run(
                self._config(load_in_4bit=True), ws
            )

        load_kwargs = flm.last_load
        self.assertTrue(load_kwargs["load_in_4bit"])
        qconfig = load_kwargs["quantization_config"]
        self.assertEqual(
            qconfig.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
        )
        self.assertIs(load_kwargs["offload_embedding"], False)
        self.assertTrue(metrics["quantization"]["effective_load_in_4bit"])
        self.assertTrue(metrics["quantization_after_load"]["effective_load_in_4bit"])
        self.assertEqual(metrics["quantization"]["quantized_module_count"], 1)
        self.assertEqual(metrics["quantization"]["packed_parameter_count"], 1)
        self.assertEqual(metrics["quantization"]["quant_type"], ["nf4"])
        self.assertTrue(metrics["precision"]["load_in_4bit"])


class ConfigParsingTest(unittest.TestCase):
    def test_new_optional_fields_parse_and_paths_expand(self):
        config_text = """
paths:
  data_dir: ~/slm-test-data
model:
  base_model: Qwen/Qwen3.5-4B
  kind: vision
  revision: 9c1f0d2e
  trust_remote_code: true
lora:
  target_modules: all-linear
  finetune_vision_layers: false
evaluation:
  chat_template_kwargs: {}
training:
  vision_collator: true
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(config_text, encoding="utf-8")
            config = cli.load_config(path)

        self.assertEqual(config["model"]["kind"], "vision")
        self.assertTrue(config["model"]["trust_remote_code"])
        self.assertEqual(config["lora"]["target_modules"], "all-linear")
        self.assertEqual(config["evaluation"]["chat_template_kwargs"], {})
        self.assertTrue(config["training"]["vision_collator"])
        self.assertTrue(str(config["paths"]["data_dir"]).startswith(str(Path.home())))

    def test_bundled_config_defaults_unchanged(self):
        config = cli.load_config(ROOT / "configs" / "vscode-bug-feature.yaml")
        self.assertEqual(config["model"].get("kind", "language"), "language")
        self.assertNotIn("chat_template_kwargs", config["evaluation"])
        self.assertFalse(config["model"].get("trust_remote_code", False))


class ComparisonIdentityTest(unittest.TestCase):
    """comparison.json must reject base identity/revision/quantization drift."""

    QUANT = {
        "requested_load_in_4bit": True,
        "effective_load_in_4bit": True,
        "quant_type": ["nf4"],
        "compute_dtype": ["torch.bfloat16"],
        "double_quant": True,
        "quantized_parameter_devices": ["cuda:0"],
        "offload": {"cpu": [], "disk": [], "meta": []},
    }

    def _metrics(self, **overrides):
        metrics = {
            "checkpoint": "base",
            "mode": "base",
            "adapter": None,
            "test_sha256": "abc",
            "split_path": "/data/test.jsonl",
            "split": "test",
            "labels": LABELS,
            "base_model": "m",
            "model_kind": "language",
            "requested_revision": "rev1",
            "resolved_revision": "rev1",
            "resolved_revision_source": "requested",
            "strict_accuracy": 0.5,
            "semantic_accuracy": 0.6,
            "valid_output_rate": 1.0,
            "per_class_recall": {"bug": 1.0, "feature-request": 1.0},
            "per_class_support": {"bug": 1, "feature-request": 1},
            "confusion": {},
            "predicted_counts": {},
            "mean_latency_ms": 1.0,
            "median_latency_ms": 1.0,
            "issues_per_second": 1.0,
            "output_tokens_per_second": 1.0,
            "total_generation_time_s": 1.0,
            "peak_allocated_gb": 1.0,
            "peak_reserved_gb": 1.0,
            "mean_input_tokens": 10.0,
            "median_input_tokens": 10.0,
            "total_generated_tokens": 10,
        }
        metrics.update(overrides)
        return metrics

    def _adapter(self, **overrides):
        return self._metrics(
            mode="adapter", checkpoint="/x/adapter", adapter="/x/adapter", **overrides
        )

    def _compare(self, baseline, finetuned):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "baseline_metrics.json").write_text(
                json.dumps(baseline), encoding="utf-8"
            )
            (run_dir / "finetuned_metrics.json").write_text(
                json.dumps(finetuned), encoding="utf-8"
            )
            return cli.build_comparison(run_dir, {})

    def test_legacy_bf16_artifacts_compare(self):
        # Artifacts from before quantization/revision metadata existed.
        def legacy(metrics):
            for key in (
                "model_kind",
                "requested_revision",
                "resolved_revision",
                "resolved_revision_source",
            ):
                metrics.pop(key)
            return metrics

        result = self._compare(legacy(self._metrics()), legacy(self._adapter()))
        self.assertFalse(result["base_identity"]["quantization"]["effective_load_in_4bit"])
        self.assertTrue(result["base_identity"]["match"])

    def test_new_bf16_quantization_metadata_matches(self):
        bf16 = {
            "requested_load_in_4bit": False,
            "effective_load_in_4bit": False,
            "quant_type": None,
            "compute_dtype": None,
            "double_quant": None,
            "quantized_parameter_devices": [],
            "offload": {"cpu": [], "disk": [], "meta": []},
        }
        result = self._compare(
            self._metrics(quantization=bf16),
            self._adapter(quantization=dict(bf16)),
        )
        self.assertFalse(result["base_identity"]["quantization"]["effective_load_in_4bit"])

    def test_legacy_and_new_bf16_metadata_match(self):
        bf16 = {
            "requested_load_in_4bit": False,
            "effective_load_in_4bit": False,
            "quant_type": None,
            "compute_dtype": None,
            "double_quant": None,
            "quantized_parameter_devices": [],
            "offload": {"cpu": [], "disk": [], "meta": []},
        }
        # Legacy artifact (no quantization key) vs new BF16 metadata, both ways.
        result = self._compare(self._metrics(), self._adapter(quantization=bf16))
        self.assertFalse(result["base_identity"]["quantization"]["effective_load_in_4bit"])
        self.assertEqual(result["base_identity"]["quantization"]["quantized_parameter_devices"], [])

        result = self._compare(self._metrics(quantization=dict(bf16)), self._adapter())
        self.assertTrue(result["base_identity"]["match"])

    def test_quantization_mismatch_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._compare(self._metrics(quantization=self.QUANT), self._adapter())
        self.assertIn("quantization", str(ctx.exception))

    def test_revision_mismatch_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._compare(self._metrics(), self._adapter(resolved_revision="rev2"))
        self.assertIn("revision", str(ctx.exception))

    def test_base_model_mismatch_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._compare(self._metrics(), self._adapter(base_model="other"))
        self.assertIn("base model", str(ctx.exception))

    def test_matching_4bit_artifacts_compare(self):
        result = self._compare(
            self._metrics(quantization=self.QUANT),
            self._adapter(quantization=dict(self.QUANT)),
        )
        self.assertTrue(result["base_identity"]["quantization"]["effective_load_in_4bit"])
        self.assertEqual(result["baseline"]["quantization"]["quant_type"], ["nf4"])
        self.assertEqual(
            result["baseline"]["quantization"]["quantized_parameter_devices"],
            ["cuda:0"],
        )

    def test_offload_count_mismatch_rejected(self):
        offloaded = dict(self.QUANT, offload={"cpu": ["layer"], "disk": [], "meta": []})
        with self.assertRaises(RuntimeError) as ctx:
            self._compare(
                self._metrics(quantization=offloaded),
                self._adapter(quantization=dict(self.QUANT)),
            )
        self.assertIn("quantization", str(ctx.exception))

    def test_quantized_device_mismatch_rejected(self):
        other_device = dict(
            self.QUANT, quantized_parameter_devices=["cuda:1"]
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._compare(
                self._metrics(quantization=other_device),
                self._adapter(quantization=dict(self.QUANT)),
            )
        self.assertIn("quantization", str(ctx.exception))

    def test_offload_module_prefixes_do_not_break_parity(self):
        # Same counts/placement, different wrapper-dependent module names.
        base_map = dict(
            self.QUANT,
            offload={"cpu": ["model.layers.0"], "disk": [], "meta": []},
        )
        adapter_map = dict(
            self.QUANT,
            offload={"cpu": ["base_model.model.layers.0"], "disk": [], "meta": []},
        )
        result = self._compare(
            self._metrics(quantization=base_map),
            self._adapter(quantization=adapter_map),
        )
        self.assertTrue(result["base_identity"]["match"])


class LossOnlyPredictionStepTest(unittest.TestCase):
    """Unsloth monkeypatches Trainer.prediction_step to force logits during eval;
    the local subclass must bypass it whenever prediction_loss_only=True."""

    def test_loss_only_bypasses_unsloth_step_and_delegates_otherwise(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - heavy ML stack optional
            self.skipTest("torch not installed")

        record = {
            "base_calls": [],
            "compute_calls": [],
            "env_during": [],
            "env_after": [],
            "batch_device": None,
        }

        class FakeBaseTrainer:
            def __init__(self):
                self.args = types.SimpleNamespace(device="cpu")

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                record["base_calls"].append((inputs, prediction_loss_only, ignore_keys))
                return ("base", None, None)

            def _prepare_inputs(self, inputs):
                return dict(inputs, prepared=True)

            def compute_loss_context_manager(self):
                return contextlib.nullcontext()

            def _get_num_items_in_batch(self, batch_samples, device):
                record["batch_device"] = (batch_samples, device)
                return 7

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                record["compute_calls"].append((inputs, return_outputs, num_items_in_batch))
                record["env_during"].append(os.environ.get("UNSLOTH_RETURN_LOGITS"))
                return torch.tensor([1.0, 2.0, 3.0])

        trainer = train_mod.loss_only_sft_trainer_class(FakeBaseTrainer)()

        with mock.patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            loss, logits, labels = trainer.prediction_step("model", {"labels": "x"}, True)
            record["env_after"].append(os.environ.get("UNSLOTH_RETURN_LOGITS"))

        # Loss-only: Unsloth's patched step never runs, no logits materialise.
        self.assertEqual(record["base_calls"], [])
        self.assertIsNone(logits)
        self.assertIsNone(labels)
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(loss.item(), 2.0)
        inputs, return_outputs, num_items = record["compute_calls"][0]
        self.assertTrue(inputs["prepared"])
        self.assertFalse(return_outputs)
        self.assertEqual(num_items, 7)
        self.assertEqual(record["batch_device"][1], "cpu")
        # Logits are disabled for the loss call, prior env value restored after.
        self.assertEqual(record["env_during"], ["0"])
        self.assertEqual(record["env_after"], ["1"])

        # Non-loss-only delegates to the (patched) super implementation.
        result = trainer.prediction_step("model", {"labels": "x"}, False, ["k"])
        self.assertEqual(result, ("base", None, None))
        self.assertEqual(record["base_calls"], [({"labels": "x"}, False, ["k"])])
        self.assertEqual(len(record["compute_calls"]), 1)

        # No env var leakage when it was unset to begin with.
        with mock.patch.dict(os.environ):
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
            trainer.prediction_step("model", {"labels": "x"}, True)
            self.assertNotIn("UNSLOTH_RETURN_LOGITS", os.environ)
        self.assertEqual(record["env_during"], ["0", "0"])


if __name__ == "__main__":
    unittest.main()
