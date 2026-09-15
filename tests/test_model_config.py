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


class FakeModel:
    def __init__(self, commit_hash=None, eos_token_id=None):
        self.config = types.SimpleNamespace(
            _commit_hash=commit_hash, eos_token_id=eos_token_id
        )
        self.eval_called = False
        self.saved = None

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


def fake_transformers(commit_hash=None):
    model = FakeModel(commit_hash)
    tokenizer = FakeTokenizer()
    processor = FakeProcessor()
    module = types.ModuleType("transformers")
    module.AutoTokenizer = FakeLoader(tokenizer)
    module.AutoModelForCausalLM = FakeLoader(model)
    module.AutoProcessor = FakeLoader(processor)
    module.AutoModelForImageTextToText = FakeLoader(model)
    return module, {"tokenizer": tokenizer, "processor": processor, "model": model}


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


class LoadModelTest(unittest.TestCase):
    def test_language_selects_causal_auto_classes(self):
        transformers_mod, fakes = fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": transformers_mod}):
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
        with mock.patch.dict(sys.modules, {"transformers": transformers_mod}):
            loaded = model_mod.load_model("m", revision="abc123", kind="language")

        meta = loaded.metadata()
        self.assertEqual(meta["resolved_revision"], "deadbeef")
        self.assertEqual(meta["resolved_revision_source"], "model_config")

    def test_vision_selects_processor_and_image_text_class(self):
        transformers_mod, fakes = fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": transformers_mod}):
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
        with mock.patch.dict(
            sys.modules, {"transformers": transformers_mod, "peft": peft_mod}
        ):
            loaded = model_mod.load_model("m", adapter="/tmp/adapter", kind="language")

        self.assertEqual(peft_cls.last_call[1], "/tmp/adapter")
        self.assertIs(loaded.model, peft_cls.last_call[0])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            model_mod.load_model("m", kind="audio")


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

    def test_library_versions_json_serializable(self):
        versions = model_mod.library_versions()
        self.assertIn("python", versions)
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

    def _run(self, config, workspace):
        dataset_calls = []
        trainer_holders = {}

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSFTTrainer:
            def __init__(self, **kwargs):
                trainer_holders["kwargs"] = kwargs
                self.state = types.SimpleNamespace(log_history=[])

            def train(self, **kwargs):
                return types.SimpleNamespace(metrics={"train_runtime": 1.0})

        class FakeCollator:
            def __init__(self, model, processor, **kwargs):
                self.model = model
                self.processor = processor
                self.kwargs = kwargs

        tokenizer = FakeTokenizer()
        processor = FakeProcessor()

        def make_fast_model(processor_result):
            return type(
                "FakeFastModel",
                (),
                {
                    "from_pretrained": classmethod(
                        lambda c, **kw: (
                            setattr(c, "last_load", kw),
                            (FakeModel("sha-abc"), processor_result),
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
        }
        with mock.patch.dict(sys.modules, modules):
            metrics = train_mod.train(config, workspace)

        return metrics, trainer_holders["kwargs"], dataset_calls, collators, (tokenizer, processor)

    def test_language_path_uses_trl24_max_length_and_no_eos_override(self):
        tmp, ws = self._workspace()
        with tmp:
            metrics, kwargs, dataset_calls, collators, (tokenizer, _) = self._run(
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
            _, kwargs, dataset_calls, collators, (_, processor) = self._run(
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

    def test_vision_collator_path_uses_message_rows(self):
        config = self._config(kind="vision")
        config["training"]["vision_collator"] = True
        tmp, ws = self._workspace()
        with tmp:
            metrics, kwargs, dataset_calls, collators, (_, processor) = self._run(config, ws)

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
            metrics, _, _, _, _ = self._run(
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
