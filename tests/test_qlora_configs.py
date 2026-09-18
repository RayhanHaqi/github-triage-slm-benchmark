"""Static checks for the QLoRA large-model track configs.

Parsing only (PyYAML + `cli.load_config`); no network, no GPU, no model loads.
Run with `python -m unittest discover -s tests`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specialist import cli  # noqa: E402
from specialist import dataset as dataset_mod  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "qlora-large"
REFERENCE_CONFIG = ROOT / "configs" / "vscode-bug-feature.yaml"

QWEN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Fixed track order: filename, identity, pinned revision, family behavior.
MODELS = [
    {
        "filename": "01-qwen3-8b.yaml",
        "name": "01-qwen3-8b",
        "base_model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "display_name": "Qwen3-8B",
        "kind": "language",
        "batch": 2,
        "ga": 4,
        "target_modules": QWEN_TARGETS,
        "vision_collator": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "vision_flags": False,
        "notes_needle": "seven-target",
    },
    {
        "filename": "02-ministral-3-8b-instruct.yaml",
        "name": "02-ministral-3-8b-instruct",
        "base_model": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
        "display_name": "Ministral-3-8B-Instruct-2512-BF16",
        "kind": "vision",
        "batch": 2,
        "ga": 4,
        "target_modules": "all-linear",
        "vision_collator": True,
        "chat_template_kwargs": {},
        "vision_flags": True,
        "notes_needle": "vision tower frozen",
    },
    {
        "filename": "03-qwen3.5-9b.yaml",
        "name": "03-qwen3.5-9b",
        "base_model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "display_name": "Qwen3.5-9B",
        "kind": "vision",
        "batch": 2,
        "ga": 4,
        "target_modules": "all-linear",
        "vision_collator": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "vision_flags": True,
        "notes_needle": "text-only",
    },
    {
        "filename": "04-ministral-3-14b-instruct.yaml",
        "name": "04-ministral-3-14b-instruct",
        "base_model": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "revision": "3cea74c1ebaf5ce5f5a2553de470e2ceab825142",
        "display_name": "Ministral-3-14B-Instruct-2512-BF16",
        "kind": "vision",
        "batch": 1,
        "ga": 8,
        "target_modules": "all-linear",
        "vision_collator": True,
        "chat_template_kwargs": {},
        "vision_flags": True,
        "notes_needle": "gradient accumulation 8",
    },
]

PINNED_DATASET = {
    "repo": dataset_mod.DATASET_REPO,
    "revision": dataset_mod.DATASET_REVISION,
}


class PinnedDatasetConfigTest(unittest.TestCase):
    def test_every_config_declares_the_pinned_dataset(self):
        for model in MODELS:
            config = cli.load_config(CONFIG_DIR / model["filename"])
            with self.subTest(model=model["filename"]):
                self.assertEqual(config["dataset"], PINNED_DATASET)
        reference = cli.load_config(REFERENCE_CONFIG)
        self.assertEqual(reference["dataset"], PINNED_DATASET)

    def test_sources_keep_label_order_without_raw_file_paths(self):
        for model in MODELS:
            config = cli.load_config(CONFIG_DIR / model["filename"])
            with self.subTest(model=model["filename"]):
                self.assertEqual(list(config["sources"]), ["bug", "feature-request"])
                for label in ("bug", "feature-request"):
                    self.assertEqual(config["sources"][label]["github_label"], label)
                    self.assertEqual(config["sources"][label]["repo"], "microsoft/vscode")
                    self.assertNotIn("file", config["sources"][label])


class QloraConfigTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.configs = {
            model["filename"]: cli.load_config(CONFIG_DIR / model["filename"])
            for model in MODELS
        }
        cls.raw = {
            model["filename"]: yaml.safe_load(
                (CONFIG_DIR / model["filename"]).read_text(encoding="utf-8")
            )
            for model in MODELS
        }
        cls.reference = cli.load_config(REFERENCE_CONFIG)

    def test_exact_files_and_fixed_order(self):
        names = [path.name for path in sorted(CONFIG_DIR.iterdir()) if path.is_file()]
        self.assertEqual(names, [model["filename"] for model in MODELS])
        self.assertEqual(len(MODELS), 4)

    def test_exact_ids_revisions_and_kinds(self):
        for model in MODELS:
            config = self.configs[model["filename"]]
            with self.subTest(model=model["filename"]):
                self.assertEqual(config["name"], model["name"])
                self.assertEqual(config["model"]["base_model"], model["base_model"])
                self.assertEqual(config["model"]["revision"], model["revision"])
                self.assertEqual(config["model"]["kind"], model["kind"])
                self.assertTrue(config["description"])

                benchmark = config["benchmark"]
                self.assertEqual(benchmark["display_name"], model["display_name"])
                # Explicit checkpoint/revision plus prior-track field-name aliases.
                self.assertEqual(benchmark["checkpoint"], model["base_model"])
                self.assertEqual(benchmark["revision"], model["revision"])
                self.assertEqual(benchmark["official_checkpoint"], model["base_model"])
                self.assertEqual(benchmark["pinned_revision"], model["revision"])

    def test_batches_and_effective_batch_eight(self):
        for model in MODELS:
            training = self.configs[model["filename"]]["training"]
            with self.subTest(model=model["filename"]):
                self.assertEqual(
                    training["per_device_train_batch_size"], model["batch"]
                )
                self.assertEqual(
                    training["gradient_accumulation_steps"], model["ga"]
                )
                self.assertEqual(training["per_device_eval_batch_size"], 1)
                self.assertEqual(model["batch"] * model["ga"], 8)
        # The only batch-1 model is the 14B with GA 8.
        self.assertEqual(
            [
                (m["batch"], m["ga"])
                for m in MODELS
                if m["batch"] == 1
            ],
            [(1, 8)],
        )

    def test_common_quantized_best_checkpoint_recipe(self):
        for model in MODELS:
            config = self.configs[model["filename"]]
            model_cfg, training, lora = config["model"], config["training"], config["lora"]
            with self.subTest(model=model["filename"]):
                self.assertIs(model_cfg["load_in_4bit"], True)
                self.assertEqual(model_cfg["max_seq_length"], 2048)
                self.assertIs(model_cfg["use_exact_model_name"], True)
                self.assertIs(model_cfg["trust_remote_code"], False)

                self.assertEqual(training["num_train_epochs"], 3)
                self.assertIs(training["bf16"], True)
                self.assertIs(training["fp16"], False)
                self.assertEqual(training["optim"], "adamw_8bit")
                self.assertEqual(training["learning_rate"], 2.0e-4)
                self.assertEqual(training["warmup_ratio"], 0.05)
                self.assertEqual(training["weight_decay"], 0.01)
                self.assertEqual(training["lr_scheduler_type"], "cosine")
                self.assertEqual(training["seed"], 42)
                self.assertEqual(training["logging_steps"], 10)
                self.assertEqual(training["eval_strategy"], "epoch")
                self.assertEqual(training["save_strategy"], "epoch")
                self.assertEqual(training["save_total_limit"], 2)
                self.assertIs(training["load_best_model_at_end"], True)
                self.assertEqual(training["metric_for_best_model"], "eval_loss")
                self.assertIs(training["greater_is_better"], False)

                self.assertEqual(lora["r"], 16)
                self.assertEqual(lora["alpha"], 32)
                self.assertEqual(lora["dropout"], 0)
                self.assertEqual(lora["bias"], "none")
                self.assertEqual(lora["gradient_checkpointing"], "unsloth")
                self.assertEqual(lora["random_state"], 42)

                benchmark = config["benchmark"]
                self.assertEqual(benchmark["method"], "QLoRA NF4")
                self.assertTrue(benchmark["compatibility_notes"].strip())
                self.assertIn("NF4", benchmark["compatibility_notes"])
                self.assertIn(model["notes_needle"], benchmark["compatibility_notes"])
                self.assertIn("BF16 reference track", benchmark["compatibility_notes"])

    def test_family_targets_collators_and_template_kwargs(self):
        for model in MODELS:
            config = self.configs[model["filename"]]
            lora, training, evaluation = config["lora"], config["training"], config["evaluation"]
            with self.subTest(model=model["filename"]):
                self.assertEqual(lora["target_modules"], model["target_modules"])
                self.assertEqual(training["vision_collator"], model["vision_collator"])
                self.assertEqual(
                    evaluation["chat_template_kwargs"], model["chat_template_kwargs"]
                )
                if model["kind"] == "vision":
                    self.assertIs(lora["finetune_vision_layers"], False)
                    self.assertIs(lora["finetune_language_layers"], True)
                    self.assertIs(lora["finetune_attention_modules"], True)
                    self.assertIs(lora["finetune_mlp_modules"], True)
                else:
                    for key in (
                        "finetune_vision_layers",
                        "finetune_language_layers",
                        "finetune_attention_modules",
                        "finetune_mlp_modules",
                    ):
                        self.assertNotIn(key, lora)

    def test_pinned_dataset_and_source_metadata_match_reference(self):
        reference = self.reference
        for model in MODELS:
            config = self.configs[model["filename"]]
            raw = self.raw[model["filename"]]
            with self.subTest(model=model["filename"]):
                self.assertEqual(raw["dataset"], PINNED_DATASET)
                self.assertEqual(config["dataset"], reference["dataset"])
                self.assertEqual(
                    list(config["sources"]), ["bug", "feature-request"]
                )
                for label in ("bug", "feature-request"):
                    self.assertEqual(raw["sources"][label]["github_label"], label)
                    self.assertEqual(raw["sources"][label]["repo"], "microsoft/vscode")
                    self.assertNotIn("file", raw["sources"][label])
                    self.assertEqual(
                        config["sources"][label], reference["sources"][label]
                    )

    def test_frozen_prompt_split_cleaning_and_eval_match_reference(self):
        reference = self.reference
        for model in MODELS:
            config = self.configs[model["filename"]]
            with self.subTest(model=model["filename"]):
                self.assertEqual(config["cleaning"], reference["cleaning"])
                self.assertEqual(config["split"], reference["split"])
                self.assertEqual(
                    config["model"]["system_prompt"],
                    reference["model"]["system_prompt"],
                )
                self.assertEqual(
                    config["paths"]["data_dir"], reference["paths"]["data_dir"]
                )
                self.assertEqual(
                    config["paths"]["runs_dir"], reference["paths"]["runs_dir"]
                )
                self.assertEqual(config["dataset"], reference["dataset"])
                for key in (
                    "split",
                    "max_user_tokens",
                    "max_new_tokens",
                    "do_sample",
                    "pad_eos",
                ):
                    self.assertEqual(
                        config["evaluation"][key], reference["evaluation"][key]
                    )

    def test_reference_config_stays_bf16_without_best_selection(self):
        reference = self.reference
        self.assertIs(reference["model"]["load_in_4bit"], False)
        self.assertNotIn("load_best_model_at_end", reference["training"])
        self.assertNotIn("metric_for_best_model", reference["training"])
        self.assertNotIn("greater_is_better", reference["training"])
        self.assertNotIn("benchmark", reference)


if __name__ == "__main__":
    unittest.main()
