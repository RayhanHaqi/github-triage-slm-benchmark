"""Cleaning + temporal split + SFT serialization.

Faithful generalization of the frozen `prepare_vscode_triage.py`:
cross-label number conflicts, exact-text duplicate normalization, verbatim
title/body leakage regexes, effective-empty rule, temporal per-class split
(int(n*0.8) / int(n*0.9)), one seeded RNG shuffling train then val then test,
and identical JSON field order/serialization.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

_FLAG_MAP = {"i": re.I, "m": re.M, "s": re.S, "x": re.X}


def compile_rules(rules: list[dict]) -> list[tuple[re.Pattern, str]]:
    """Compile `{pattern, flags, replacement}` rules preserving listed order."""
    compiled = []
    for rule in rules:
        flags = 0
        for ch in str(rule.get("flags", "")):
            flags |= _FLAG_MAP[ch.lower()]
        compiled.append(
            (re.compile(rule["pattern"], flags), rule.get("replacement", ""))
        )
    return compiled


def _apply_rules(text: str | None, rules: list[tuple[re.Pattern, str]]) -> str:
    text = text or ""
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def clean_title(text: str | None, rules: list[tuple[re.Pattern, str]]) -> str:
    return _apply_rules(text, rules).strip()


def clean_body(text: str | None, rules: list[tuple[re.Pattern, str]]) -> str:
    return _apply_rules(text, rules).strip()


def raw_path_for(config: dict, workspace: str | Path, class_name: str) -> Path:
    """Prefer gathered `<workspace>/raw/<class>.json`, else the configured file."""
    raw_path = Path(workspace) / "raw" / f"{class_name}.json"
    if raw_path.is_file():
        return raw_path
    source_file = config["sources"][class_name].get("file")
    if source_file:
        source = Path(source_file)
        if source.is_file():
            return source
    raise FileNotFoundError(
        f"no raw data for class {class_name!r}: expected {raw_path} "
        f"(run `specialist gather` first) or a configured source file"
    )


def load_raw(config: dict, workspace: str | Path) -> dict[str, list[dict]]:
    max_examples = int(config["cleaning"]["max_examples_per_class"])
    raw = {}
    for class_name in config["sources"]:
        path = raw_path_for(config, workspace, class_name)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"raw source for {class_name!r} must be a JSON array: {path}")
        raw[class_name] = rows[:max_examples]
        print(f"{class_name:18s}: {len(raw[class_name]):5d} raw")
    return raw


def build_splits(config: dict, raw: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    """Return ({train, val, test}, stats) following the frozen recipe exactly."""
    cleaning = config["cleaning"]
    split_cfg = config["split"]

    max_examples = int(cleaning["max_examples_per_class"])
    min_visible = int(cleaning["min_visible_chars"])
    whitespace = re.compile(cleaning["whitespace_pattern"])
    lowercase_key = bool(cleaning.get("duplicate_key_lowercase", True))

    title_rules = compile_rules(cleaning["title_patterns"])
    body_rules = compile_rules(cleaning["body_patterns"])

    raw = {class_name: rows[:max_examples] for class_name, rows in raw.items()}

    # ------------------------------------------------------------
    # Detect issues appearing under >1 target class
    # ------------------------------------------------------------
    number_to_labels: dict[str, set[str]] = defaultdict(set)
    for label, rows in raw.items():
        for row in rows:
            number_to_labels[row["number"]].add(label)

    conflicting_numbers = {
        number for number, labels in number_to_labels.items() if len(labels) > 1
    }
    print(f"\nCross-label conflicts: {len(conflicting_numbers)}")
    for number in sorted(conflicting_numbers)[:10]:
        print(f"  #{number}: {sorted(number_to_labels[number])}")

    # ------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------
    records = []
    dropped = Counter()

    for label, rows in raw.items():
        for row in rows:
            if row["number"] in conflicting_numbers:
                dropped["cross_label_conflict"] += 1
                continue

            title = clean_title(row.get("title"), title_rules)
            body = clean_body(row.get("body"), body_rules)
            text = f"Title: {title}\n\nDescription:\n{body}".strip()

            visible = whitespace.sub("", title + body)
            if len(visible) < min_visible:
                dropped["effectively_empty"] += 1
                continue

            records.append({
                "number": row["number"],
                "created_at": row["createdAt"],
                "label": label,
                "title": title,
                "body": body,
                "text": text,
            })

    counts = Counter(r["label"] for r in records)
    print("\nAfter cleaning:")
    for label in config["sources"]:
        print(f"{label:18s}: {counts[label]:5d}")
    print(f"{'TOTAL':18s}: {len(records):5d}")

    # ------------------------------------------------------------
    # Exact text duplicate detection
    # ------------------------------------------------------------
    by_text: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = whitespace.sub(" ", r["text"]).strip()
        if lowercase_key:
            key = key.lower()
        by_text[key].append(r)

    deduped = []
    for same_text_records in by_text.values():
        labels = {r["label"] for r in same_text_records}
        if len(labels) > 1:
            dropped["duplicate_conflicting_labels"] += len(same_text_records)
            continue
        if len(same_text_records) > 1:
            dropped["duplicate_same_label"] += len(same_text_records) - 1
        deduped.append(same_text_records[0])

    records = deduped

    # ------------------------------------------------------------
    # Temporal split WITHIN each class
    # ------------------------------------------------------------
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    temporal_key = split_cfg["temporal_key"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])

    for label, rows in by_label.items():
        rows.sort(key=lambda x: x[temporal_key])
        n = len(rows)
        i1 = int(n * train_ratio)
        i2 = int(n * val_ratio)
        splits["train"].extend(rows[:i1])
        splits["val"].extend(rows[i1:i2])
        splits["test"].extend(rows[i2:])

    # Shuffle only after temporal partition assignment, one seeded sequence:
    # train first, then val, then test.
    rng = random.Random(split_cfg["seed"])
    for name in ("train", "val", "test"):
        rng.shuffle(splits[name])

    stats = {
        "max_examples_per_class": max_examples,
        "sources": {
            class_name: {
                "raw_rows": len(raw[class_name]),
                "after_cleaning": counts[class_name],
            }
            for class_name in config["sources"]
        },
        "cross_label_conflicts": {
            "count": len(conflicting_numbers),
            "numbers": sorted(conflicting_numbers),
        },
        "dropped": dict(sorted(dropped.items())),
        "records_total": len(records),
        "split": {
            "seed": split_cfg["seed"],
            "temporal_key": temporal_key,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "sizes": {name: len(rows) for name, rows in splits.items()},
            "class_counts": {
                name: dict(Counter(r["label"] for r in rows))
                for name, rows in splits.items()
            },
        },
        "system_prompt": config["model"]["system_prompt"],
    }

    return splits, stats


def to_output_record(r: dict, system_prompt: str) -> dict:
    return {
        "issue_number": r["number"],
        "created_at": r["created_at"],
        "label": r["label"],
        # Keep raw-ish input separately for easy evaluation
        "input": r["text"],
        # Ready for chat-style SFT
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["text"]},
            {"role": "assistant", "content": r["label"]},
        ],
    }


def save_jsonl(path: str | Path, rows: list[dict], system_prompt: str) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_output_record(r, system_prompt), ensure_ascii=False) + "\n")


def prepare(config: dict, workspace: str | Path) -> dict:
    """Gather-independent preparation: write train/val/test jsonl + dataset_stats.json."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    raw = load_raw(config, workspace)
    splits, stats = build_splits(config, raw)

    system_prompt = config["model"]["system_prompt"]
    for name, rows in splits.items():
        save_jsonl(workspace / f"{name}.jsonl", rows, system_prompt)

        counts = Counter(r["label"] for r in rows)
        print(f"\n{name}.jsonl: {len(rows)}")
        for label in config["sources"]:
            print(f"  {label:18s}: {counts[label]}")

    (workspace / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: train.jsonl, val.jsonl, test.jsonl, dataset_stats.json in {workspace}")
    return stats
