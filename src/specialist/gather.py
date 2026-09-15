"""Raw data gathering: copy configured local JSON files or fetch live issues via `gh`.

Existing raw order is source truth; `cleaning.max_examples_per_class` takes the
first N rows in that order. A configured local `file` that is not larger than the
cap is copied byte-for-byte so downstream preparation sees identical bytes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Same fields the frozen raw files carry; `gh` only returns what is requested.
GH_JSON_FIELDS = "number,title,body,createdAt,closedAt"


def _write_rows(path: Path, rows: list) -> None:
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def gh_issue_list(repo: str, label: str | None, limit: int) -> list:
    """Fetch issues with `gh issue list`; returned order is kept as-is."""
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--state", "all",
        "--limit", str(limit),
        "--json", GH_JSON_FIELDS,
    ]
    if label:
        cmd += ["--label", label]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`gh issue list` failed for {repo} (label={label!r}):\n{proc.stderr.strip()}"
        )

    rows = json.loads(proc.stdout)
    if not isinstance(rows, list):
        raise ValueError(f"unexpected gh output for {repo}: {type(rows).__name__}")
    return rows


def gather(config: dict, workspace: str | Path) -> dict:
    """Gather raw issues for every configured class into `<workspace>/raw/<class>.json`."""
    workspace = Path(workspace)
    raw_dir = workspace / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_examples = int(config["cleaning"]["max_examples_per_class"])
    report = {}

    for class_name, spec in config["sources"].items():
        dest = raw_dir / f"{class_name}.json"
        source_file = spec.get("file")

        if source_file:
            src = Path(source_file)
            if not src.is_file():
                raise FileNotFoundError(f"source file for {class_name!r} not found: {src}")
            data = src.read_bytes()
            rows = json.loads(data)
            if not isinstance(rows, list):
                raise ValueError(f"source file for {class_name!r} must contain a JSON array")
            if len(rows) <= max_examples:
                dest.write_bytes(data)  # byte-identical copy
                mode = "copied"
            else:
                _write_rows(dest, rows[:max_examples])
                mode = f"truncated:{len(rows)}->{max_examples}"
            source = str(src)
        else:
            repo = spec.get("repo")
            if not repo:
                raise ValueError(f"source {class_name!r} needs either `file` or `repo`")
            rows = gh_issue_list(repo, spec.get("github_label"), max_examples)[:max_examples]
            _write_rows(dest, rows)
            mode = "gh"
            source = f"gh:{repo}"

        report[class_name] = {
            "class": class_name,
            "source": source,
            "mode": mode,
            "rows": len(json.loads(dest.read_text(encoding="utf-8"))),
            "path": str(dest),
        }
        print(
            f"{class_name:18s}: {report[class_name]['rows']:5d} rows "
            f"({mode}) -> {dest}"
        )

    return report
