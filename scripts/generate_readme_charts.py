#!/usr/bin/env python3
"""Regenerate the README chart from the canonical committed benchmark CSVs.

Stdlib only (no new dependencies). Sources are read-only committed evidence:

    benchmarks/20260915T112519Z/results.csv              (BF16 LoRA track)
    benchmarks/qlora-large/20260916T185922Z/results.csv  (QLoRA NF4 track)

Output:

    assets/strict-accuracy-dumbbell.svg

Usage:
    python scripts/generate_readme_charts.py          # regenerate the SVG
    python scripts/generate_readme_charts.py --check  # self-check data facts + byte-identical SVG

The figure shows base vs fine-tuned strict accuracy per configuration, with the
two tracks drawn as separate blocks. The cross-track caveat lives in the README.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACKS = (
    {
        "key": "bf16",
        "label": "BF16 LoRA",
        "csv": "benchmarks/20260915T112519Z/results.csv",
        "color": "#0969da",
        "labels": {
            "01-qwen3.5-0.8b": "Qwen3.5 0.8B",
            "02-lfm2.5-1.2b-instruct": "LFM2.5 1.2B",
            "03-qwen3-1.7b": "Qwen3 1.7B",
            "04-qwen3.5-2b": "Qwen3.5 2B",
            "05-ministral-3-3b-instruct": "Ministral 3B",
            "06-qwen3.5-4b": "Qwen3.5 4B",
        },
    },
    {
        "key": "qlora",
        "label": "QLoRA NF4",
        "csv": "benchmarks/qlora-large/20260916T185922Z/results.csv",
        "color": "#bc4c00",
        "labels": {
            "01-qwen3-8b": "Qwen3 8B",
            "02-ministral-3-8b-instruct": "Ministral 8B",
            "03-qwen3.5-9b": "Qwen3.5 9B",
            "04-ministral-3-14b-instruct": "Ministral 14B",
        },
    },
)

OUT_SVG = "assets/strict-accuracy-dumbbell.svg"

# Expected values from the committed evidence; --check fails if the CSVs drift.
EXPECTED = {
    "configurations": 10,
    "strict_improved": 9,
    "semantic_improved": 10,
    "best_delta_pp": 31.0,
    "best_delta_model": "Qwen3-1.7B",
    "worst_delta_pp": -3.5,
    "worst_delta_model": "Qwen3.5-4B",
    "bf16_adapter_peak_reserved_max_gib": 9.35742,
    "qlora_acceptance_peak_reserved_max_gib": 9.93945,
}

# Geometry and palette (light-neutral panel, readable on GitHub light and dark).
W = 860
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
INK = "#1f2328"
INK2 = "#57606a"
INK3 = "#6e7781"
GRID = "#eaeef2"
RULE = "#d0d7de"
AXIS = "#afb8c1"
NEGATIVE = "#cf222e"
PANEL = "#ffffff"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x, y, s, size, fill, anchor="start", weight="400", font=FONT):
    extra = f' font-weight="{weight}"' if weight != "400" else ""
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill}"'
        f' text-anchor="{anchor}"{extra} font-family="{font}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, stroke, width=1.0):
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"'
        f' stroke="{stroke}" stroke-width="{width:g}"/>'
    )


def delta_label(value):
    return f"\u2212{abs(value):.1f}" if value < 0 else f"+{value:.1f}"


# ---------------------------------------------------------------- data loading


def load_track(track):
    path = ROOT / track["csv"]
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    configs, by_slug = [], {}
    for row in rows:
        slug = row["model_slug"]
        if slug not in track["labels"]:
            raise SystemExit(f"{path}: no chart label for new model slug {slug!r}")
        if slug not in by_slug:
            cfg = {
                "slug": slug,
                "name": row["display_name"],
                "short": track["labels"][slug],
                "base": None,
                "ft": None,
            }
            by_slug[slug] = cfg
            configs.append(cfg)
        phase = row["phase"]
        if phase == "baseline":
            by_slug[slug]["base"] = row
        elif phase == "finetuned":
            by_slug[slug]["ft"] = row
        else:
            raise SystemExit(f"{path}: unexpected phase {phase!r}")
    for cfg in configs:
        if cfg["base"] is None or cfg["ft"] is None:
            raise SystemExit(f"{path}: {cfg['slug']} is missing a baseline/finetuned row")
        base, ft = cfg["base"], cfg["ft"]
        cfg["base_strict"] = float(base["strict_accuracy"])
        cfg["ft_strict"] = float(ft["strict_accuracy"])
        cfg["delta_strict"] = float(ft["delta_strict_pp"])
        cfg["delta_semantic"] = float(ft["delta_semantic_pp"])
        cfg["ft_peak_reserved"] = float(ft["peak_reserved_gib"])
        if abs((cfg["ft_strict"] - cfg["base_strict"]) * 100 - cfg["delta_strict"]) > 0.051:
            raise SystemExit(f"{path}: {cfg['slug']} strict delta disagrees with accuracies")
    return configs


def load_all():
    return [(track, load_track(track)) for track in TRACKS]


# -------------------------------------------------------------------- figure


def dumbbell_svg(grouped):
    x0, x1 = 210.0, 720.0
    lo, hi = 0.55, 0.95
    px = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731

    grid, body = [], []
    y = 26.0
    plot_top = plot_bottom = None
    for ti, (track, configs) in enumerate(grouped):
        if ti:
            y += 42
        body.append(text(16, y, track["label"], 14, track["color"], weight="700"))
        row_y = y + 22
        if plot_top is None:
            plot_top = row_y - 12
        for cfg in configs:
            body.append(text(16, row_y + 4.5, cfg["short"], 13.5, INK))
            body.append(line(px(cfg["base_strict"]), row_y, px(cfg["ft_strict"]), row_y, RULE, 2.0))
            body.append(
                f'<circle cx="{px(cfg["base_strict"]):.1f}" cy="{row_y:g}" r="4"'
                f' fill="#ffffff" stroke="{INK3}" stroke-width="1.5"/>'
            )
            ft_x = px(cfg["ft_strict"])
            if track["key"] == "bf16":
                body.append(f'<circle cx="{ft_x:.1f}" cy="{row_y:g}" r="5.5" fill="{track["color"]}"/>')
            else:
                body.append(
                    f'<rect x="{ft_x - 5:.1f}" y="{row_y - 5:g}" width="10" height="10"'
                    f' rx="2" fill="{track["color"]}"/>'
                )
            delta = cfg["delta_strict"]
            body.append(
                text(
                    W - 16,
                    row_y + 4.5,
                    delta_label(delta),
                    12.5,
                    NEGATIVE if delta < 0 else INK,
                    anchor="end",
                    font=MONO,
                )
            )
            row_y += 30
        y = row_y - 30
    plot_bottom = y + 12

    for v in (0.60, 0.70, 0.80, 0.90):
        grid.append(line(px(v), plot_top, px(v), plot_bottom, GRID))
    axis_y = y + 20
    body.append(line(x0, axis_y, x1, axis_y, AXIS))
    for v in (0.60, 0.70, 0.80, 0.90):
        body.append(text(px(v), axis_y + 16, f"{v * 100:.0f}%", 11.5, INK2, anchor="middle", font=MONO))

    height = axis_y + 30
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height:g}"'
        f' viewBox="0 0 {W} {height:g}" role="img"'
        f' aria-labelledby="dumbbell-title" aria-describedby="dumbbell-desc">',
        '<title id="dumbbell-title">Strict accuracy: base checkpoint vs fine-tuned adapter</title>',
        '<desc id="dumbbell-desc">Dumbbell chart of base vs fine-tuned strict accuracy for ten '
        "configurations on the frozen 200-row test split, split into BF16 LoRA and QLoRA NF4 "
        "blocks. Open markers are base checkpoints, filled markers are fine-tuned adapters, and "
        "the signed value at right is the adapter-minus-base delta in percentage points.</desc>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1:g}" rx="8"'
        f' fill="{PANEL}" stroke="{RULE}"/>',
    ]
    return "\n".join(header + ["".join(grid), "".join(body), "</svg>"])


# --------------------------------------------------------------- self-checks


def data_facts(grouped):
    configs = [cfg for _, configs_ in grouped for cfg in configs_]
    best = max(configs, key=lambda c: c["delta_strict"])
    worst = min(configs, key=lambda c: c["delta_strict"])
    bf16_peaks = [cfg["ft_peak_reserved"] for track, configs_ in grouped if track["key"] == "bf16" for cfg in configs_]
    qlora_csv = ROOT / TRACKS[1]["csv"]
    with qlora_csv.open(newline="", encoding="utf-8") as fh:
        ql_rows = list(csv.DictReader(fh))
    return {
        "configurations": len(configs),
        "strict_improved": sum(1 for c in configs if c["delta_strict"] > 0),
        "semantic_improved": sum(1 for c in configs if c["delta_semantic"] > 0),
        "best_delta_pp": round(best["delta_strict"], 1),
        "best_delta_model": best["name"],
        "worst_delta_pp": round(worst["delta_strict"], 1),
        "worst_delta_model": worst["name"],
        "bf16_adapter_peak_reserved_max_gib": round(max(bf16_peaks), 5),
        "qlora_acceptance_peak_reserved_max_gib": round(
            max(float(r["acceptance_peak_reserved_gib"]) for r in ql_rows if r["phase"] == "finetuned"), 5
        ),
    }


def self_check(grouped):
    facts = data_facts(grouped)
    for key, expected in EXPECTED.items():
        actual = facts.get(key)
        assert actual == expected, f"self-check failed: {key} is {actual!r}, expected {expected!r}"
    return facts


# ---------------------------------------------------------------------- main


def main(argv):
    check = "--check" in argv
    grouped = load_all()
    svg = dumbbell_svg(grouped)
    facts = self_check(grouped)
    print(
        "data check ok: {} configurations, {} strict / {} semantic improved, "
        "best {} pp ({}), worst {} pp ({})".format(
            facts["configurations"],
            facts["strict_improved"],
            facts["semantic_improved"],
            facts["best_delta_pp"],
            facts["best_delta_model"],
            facts["worst_delta_pp"],
            facts["worst_delta_model"],
        )
    )
    path = ROOT / OUT_SVG
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if check:
        if old == svg:
            print(f"ok        {OUT_SVG} (byte-identical, {len(svg)} bytes)")
            return 0
        print(f"DRIFT     {OUT_SVG}: committed file does not match regenerated output")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    state = "unchanged" if old == svg else ("updated" if old is not None else "created")
    print(f"{state:9} {OUT_SVG} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
