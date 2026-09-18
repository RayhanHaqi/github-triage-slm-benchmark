#!/usr/bin/env python3
"""Regenerate the README SVGs from the canonical committed benchmark CSVs.

Stdlib only (no new dependencies). Sources are read-only committed evidence:

    benchmarks/20260915T112519Z/results.csv              (BF16 LoRA track)
    benchmarks/qlora-large/20260916T185922Z/results.csv  (QLoRA NF4 track)

Outputs:

    assets/strict-accuracy-dumbbell.svg
    assets/strict-delta-vs-vram.svg

Usage:
    python scripts/generate_readme_charts.py          # regenerate the SVGs
    python scripts/generate_readme_charts.py --check  # self-check data facts + byte-identical SVGs

The charts are deliberately descriptive: two tracks with different precision,
checkpoint-selection policy and provenance are drawn as separate series and the
cross-track caveat is printed inside both figures.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACKS = (
    {
        "key": "bf16",
        "label": "BF16 LoRA",
        "csv": "benchmarks/20260915T112519Z/results.csv",
        "detail": "run 20260915T112519Z \u00b7 6 configurations \u00b7 full BF16 base + LoRA \u00b7 adapter = final epoch (3)",
        "color": "#0969da",
    },
    {
        "key": "qlora",
        "label": "QLoRA NF4",
        "csv": "benchmarks/qlora-large/20260916T185922Z/results.csv",
        "detail": "run 20260916T185922Z \u00b7 4 configurations \u00b7 4-bit NF4 base (bf16 compute) \u00b7 best checkpoint (epoch 2) restored for the remaining three QLoRA configurations",
        "color": "#bc4c00",
    },
)

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

W = 980
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
INK = "#1f2328"
INK2 = "#57606a"
INK3 = "#6e7781"
GRID = "#eaeef2"
RULE = "#d0d7de"
AXIS = "#afb8c1"
WARN_BG = "#fff8c5"
WARN_BAR = "#bf8700"
WARN_INK = "#4d2d00"
PANEL = "#ffffff"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x, y, s, size=11.5, fill=INK, anchor="start", weight="400", font=FONT):
    extra = f' font-weight="{weight}"' if weight != "400" else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size:g}" fill="{fill}"'
        f' text-anchor="{anchor}"{extra} font-family="{font}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, stroke=RULE, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"'
        f' stroke="{stroke}" stroke-width="{width:g}"{d}/>'
    )


# ---------------------------------------------------------------- data loading


def _num(row, key):
    return float(row[key])


def load_track(track):
    path = ROOT / track["csv"]
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    configs, by_slug = [], {}
    for row in rows:
        slug = row["model_slug"]
        if slug not in by_slug:
            cfg = {
                "slug": slug,
                "name": row["display_name"],
                "provenance": row.get("model_provenance") or "",
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
        cfg["base_strict"] = _num(base, "strict_accuracy")
        cfg["ft_strict"] = _num(ft, "strict_accuracy")
        cfg["delta_strict"] = _num(ft, "delta_strict_pp")
        cfg["delta_semantic"] = _num(ft, "delta_semantic_pp")
        cfg["ft_peak_reserved"] = _num(ft, "peak_reserved_gib")
        cfg["ft_speed"] = _num(ft, "output_tokens_per_second")
        if abs((cfg["ft_strict"] - cfg["base_strict"]) * 100 - cfg["delta_strict"]) > 0.051:
            raise SystemExit(f"{path}: {cfg['slug']} strict delta disagrees with accuracies")
        cfg["short"] = re.sub(r"-2512-BF16$", "", cfg["name"])
        if "imported" in cfg["provenance"]:
            cfg["short"] += "\u2020"
    return configs


def load_all():
    return [(track, load_track(track)) for track in TRACKS]


def pct(value):
    return f"{value * 100:.1f}"


def signed(value):
    return f"{value:+.1f}"


# ------------------------------------------------------------ shared fragments


def panel_start(height, title, desc, prefix):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height:g}"'
        f' viewBox="0 0 {W} {height:g}" role="img"'
        f' aria-labelledby="{prefix}-title" aria-describedby="{prefix}-desc">',
        f'<title id="{prefix}-title">{esc(title)}</title>',
        f'<desc id="{prefix}-desc">{esc(desc)}</desc>',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1:g}" rx="8"'
        f' fill="{PANEL}" stroke="{RULE}"/>',
    ]


def warn_box(y, width_lines):
    """Amber caveat band. width_lines is a list of raw inner-SVG text strings."""
    height = 20 + 16 * (len(width_lines) - 1) + 12
    parts = [
        f'<rect x="16" y="{y:g}" width="{W - 32}" height="{height:g}" rx="6" fill="{WARN_BG}"/>',
        f'<rect x="16" y="{y:g}" width="3" height="{height:g}" rx="1.5" fill="{WARN_BAR}"/>',
    ]
    for i, inner in enumerate(width_lines):
        parts.append(
            f'<text x="32" y="{y + 20 + 16 * i:g}" font-size="11.5" fill="{WARN_INK}"'
            f' font-family="{FONT}">{inner}</text>'
        )
    return parts, height


def legend_item(x, y, track, shape):
    color = track["color"]
    if shape == "circle":
        marker = f'<circle cx="{x:g}" cy="{y:g}" r="5" fill="{color}"/>'
    else:
        marker = f'<rect x="{x - 5:g}" y="{y - 5:g}" width="10" height="10" rx="2" fill="{color}"/>'
    return [marker, text(x + 12, y + 4, track["label"], 11.5, INK2)]


# ------------------------------------------------------------------- dumbbell


def dumbbell_svg(grouped):
    x0, x1 = 278.0, 838.0
    lo, hi = 0.55, 0.95
    px = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731

    grid, body = [], []
    top = 36.0
    body.append(text(16, top, "Strict accuracy: base checkpoint \u2192 fine-tuned adapter", 16.5, INK, weight="700"))
    body.append(text(16, top + 22, "All ten configurations on the frozen 200-row test split \u00b7 one single run per configuration", 12, INK2))
    body += legend_item(16, top + 48, TRACKS[0], "circle")
    body += legend_item(150, top + 48, TRACKS[1], "square")
    body += [
        f'<circle cx="290" cy="{top + 44:g}" r="4.5" fill="#ffffff" stroke="{INK3}" stroke-width="1.5"/>',
        text(302, top + 48, "base checkpoint", 11.5, INK2),
        text(420, top + 48, "\u2020 Qwen3-8B imported from run 20260916T120019Z (not rerun)", 10.5, INK2),
    ]

    y = 118.0
    plot_top = None
    for ti, (track, configs) in enumerate(grouped):
        if ti:
            y += 12
            body.append(line(16, y, W - 16, y, RULE))
            y += 6
        header = y + 8
        body.append(text(16, header, track["label"], 12.5, track["color"], weight="700"))
        body.append(text(16, header + 17, track["detail"], 11, INK2))
        body.append(text(W - 16, header + 4, "\u0394 strict (pp)", 11, INK2, anchor="end"))
        row_y = header + 40
        if plot_top is None:
            plot_top = row_y - 8
        for cfg in configs:
            body.append(text(16, row_y + 4, cfg["short"], 12.5, INK))
            body.append(line(px(cfg["base_strict"]), row_y, px(cfg["ft_strict"]), row_y, RULE, 2.0))
            body.append(
                f'<circle cx="{px(cfg["base_strict"]):.1f}" cy="{row_y:g}" r="4.5"'
                f' fill="#ffffff" stroke="{INK3}" stroke-width="1.5"/>'
            )
            color = track["color"]
            if track["key"] == "bf16":
                body.append(f'<circle cx="{px(cfg["ft_strict"]):.1f}" cy="{row_y:g}" r="5.5" fill="{color}"/>')
            else:
                cx = px(cfg["ft_strict"])
                body.append(
                    f'<rect x="{cx - 5:.1f}" y="{row_y - 5:g}" width="10" height="10" rx="2" fill="{color}"/>'
                )
            delta = cfg["delta_strict"]
            delta_color = INK if delta >= 0 else "#cf222e"
            body.append(
                text(W - 16, row_y + 4, signed(delta), 12, delta_color, anchor="end", font=MONO)
            )
            if delta < 0:
                body.append(text(W - 16, row_y + 18, "regression", 9.5, "#cf222e", anchor="end"))
            row_y += 33
        y = row_y
    plot_bottom = y + 8
    for v_i in range(9):
        v = lo + (hi - lo) * v_i / 8
        grid.append(line(px(v), plot_top, px(v), plot_bottom, GRID))

    axis_y = y + 18
    body.append(line(x0, axis_y, x1, axis_y, AXIS))
    for v_i in range(9):
        v = lo + (hi - lo) * v_i / 8
        body.append(text(px(v), axis_y + 15, f"{v:.2f}", 10, INK2, anchor="middle", font=MONO))
    body.append(text((x0 + x1) / 2, axis_y + 32, "strict accuracy (fraction of the 200 test issues)", 11, INK2, anchor="middle"))

    warn, warn_h = warn_box(
        axis_y + 44,
        [
            '<tspan font-weight="700">Do not rank across tracks.</tspan> BF16 LoRA and QLoRA NF4 differ in precision, checkpoint-selection policy and provenance.',
            "Matches hold only base vs fine-tuned within one configuration; no bigger-is-better implication is made for model size or parameter count.",
            "Single runs on a frozen 200-row holdout: differences are descriptive, not statistically significant.",
        ],
    )
    height = axis_y + 44 + warn_h + 16
    return "\n".join(
        panel_start(
            height,
            "Strict accuracy before and after fine-tuning",
            "Dumbbell chart of base versus fine-tuned strict accuracy for ten configurations, split by BF16 LoRA and QLoRA NF4 track. Tracks are not a controlled cross-track ranking.",
            "dumbbell",
        )
        + ["".join(grid), "".join(body), "".join(warn)]
        + ["</svg>"]
    )


# -------------------------------------------------------------------- scatter


def scatter_svg(grouped):
    x0, x1 = 90.0, 920.0
    y0, y1 = 118.0, 470.0
    x_lo, x_hi = 0.0, 10.5
    y_lo, y_hi = -6.0, 34.0
    px = lambda v: x0 + (v - x_lo) / (x_hi - x_lo) * (x1 - x0)  # noqa: E731
    py = lambda v: y1 - (v - y_lo) / (y_hi - y_lo) * (y1 - y0)  # noqa: E731

    grid, body = [], []
    body.append(text(16, 36, "Strict accuracy delta vs fine-tuned evaluation peak VRAM", 16.5, INK, weight="700"))
    body.append(text(16, 58, "Each point is one configuration (single run); two tracks with different precision, checkpoint policy and provenance.", 12, INK2))
    body += legend_item(16, 88, TRACKS[0], "circle")
    body += legend_item(130, 88, TRACKS[1], "square")
    body.append(text(244, 92, "\u2020 imported Qwen3-8B evidence: measured without GPU sharing", 11, INK2))

    for v in (0, 2, 4, 6, 8, 10):
        grid.append(line(px(v), y0, px(v), y1, GRID))
        body.append(text(px(v), y1 + 16, f"{v:g}", 10, INK2, anchor="middle", font=MONO))
    for v in (-5, 0, 5, 10, 15, 20, 25, 30):
        grid.append(line(x0, py(v), x1, py(v), GRID))
        body.append(text(x0 - 8, py(v) + 4, f"{v:g}", 10, INK2, anchor="end", font=MONO))
    grid.append(line(x0, py(0), x1, py(0), AXIS, 1.0, dash="4 3"))
    body.append(line(x0, y1, x1, y1, AXIS))

    placed = []
    for track, configs in grouped:
        color = track["color"]
        for cfg in configs:
            cx, cy = px(cfg["ft_peak_reserved"]), py(cfg["delta_strict"])
            if track["key"] == "bf16":
                body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}"/>')
            else:
                body.append(f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
            label = cfg["short"]
            est_w = 6.1 * len(label) + 8
            candidates = [
                ("middle", 0, -11),
                ("middle", 0, 19),
                ("start", 11, 4),
                ("end", -11, 4),
                ("start", 11, -10),
                ("end", -11, -10),
            ]
            chosen = candidates[0]
            for anchor, dx, dy in candidates:
                tx = cx + dx
                left = tx - est_w / 2 if anchor == "middle" else (tx if anchor == "start" else tx - est_w)
                box = (left, cy + dy - 10, left + est_w, cy + dy + 4)
                if box[0] < x0 - 4 or box[2] > x1 + 30:
                    continue
                if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                    chosen = (anchor, dx, dy)
                    placed.append(box)
                    break
            else:
                placed.append((cx - est_w / 2, cy - 21, cx + est_w / 2, cy - 7))
            anchor, dx, dy = chosen
            body.append(text(f"{cx + dx:.1f}", f"{cy + dy:.1f}", label, 10.5, INK, anchor=anchor))

    body.append(text(x0, y0 - 10, "\u0394 strict accuracy (pp)", 11, INK2))
    body.append(text((x0 + x1) / 2, y1 + 42, "fine-tuned evaluation process peak reserved VRAM (GiB, PyTorch allocator, counters reset after model load)", 11, INK2, anchor="middle"))

    warn, warn_h = warn_box(
        y1 + 58,
        [
            '<tspan font-weight="700">Not a causal or efficiency ranking.</tspan> The two tracks differ in precision, checkpoint-selection policy and provenance.',
            "Qwen3-8B\u2020 is imported evidence; the remaining three QLoRA configurations ran under a shared-GPU policy, while the BF16 track did not.",
            "Peak VRAM is a process-level allocator measurement, not whole-GPU usage.",
            "Single runs on a frozen 200-row holdout; no model-size or precision effect is claimed.",
        ],
    )
    height = y1 + 58 + warn_h + 16
    return "\n".join(
        panel_start(
            height,
            "Strict accuracy delta versus fine-tuned evaluation peak VRAM",
            "Scatter plot of strict accuracy delta in percentage points against fine-tuned evaluation peak reserved VRAM in GiB for ten configurations across two tracks. Descriptive and non-causal.",
            "scatter",
        )
        + ["".join(grid), "".join(body), "".join(warn)]
        + ["</svg>"]
    )


# --------------------------------------------------------------- self-checks


def data_facts(grouped):
    configs = [cfg for _, configs_ in grouped for cfg in configs_]
    best = max(configs, key=lambda c: c["delta_strict"])
    worst = min(configs, key=lambda c: c["delta_strict"])
    bf16_peaks = [cfg["ft_peak_reserved"] for track, configs_ in grouped if track["key"] == "bf16" for cfg in configs_]
    qlora_peaks = [cfg["ft_peak_reserved"] for track, configs_ in grouped if track["key"] == "qlora" for cfg in configs_]
    facts = {
        "configurations": len(configs),
        "strict_improved": sum(1 for c in configs if c["delta_strict"] > 0),
        "semantic_improved": sum(1 for c in configs if c["delta_semantic"] > 0),
        "best_delta_pp": round(best["delta_strict"], 1),
        "best_delta_model": best["name"],
        "worst_delta_pp": round(worst["delta_strict"], 1),
        "worst_delta_model": worst["name"],
        "bf16_adapter_peak_reserved_max_gib": round(max(bf16_peaks), 5),
        "qlora_adapter_peak_reserved_max_gib": round(max(qlora_peaks), 5),
    }
    qlora_csv = ROOT / TRACKS[1]["csv"]
    with qlora_csv.open(newline="", encoding="utf-8") as fh:
        ql_rows = list(csv.DictReader(fh))
    if ql_rows and "acceptance_peak_reserved_gib" in ql_rows[0]:
        facts["qlora_acceptance_peak_reserved_max_gib"] = round(
            max(float(r["acceptance_peak_reserved_gib"]) for r in ql_rows if r["phase"] == "finetuned"), 5
        )
    return facts


def self_check(grouped):
    facts = data_facts(grouped)
    for key, expected in EXPECTED.items():
        actual = facts.get(key)
        assert actual == expected, f"self-check failed: {key} is {actual!r}, expected {expected!r}"
    return facts


# ---------------------------------------------------------------------- main


def render_all(grouped):
    return {
        "assets/strict-accuracy-dumbbell.svg": dumbbell_svg(grouped),
        "assets/strict-delta-vs-vram.svg": scatter_svg(grouped),
    }


def main(argv):
    check = "--check" in argv
    grouped = load_all()
    svg_by_name = render_all(grouped)
    facts = data_facts(grouped)
    self_check(grouped)
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
    failures = 0
    for name, svg in svg_by_name.items():
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if check:
            if old == svg:
                print(f"ok        {name} (byte-identical, {len(svg)} bytes)")
            else:
                print(f"DRIFT     {name}: committed file does not match regenerated output")
                failures += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg, encoding="utf-8")
            state = "unchanged" if old == svg else ("updated" if old is not None else "created")
            print(f"{state:9} {name} ({len(svg)} bytes)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
