#!/usr/bin/env python
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    label: str
    group: str
    setup: str
    frozen: str
    csv_path: str
    notes: str = ""


EXPERIMENTS: List[Experiment] = [
    Experiment(
        exp_id="additive_v2",
        label="Additive (Run A)",
        group="Selective Strip Fusion",
        setup="Unfrozen, 1500 steps/epoch",
        frozen="No",
        csv_path="results/Rain100L_additive_v2/psnr_per_epoch.csv",
        notes="Older long-run setup",
    ),
    Experiment(
        exp_id="concat_v1",
        label="Concat (Run A)",
        group="Selective Strip Fusion",
        setup="Unfrozen, 1500 steps/epoch",
        frozen="No",
        csv_path="results/Rain100L_concat_v1/psnr_per_epoch.csv",
        notes="Older long-run setup",
    ),
    Experiment(
        exp_id="additive_v3",
        label="Additive (Run B)",
        group="Selective Strip Fusion",
        setup="Unfrozen, 50 steps/epoch",
        frozen="No",
        csv_path="results/Rain100L_additive_v3/metrics_per_epoch.csv",
    ),
    Experiment(
        exp_id="concat_v2",
        label="Concat (Run B)",
        group="Selective Strip Fusion",
        setup="Unfrozen, 50 steps/epoch",
        frozen="No",
        csv_path="results/Rain100L_concat_v2/metrics_per_epoch.csv",
    ),
    Experiment(
        exp_id="gated_v1",
        label="Gated (Run B)",
        group="Selective Strip Fusion",
        setup="Unfrozen, 50 steps/epoch",
        frozen="No",
        csv_path="results/Rain100L_gated_v1/metrics_per_epoch.csv",
        notes="Winner of screening run",
    ),
    Experiment(
        exp_id="baseline_frozen",
        label="Baseline Frozen",
        group="Frozen Backbone Comparison",
        setup="Frozen backbone, 50 steps/epoch",
        frozen="Yes",
        csv_path="results/Rain100L_baseline_frozen/metrics_per_epoch.csv",
    ),
    Experiment(
        exp_id="gated_frozen",
        label="Gated Frozen",
        group="Frozen Backbone Comparison",
        setup="Frozen backbone, 50 steps/epoch",
        frozen="Yes",
        csv_path="results/Rain100L_gated_frozen/metrics_per_epoch.csv",
        notes="Best clean ablation result",
    ),
    Experiment(
        exp_id="channel_attn_frozen",
        label="Channel Attention Frozen",
        group="Mechanism Swap",
        setup="Frozen backbone, 50 steps/epoch",
        frozen="Yes",
        csv_path="results/Rain100L_channel_attn_frozen/metrics_per_epoch.csv",
    ),
    Experiment(
        exp_id="self_attn_frozen",
        label="Self Attention Frozen",
        group="Mechanism Swap",
        setup="Frozen backbone, 50 steps/epoch",
        frozen="Yes",
        csv_path="results/Rain100L_self_attn_frozen/metrics_per_epoch.csv",
    ),
    Experiment(
        exp_id="swin_attn_frozen",
        label="Swin Attention Frozen",
        group="Mechanism Swap",
        setup="Frozen backbone, 50 steps/epoch",
        frozen="Yes",
        csv_path="results/Rain100L_swin_attn_frozen/metrics_per_epoch.csv",
    ),
]


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def read_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: Dict[str, float] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def summarize_experiment(exp: Experiment, repo_root: Path) -> Dict[str, object]:
    csv_path = repo_root / exp.csv_path
    rows = read_rows(csv_path)
    best = max(rows, key=lambda row: row["psnr"])
    final = rows[-1]
    return {
        "exp_id": exp.exp_id,
        "label": exp.label,
        "group": exp.group,
        "setup": exp.setup,
        "frozen": exp.frozen,
        "best_epoch": int(best["epoch"]),
        "best_psnr": best.get("psnr"),
        "best_ssim": best.get("ssim"),
        "best_lpips": best.get("lpips"),
        "final_epoch": int(final["epoch"]),
        "final_psnr": final.get("psnr"),
        "final_ssim": final.get("ssim"),
        "csv_path": exp.csv_path,
        "notes": exp.notes,
        "rows": rows,
    }


def write_summary_csv(summary_rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "group", "label", "setup", "frozen",
        "best_epoch", "best_psnr", "best_ssim", "best_lpips",
        "final_epoch", "final_psnr", "final_ssim",
        "csv_path", "notes",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_markdown(summary_rows: List[Dict[str, object]], out_md: Path) -> None:
    groups = []
    for row in summary_rows:
        if row["group"] not in groups:
            groups.append(row["group"])

    with out_md.open("w") as handle:
        handle.write("# Rain100L Results Summary\n\n")
        for group in groups:
            handle.write(f"## {group}\n\n")
            handle.write("| Experiment | Setup | Frozen | Best Epoch | Best PSNR | Best SSIM | Best LPIPS | Notes |\n")
            handle.write("|---|---|---|---:|---:|---:|---:|---|\n")
            for row in summary_rows:
                if row["group"] != group:
                    continue
                handle.write(
                    f"| {row['label']} | {row['setup']} | {row['frozen']} | "
                    f"{row['best_epoch']} | {fmt(row['best_psnr'])} | {fmt(row['best_ssim'])} | "
                    f"{fmt(row['best_lpips'])} | {row['notes'] or '-'} |\n"
                )
            handle.write("\n")


def _nice_bounds(values: List[float]) -> Tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return vmin - 1.0, vmax + 1.0
    pad = 0.08 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _draw_axes(draw: ImageDraw.ImageDraw, width: int, height: int, margin_left: int,
               margin_right: int, margin_top: int, margin_bottom: int) -> None:
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)


def _draw_title(draw: ImageDraw.ImageDraw, title: str, width: int, font) -> None:
    draw.text((width // 2, 18), title, fill="black", font=font, anchor="ma")


def _draw_legend(draw: ImageDraw.ImageDraw, labels: List[str], colors: List[str], x: int, y: int, font) -> None:
    line_h = 22
    for idx, (label, color) in enumerate(zip(labels, colors)):
        yy = y + idx * line_h
        draw.rectangle((x, yy + 4, x + 14, yy + 18), fill=color, outline=color)
        draw.text((x + 22, yy), label, fill="black", font=font)


def plot_metric_curves(summary_rows: List[Dict[str, object]], exp_ids: List[str], metric: str,
                       title: str, out_path: Path) -> None:
    selected = [row for row in summary_rows if row["exp_id"] in exp_ids]
    series = []
    for row in selected:
        metric_rows = [entry for entry in row["rows"] if metric in entry]
        if metric_rows:
            series.append((row["label"], metric_rows))
    if not series:
        return

    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 95, 260, 70, 85
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    font = ImageFont.load_default()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_axes(draw, width, height, margin_left, margin_right, margin_top, margin_bottom)
    _draw_title(draw, title, width, font)

    all_epochs = [entry["epoch"] for _, rows in series for entry in rows]
    all_values = [entry[metric] for _, rows in series for entry in rows]
    x_min, x_max = min(all_epochs), max(all_epochs)
    y_min, y_max = _nice_bounds(all_values)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    labels_used = []
    colors_used = []

    for idx, (label, rows) in enumerate(series):
        color = colors[idx % len(colors)]
        labels_used.append(label)
        colors_used.append(color)
        points = []
        for entry in rows:
            x = margin_left + ((entry["epoch"] - x_min) / max(1, x_max - x_min)) * chart_w
            y = margin_top + (1.0 - ((entry[metric] - y_min) / max(1e-8, y_max - y_min))) * chart_h
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points[::max(1, len(points) // 25)]:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)

    for tick_idx in range(6):
        frac = tick_idx / 5
        y = margin_top + (1.0 - frac) * chart_h
        value = y_min + frac * (y_max - y_min)
        draw.line((margin_left - 6, y, margin_left, y), fill="black", width=1)
        draw.text((margin_left - 10, y), f"{value:.2f}", fill="black", font=font, anchor="ra")

    for tick_idx in range(6):
        frac = tick_idx / 5
        x = margin_left + frac * chart_w
        value = x_min + frac * (x_max - x_min)
        draw.line((x, height - margin_bottom, x, height - margin_bottom + 6), fill="black", width=1)
        draw.text((x, height - margin_bottom + 12), f"{value:.0f}", fill="black", font=font, anchor="ma")

    draw.text((width // 2 - 110, height - 38), "Epoch", fill="black", font=font)
    draw.text((24, height // 2), metric.upper(), fill="black", font=font)
    _draw_legend(draw, labels_used, colors_used, width - margin_right + 20, margin_top + 20, font)
    image.save(out_path)
def plot_best_bar(summary_rows: List[Dict[str, object]], exp_ids: List[str], metric: str,
                  title: str, out_path: Path) -> None:
    selected = [row for row in summary_rows if row["exp_id"] in exp_ids and row.get(f"best_{metric}") is not None]
    if not selected:
        return

    labels = [row["label"] for row in selected]
    values = [row[f"best_{metric}"] for row in selected]
    width, height = 1300, 720
    margin_left, margin_right, margin_top, margin_bottom = 95, 40, 70, 170
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    font = ImageFont.load_default()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_axes(draw, width, height, margin_left, margin_right, margin_top, margin_bottom)
    _draw_title(draw, title, width, font)

    y_min, y_max = _nice_bounds(values)
    bar_count = len(values)
    bar_slot = chart_w / max(1, bar_count)
    bar_width = max(16, bar_slot * 0.55)
    color = "#4e79a7"

    for idx, (label, value) in enumerate(zip(labels, values)):
        x_center = margin_left + (idx + 0.5) * bar_slot
        bar_h = ((value - y_min) / max(1e-8, y_max - y_min)) * chart_h
        x0 = x_center - bar_width / 2
        y0 = margin_top + chart_h - bar_h
        x1 = x_center + bar_width / 2
        y1 = margin_top + chart_h
        draw.rectangle((x0, y0, x1, y1), fill=color, outline="black")
        draw.text((x_center, y0 - 10), f"{value:.2f}", fill="black", font=font, anchor="ms")
        draw.text((x_center, y1 + 12), label, fill="black", font=font, anchor="ma")

    for tick_idx in range(6):
        frac = tick_idx / 5
        y = margin_top + (1.0 - frac) * chart_h
        value = y_min + frac * (y_max - y_min)
        draw.line((margin_left - 6, y, margin_left, y), fill="black", width=1)
        draw.text((margin_left - 10, y), f"{value:.2f}", fill="black", font=font, anchor="ra")

    draw.text((24, height // 2), metric.upper(), fill="black", font=font)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RAINet experiment CSVs and make plots.")
    parser.add_argument("--repo_root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--out_dir", type=Path, default=Path("analysis_outputs"), help="Output directory")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [summarize_experiment(exp, repo_root) for exp in EXPERIMENTS]

    write_summary_csv(summary_rows, out_dir / "results_summary.csv")
    write_summary_markdown(summary_rows, out_dir / "results_summary.md")

    plot_metric_curves(
        summary_rows,
        ["additive_v2", "concat_v1"],
        "psnr",
        "Selective Strip Fusion (Run A): PSNR vs Epoch",
        plots_dir / "selective_strip_run_a_psnr.png",
    )
    plot_metric_curves(
        summary_rows,
        ["additive_v3", "concat_v2", "gated_v1"],
        "psnr",
        "Selective Strip Fusion (Run B): PSNR vs Epoch",
        plots_dir / "selective_strip_run_b_psnr.png",
    )
    plot_metric_curves(
        summary_rows,
        ["baseline_frozen", "gated_frozen", "channel_attn_frozen", "self_attn_frozen", "swin_attn_frozen"],
        "psnr",
        "Frozen Backbone and Mechanism Swap: PSNR vs Epoch",
        plots_dir / "frozen_and_mechswap_psnr.png",
    )
    plot_best_bar(
        summary_rows,
        ["baseline_frozen", "gated_frozen", "channel_attn_frozen", "self_attn_frozen", "swin_attn_frozen"],
        "psnr",
        "Best PSNR: Frozen Backbone Comparison and Mechanism Swap",
        plots_dir / "best_psnr_frozen_and_mechswap.png",
    )
    plot_best_bar(
        summary_rows,
        [row["exp_id"] for row in summary_rows],
        "psnr",
        "Best PSNR Across All Experiments",
        plots_dir / "best_psnr_all_experiments.png",
    )

    print(f"Saved summary CSV to {out_dir / 'results_summary.csv'}")
    print(f"Saved summary markdown to {out_dir / 'results_summary.md'}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
