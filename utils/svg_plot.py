import argparse
import json
import math
import os
from typing import Iterable, List, Sequence, Tuple


Palette = [
    "#2E86AB",
    "#E67E22",
    "#AF7AC5",
    "#16A085",
    "#D63031",
    "#6C5CE7",
    "#00B894",
    "#E84393",
]


def _detect_axis(log_data: Sequence[dict]) -> Tuple[str, str]:
    if not log_data:
        raise ValueError("Log data is empty")
    if "update" in log_data[0]:
        return "update", "Update"
    if "episode" in log_data[0]:
        return "episode", "Episode"
    raise KeyError("Log data must contain either 'update' or 'episode' keys")


def _scale(value: float, min_value: float, max_value: float, span: float, offset: float) -> float:
    if math.isclose(max_value, min_value):
        return offset
    return offset + (value - min_value) / (max_value - min_value) * span


def _ticks(min_value: float, max_value: float, count: int) -> List[float]:
    if count <= 1:
        return [min_value]
    if math.isclose(min_value, max_value):
        return [min_value] * count
    step = (max_value - min_value) / (count - 1)
    return [min_value + i * step for i in range(count)]


def _polyline(points: Iterable[Tuple[float, float]], color: str, width: float = 2.0) -> str:
    path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{path}" />'


def build_svg(
    series: Sequence[Tuple[str, List[Tuple[float, float]]]],
    x_label: str,
    y_label: str,
    title: str,
    width: int = 900,
    height: int = 520,
    margin: int = 60,
) -> str:
    all_x = [x for _, points in series for x, _ in points]
    all_y = [y for _, points in series for _, y in points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    inner_w = width - 2 * margin
    inner_h = height - 2 * margin
    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="{margin/2:.1f}" text-anchor="middle" font-size="18" fill="#2c3e50">{title}</text>',
    ]

    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    svg_parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#2c3e50" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#2c3e50" stroke-width="2"/>')

    for tick in _ticks(min_x, max_x, 6):
        x = _scale(tick, min_x, max_x, inner_w, margin)
        svg_parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0+6}" stroke="#95a5a6" stroke-width="1"/>')
        svg_parts.append(f'<text x="{x:.2f}" y="{y0+20}" text-anchor="middle" font-size="12" fill="#34495e">{tick:.2f}</text>')

    for tick in _ticks(min_y, max_y, 6):
        y = _scale(tick, min_y, max_y, inner_h, margin)
        y = height - y
        svg_parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0-6}" y2="{y:.2f}" stroke="#95a5a6" stroke-width="1"/>')
        svg_parts.append(f'<text x="{x0-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#34495e">{tick:.2f}</text>')

    svg_parts.append(f'<text x="{(x0 + x1)/2:.1f}" y="{height - margin/3:.1f}" text-anchor="middle" font-size="14" fill="#2c3e50">{x_label}</text>')
    svg_parts.append(f'<text x="{margin/3:.1f}" y="{(y0 + y1)/2:.1f}" text-anchor="middle" font-size="14" fill="#2c3e50" transform="rotate(-90 {margin/3:.1f},{(y0 + y1)/2:.1f})">{y_label}</text>')

    for idx, (label, points) in enumerate(series):
        color = Palette[idx % len(Palette)]
        scaled = [
            (
                _scale(x, min_x, max_x, inner_w, margin),
                height - _scale(y, min_y, max_y, inner_h, margin),
            )
            for x, y in points
        ]
        svg_parts.append(_polyline(scaled, color))
        legend_x = width - margin + 10
        legend_y = margin + idx * 18
        svg_parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}" />')
        svg_parts.append(f'<text x="{legend_x + 20}" y="{legend_y+2}" font-size="12" fill="#2c3e50">{label}</text>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def load_series(log_path: str, metric: str) -> Tuple[str, List[Tuple[float, float]], str, str]:
    with open(log_path, "r") as f:
        log_data = json.load(f)
    axis_key, x_label = _detect_axis(log_data)
    y_label = metric.replace("_", " ").title()
    points = [(entry[axis_key], entry[metric]) for entry in log_data if metric in entry]
    label = os.path.splitext(os.path.basename(log_path))[0]
    return label, points, x_label, y_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight SVG plots from log files without external plotting libraries.")
    parser.add_argument("--logs", nargs="+", required=True, help="One or more JSON log files (log_data_*.json)")
    parser.add_argument("--metric", default="reward", help="Metric to plot (reward, latency, energy, fairness, etc.)")
    parser.add_argument("--output", required=True, help="Path to the output SVG file")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    loaded: List[Tuple[str, List[Tuple[float, float]]]] = []
    x_label = "Step"
    y_label = args.metric.replace("_", " ").title()
    for log in args.logs:
        label, points, x_label, y_label = load_series(log, args.metric)
        if not points:
            raise ValueError(f"No data for metric '{args.metric}' in {log}")
        loaded.append((label, points))

    title = args.title or f"{args.metric.replace('_', ' ').title()} Comparison"
    svg = build_svg(loaded, x_label, y_label, title)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(svg)
    print(f"✅ SVG plot saved to {args.output}")


if __name__ == "__main__":
    main()
