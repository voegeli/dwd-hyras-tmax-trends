from __future__ import annotations

import argparse
from pathlib import Path

from dwd_hyras.download import DWD_HYRAS_TASMAX_URL, download_files, list_netcdf_urls
from dwd_hyras.plot import plot_annual_max_tmax, plot_annual_mean_tmax, plot_hot_area_days_30
from dwd_hyras.process import compute_annual_metrics, open_tasmax, write_annual_metrics_csv
from dwd_hyras.thresholds import (
    compute_threshold_day_counts_from_files,
    write_threshold_counts_csv,
    write_threshold_interactive_html,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DWD HYRAS TASMAX NetCDF files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download HYRAS TASMAX NetCDF files.")
    download_parser.add_argument("--url", default=DWD_HYRAS_TASMAX_URL)
    download_parser.add_argument("--output-dir", default="data/raw")
    download_parser.add_argument("--overwrite", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Compute annual metrics and plots.")
    analyze_parser.add_argument("inputs", nargs="+", help="Input NetCDF files or glob patterns.")
    analyze_parser.add_argument("--variable", help="TASMAX variable name if it cannot be inferred.")
    analyze_parser.add_argument("--output-dir", default="outputs")
    analyze_parser.add_argument("--chunks-time", type=int, default=365)

    threshold_parser = subparsers.add_parser(
        "threshold-page",
        help="Create an interactive hot-day threshold HTML page.",
    )
    threshold_parser.add_argument("inputs", nargs="+", help="Input NetCDF files or glob patterns.")
    threshold_parser.add_argument("--variable", help="TASMAX variable name if it cannot be inferred.")
    threshold_parser.add_argument("--output-dir", default="outputs")
    threshold_parser.add_argument("--chunks-time", type=int, default=365)
    threshold_parser.add_argument("--min-threshold", type=float, default=25.0)
    threshold_parser.add_argument("--max-threshold", type=float, default=42.0)
    threshold_parser.add_argument("--step", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "download":
        urls = list_netcdf_urls(args.url)
        paths = download_files(urls, args.output_dir, overwrite=args.overwrite)
        for path in paths:
            print(path)
        return

    input_paths = _expand_inputs(args.inputs)
    output_dir = Path(args.output_dir)

    if args.command == "threshold-page":
        thresholds = _threshold_range(args.min_threshold, args.max_threshold, args.step)
        metrics = compute_threshold_day_counts_from_files(input_paths, thresholds, variable=args.variable)
        write_threshold_counts_csv(metrics, output_dir / "hot_days_threshold_counts.csv")
        write_threshold_interactive_html(metrics, output_dir / "hot_days_threshold_interactive.html")
        return

    chunks = None if args.chunks_time <= 0 else {"time": args.chunks_time}
    tasmax = open_tasmax(input_paths, variable=args.variable, chunks=chunks)
    metrics = compute_annual_metrics(tasmax)

    write_annual_metrics_csv(metrics, output_dir / "dwd_hyras_tasmax_analysis.csv")
    plot_annual_mean_tmax(metrics, output_dir / "annual_mean_tmax.png")
    plot_annual_max_tmax(metrics, output_dir / "annual_max_tmax.png")
    plot_hot_area_days_30(metrics, output_dir / "hot_area_days_30.png")


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = sorted(Path().glob(item)) if any(char in item for char in "*?[") else [Path(item)]
        paths.extend(matches)

    if not paths:
        raise FileNotFoundError("No input NetCDF files matched.")
    return paths


def _threshold_range(min_threshold: float, max_threshold: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be greater than 0.")
    if max_threshold < min_threshold:
        raise ValueError("--max-threshold must be greater than or equal to --min-threshold.")

    values: list[float] = []
    current = min_threshold
    while current <= max_threshold + (step / 10):
        values.append(round(current, 1))
        current += step
    return values


if __name__ == "__main__":
    main()
