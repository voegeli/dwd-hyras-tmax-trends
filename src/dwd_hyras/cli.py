from __future__ import annotations

import argparse
import glob as _glob
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from dwd_hyras.download import (
    DWD_DAILY_KL_HISTORICAL_URL,
    DWD_DAILY_KL_100_YEAR_TXK_URL,
    DWD_HYRAS_TASMAX_URL,
    download_files,
    list_netcdf_urls,
    list_station_locations_from_overview,
    list_station_ids_from_overview,
    list_zip_urls,
)
from dwd_hyras.plot import plot_annual_max_tmax, plot_annual_mean_tmax, plot_hot_area_days_30
from dwd_hyras.process import compute_annual_metrics, open_tasmax, write_annual_metrics_csv
from dwd_hyras.stations import compute_station_threshold_day_counts_from_files
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
    threshold_parser.add_argument("--city-buffer-km", type=float, default=20.0)
    threshold_parser.add_argument("--airport-buffer-km", type=float, default=10.0)
    threshold_parser.add_argument(
        "--station-inputs",
        nargs="+",
        help="Optional DWD daily KL station ZIP files or glob patterns. Downloads station data if omitted or unmatched.",
    )
    threshold_parser.add_argument("--station-dir", default="data/stations")
    threshold_parser.add_argument("--station-url", default=DWD_DAILY_KL_HISTORICAL_URL)
    threshold_parser.add_argument("--station-overview-url", default=DWD_DAILY_KL_100_YEAR_TXK_URL)
    threshold_parser.add_argument("--skip-station-chart", action="store_true")

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
        hyras_counts_path = output_dir / "hot_days_threshold_counts.csv"
        station_counts_path = output_dir / "hot_days_threshold_station_counts.csv"
        metrics = _cached_threshold_counts(hyras_counts_path, thresholds)
        if metrics is None:
            print(
                f"Computing HYRAS threshold counts for {len(input_paths)} file(s) "
                f"and {len(thresholds)} threshold(s).",
                flush=True,
            )
            metrics = compute_threshold_day_counts_from_files(
                input_paths,
                thresholds,
                variable=args.variable,
                city_buffer_km=args.city_buffer_km,
                airport_buffer_km=args.airport_buffer_km,
            )
            print("HYRAS threshold counts complete.", flush=True)
        else:
            print(f"Using cached HYRAS threshold counts from {hyras_counts_path}.", flush=True)

        station_metrics = None
        if not args.skip_station_chart:
            station_metrics = _cached_threshold_counts(station_counts_path, thresholds, require_scenarios=True)
            if station_metrics is None:
                station_locations = list_station_locations_from_overview(args.station_overview_url)
                station_paths = _resolve_station_paths(
                    args.station_inputs,
                    Path(args.station_dir),
                    station_url=args.station_url,
                    station_overview_url=args.station_overview_url,
                )
                print(
                    f"Computing station threshold counts for {len(station_paths)} station ZIP file(s).",
                    flush=True,
                )
                station_metrics = compute_station_threshold_day_counts_from_files(
                    station_paths,
                    thresholds,
                    station_locations=station_locations,
                    city_buffer_km=args.city_buffer_km,
                    airport_buffer_km=args.airport_buffer_km,
                )
                print("Station threshold counts complete.", flush=True)
                write_threshold_counts_csv(station_metrics, station_counts_path)
            else:
                print(f"Using cached station threshold counts from {station_counts_path}.", flush=True)

        print(f"Writing threshold outputs to {output_dir}.", flush=True)
        write_threshold_counts_csv(metrics, hyras_counts_path)
        write_threshold_interactive_html(
            metrics,
            output_dir / "hot_days_threshold_interactive.html",
            station_metrics=station_metrics,
        )
        print("Threshold page complete.", flush=True)
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
        if any(char in item for char in "*?["):
            matches = [Path(p) for p in sorted(_glob.glob(item))]
        else:
            matches = [Path(item)]
        paths.extend(matches)

    if not paths:
        raise FileNotFoundError("No input NetCDF files matched.")
    return paths


def _expand_existing_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if any(char in item for char in "*?["):
            matches = [Path(p) for p in sorted(_glob.glob(item))]
        else:
            path = Path(item)
            matches = [path] if path.exists() else []
        paths.extend(matches)
    return paths


def _resolve_station_paths(
    station_inputs: list[str] | None,
    station_dir: Path,
    station_url: str = DWD_DAILY_KL_HISTORICAL_URL,
    station_overview_url: str = DWD_DAILY_KL_100_YEAR_TXK_URL,
    list_urls: Callable[[str], list[str]] = list_zip_urls,
    list_station_ids: Callable[[str], list[str]] = list_station_ids_from_overview,
    download: Callable[[list[str], str | Path], list[Path]] = download_files,
) -> list[Path]:
    if station_inputs:
        paths = _expand_existing_inputs(station_inputs)
        if paths:
            print(f"Using {len(paths)} station ZIP file(s) from --station-inputs.", flush=True)
            return paths

    cached = sorted(station_dir.glob("*.zip"))
    if cached:
        print(f"Using {len(cached)} cached station ZIP file(s) from {station_dir}.", flush=True)
        return cached

    print("No cached station ZIP files found.", flush=True)
    print(f"Loading DWD long-running TXK station overview: {station_overview_url}", flush=True)
    station_ids = list_station_ids(station_overview_url)
    print(f"Found {len(station_ids)} long-running TXK station(s).", flush=True)
    print(f"Loading DWD station ZIP directory: {station_url}", flush=True)
    urls = _match_station_urls(list_urls(station_url), station_ids)
    if not urls:
        raise FileNotFoundError(f"No station ZIP files found for the TXK station overview at {station_url}")
    print(f"Downloading {len(urls)} station ZIP file(s) into {station_dir}.", flush=True)
    return download(urls, station_dir)


def _match_station_urls(urls: list[str], station_ids: list[str]) -> list[str]:
    requested = {station_id.zfill(5) for station_id in station_ids}
    return [
        url
        for url in urls
        if any(f"_KL_{station_id}_" in url for station_id in requested)
    ]


def _cached_threshold_counts(
    path: Path,
    thresholds: list[float],
    require_scenarios: bool = False,
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    metrics = pd.read_csv(path)
    scenarios = ["all"]
    if require_scenarios:
        scenarios.extend(["no_airports", "no_cities", "no_airports_no_cities"])
    required = {
        "year",
        *(
            _station_threshold_column(threshold, scenario)
            for threshold in thresholds
            for scenario in scenarios
        ),
    }
    if required.issubset(metrics.columns):
        return metrics
    return None


def _station_threshold_column(threshold: float, scenario: str = "all") -> str:
    value = f"{threshold:.1f}".replace(".", "_")
    suffix = "" if scenario == "all" else f"__{scenario}"
    return f"days_ge_{value}c{suffix}"


def _threshold_range(min_threshold: float, max_threshold: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be greater than 0.")
    if max_threshold < min_threshold:
        raise ValueError("--max-threshold must be greater than or equal to --min-threshold.")

    n_steps = round((max_threshold - min_threshold) / step)
    return sorted({round(min_threshold + i * step, 1) for i in range(n_steps + 1)})


if __name__ == "__main__":
    main()
