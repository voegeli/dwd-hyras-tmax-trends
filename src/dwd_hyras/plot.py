from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_annual_mean_tmax(metrics: pd.DataFrame, output_path: str | Path) -> None:
    _plot_line(
        metrics,
        y="annual_mean_tmax_c",
        output_path=output_path,
        ylabel="Annual mean daily Tmax (degC)",
        title="Germany HYRAS annual mean daily maximum temperature",
    )


def plot_annual_max_tmax(metrics: pd.DataFrame, output_path: str | Path) -> None:
    _plot_bar(
        metrics,
        y="annual_max_tmax_c",
        output_path=output_path,
        ylabel="Highest daily Tmax anywhere in Germany (degC)",
        title="Germany HYRAS annual maximum daily temperature",
    )


def plot_hot_area_days_30(metrics: pd.DataFrame, output_path: str | Path) -> None:
    _plot_line(
        metrics,
        y="days_any_hot_30",
        output_path=output_path,
        ylabel="Days with at least one grid cell >= 30 degC",
        title="Germany HYRAS hot days >= 30 degC",
    )


def _plot_line(
    metrics: pd.DataFrame,
    y: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics["year"], metrics[y], color="#1f77b4", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_bar(
    metrics: pd.DataFrame,
    y: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics["year"], metrics[y], color="#f28e2b", width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
