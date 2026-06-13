from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from dwd_hyras.process import DEFAULT_VARIABLE_CANDIDATES, normalize_to_celsius


def compute_threshold_day_counts(
    tasmax_c: xr.DataArray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Count days per year where any valid grid cell reaches each threshold."""
    if "time" not in tasmax_c.dims:
        raise ValueError("TASMAX data must include a 'time' dimension.")

    tasmax_c = normalize_to_celsius(tasmax_c)
    spatial_dims = tuple(dim for dim in tasmax_c.dims if dim != "time")
    if not spatial_dims:
        raise ValueError("TASMAX data must include at least one spatial dimension.")

    daily_max = tasmax_c.max(dim=spatial_dims, skipna=True).load()
    years = pd.DatetimeIndex(daily_max["time"].values).year.astype(int)
    unique_years = np.unique(years)

    data: dict[str, np.ndarray] = {"year": unique_years}
    daily_values = daily_max.values
    for threshold in thresholds:
        reached = daily_values >= threshold
        data[_threshold_column(threshold)] = np.array(
            [int(reached[years == year].sum()) for year in unique_years],
            dtype=int,
        )

    return pd.DataFrame(data).sort_values("year").reset_index(drop=True)


def compute_threshold_day_counts_from_files(
    paths: Iterable[str | Path],
    thresholds: Iterable[float],
    variable: str | None = None,
) -> pd.DataFrame:
    """Compute threshold counts by loading NetCDF files one at a time."""
    frames: list[pd.DataFrame] = []
    threshold_list = list(thresholds)
    for path in paths:
        with xr.open_dataset(path) as dataset:
            data_var = variable or _find_tasmax_variable(dataset)
            if data_var not in dataset:
                raise KeyError(f"Variable {data_var!r} not found in {path}.")
            frame = compute_threshold_day_counts(dataset[data_var], threshold_list)
            frames.append(frame)

    if not frames:
        raise ValueError("At least one NetCDF path is required.")

    return pd.concat(frames, ignore_index=True).sort_values("year").reset_index(drop=True)


def write_threshold_counts_csv(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)


def write_threshold_interactive_html(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [
        _column_threshold(column)
        for column in metrics.columns
        if column.startswith("days_ge_") and column.endswith("c")
    ]
    thresholds = sorted(thresholds)
    payload = {
        "years": metrics["year"].astype(int).tolist(),
        "thresholds": thresholds,
        "series": {
            _threshold_key(threshold): metrics[_threshold_column(threshold)].astype(int).tolist()
            for threshold in thresholds
        },
    }

    output.write_text(_html_template(payload), encoding="utf-8")


def _threshold_column(threshold: float) -> str:
    value = f"{threshold:.1f}".replace(".", "_")
    return f"days_ge_{value}c"


def _column_threshold(column: str) -> float:
    value = column.removeprefix("days_ge_").removesuffix("c").replace("_", ".")
    return float(value)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.1f}"


def _find_tasmax_variable(dataset: xr.Dataset) -> str:
    for candidate in DEFAULT_VARIABLE_CANDIDATES:
        if candidate in dataset.data_vars:
            return candidate

    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    raise KeyError(
        "Could not infer TASMAX variable. Pass --variable explicitly. "
        f"Available variables: {', '.join(data_vars)}"
    )


def _html_template(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HYRAS Hitzetage nach Schwelle</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #62666d;
      --grid: #dedede;
      --accent: #f28e2b;
      --accent-dark: #c96112;
      --line: #1f77b4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(24px, 3vw, 38px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      max-width: 780px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
    }}
    .value {{
      min-width: 142px;
      text-align: right;
      font-size: 34px;
      font-weight: 700;
      color: var(--accent-dark);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid #ddddda;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px 18px;
      align-items: center;
      margin-bottom: 14px;
    }}
    label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .stat {{
      border-top: 1px solid #e3e3df;
      padding-top: 10px;
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
      margin-bottom: 3px;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 13px;
    }}
    svg {{
      display: block;
      width: 100%;
      height: min(56vw, 520px);
      min-height: 340px;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .axis path,
    .axis line {{
      stroke: #9a9a96;
      shape-rendering: crispEdges;
    }}
    .grid line {{
      stroke: var(--grid);
      stroke-opacity: 0.8;
      shape-rendering: crispEdges;
    }}
    .grid path {{ display: none; }}
    .bar {{ fill: var(--accent); }}
    .trend {{
      fill: none;
      stroke: var(--line);
      stroke-width: 2.5;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(32, 33, 36, 0.94);
      color: white;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 13px;
      transform: translate(-50%, -120%);
      opacity: 0;
      transition: opacity 120ms ease;
      white-space: nowrap;
    }}
    @media (max-width: 760px) {{
      header {{
        display: block;
      }}
      .value {{
        text-align: left;
        margin-top: 12px;
      }}
      .controls,
      .stats {{
        grid-template-columns: 1fr;
      }}
      svg {{
        min-height: 300px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>HYRAS-Hitzetage nach Temperatur-Schwelle</h1>
        <p>Gezählt werden Tage, an denen irgendwo in Deutschland mindestens eine gültige HYRAS-Rasterzelle die gewählte Tageshöchsttemperatur erreicht oder überschritten hat.</p>
      </div>
      <div class="value"><span id="thresholdLabel"></span> °C</div>
    </header>
    <section class="panel">
      <div class="controls">
        <label for="threshold">Lokale Tmax-Schwelle</label>
        <output id="countLabel"></output>
        <input id="threshold" type="range">
      </div>
      <svg id="chart" role="img" aria-label="Hitzetage pro Jahr nach Temperatur-Schwelle"></svg>
      <div class="stats">
        <div class="stat"><strong id="maxYear"></strong><span>Jahr mit den meisten Tagen</span></div>
        <div class="stat"><strong id="latestValue"></strong><span>Wert im letzten Jahr der Reihe</span></div>
        <div class="stat"><strong id="trendValue"></strong><span>Linearer Trend pro Jahrzehnt</span></div>
      </div>
    </section>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const DATA = {data_json};
    const slider = document.getElementById("threshold");
    const thresholdLabel = document.getElementById("thresholdLabel");
    const countLabel = document.getElementById("countLabel");
    const maxYear = document.getElementById("maxYear");
    const latestValue = document.getElementById("latestValue");
    const trendValue = document.getElementById("trendValue");
    const svg = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");

    slider.min = 0;
    slider.max = DATA.thresholds.length - 1;
    slider.step = 1;
    slider.value = DATA.thresholds.findIndex((value) => value === 30);
    if (slider.value < 0) slider.value = 0;

    function thresholdAtSlider() {{
      return DATA.thresholds[Number(slider.value)];
    }}

    function render() {{
      const threshold = thresholdAtSlider();
      const series = DATA.series[threshold.toFixed(1)];
      thresholdLabel.textContent = threshold.toFixed(1);
      countLabel.textContent = `Schwellen: ${{DATA.thresholds[0].toFixed(1)}}-${{DATA.thresholds.at(-1).toFixed(1)}} °C`;

      const width = svg.clientWidth || 1000;
      const height = svg.clientHeight || 480;
      const margin = {{ top: 22, right: 22, bottom: 42, left: 58 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const years = DATA.years;
      const maxY = Math.max(5, ...series);
      const yTop = Math.ceil(maxY / 10) * 10;
      const barWidth = Math.max(2, innerWidth / years.length * 0.82);

      const x = (year) => margin.left + ((year - years[0]) / (years.at(-1) - years[0])) * innerWidth;
      const y = (value) => margin.top + innerHeight - (value / yTop) * innerHeight;

      const ticksY = Array.from({{ length: 6 }}, (_, index) => Math.round((yTop / 5) * index));
      const ticksX = years.filter((year) => year % 10 === 0);
      const trend = linearTrend(years, series);
      const trendPath = years
        .map((year, index) => `${{index === 0 ? "M" : "L"}} ${{x(year)}} ${{y(trend.intercept + trend.slope * year)}}`)
        .join(" ");

      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `
        <g class="grid">
          ${{ticksY.map((tick) => `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y(tick)}}" y2="${{y(tick)}}"></line>`).join("")}}
        </g>
        <g>
          ${{years.map((year, index) => `
            <rect class="bar" x="${{x(year) - barWidth / 2}}" y="${{y(series[index])}}" width="${{barWidth}}" height="${{Math.max(0, margin.top + innerHeight - y(series[index]))}}" data-year="${{year}}" data-value="${{series[index]}}"></rect>
          `).join("")}}
        </g>
        <path class="trend" d="${{trendPath}}"></path>
        <g class="axis">
          <line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{margin.top + innerHeight}}" y2="${{margin.top + innerHeight}}"></line>
          <line x1="${{margin.left}}" x2="${{margin.left}}" y1="${{margin.top}}" y2="${{margin.top + innerHeight}}"></line>
          ${{ticksY.map((tick) => `<text x="${{margin.left - 10}}" y="${{y(tick) + 4}}" text-anchor="end">${{tick}}</text>`).join("")}}
          ${{ticksX.map((tick) => `<text x="${{x(tick)}}" y="${{height - 12}}" text-anchor="middle">${{tick}}</text>`).join("")}}
          <text x="${{margin.left}}" y="15">Tage pro Jahr</text>
        </g>
      `;

      svg.querySelectorAll(".bar").forEach((bar) => {{
        bar.addEventListener("mousemove", (event) => {{
          tooltip.textContent = `${{bar.dataset.year}}: ${{bar.dataset.value}} Tage >= ${{threshold.toFixed(1)}} °C`;
          tooltip.style.left = `${{event.clientX}}px`;
          tooltip.style.top = `${{event.clientY}}px`;
          tooltip.style.opacity = 1;
        }});
        bar.addEventListener("mouseleave", () => {{
          tooltip.style.opacity = 0;
        }});
      }});

      const maxIndex = series.reduce((best, value, index) => value > series[best] ? index : best, 0);
      maxYear.textContent = `${{years[maxIndex]}}: ${{series[maxIndex]}} Tage`;
      latestValue.textContent = `${{years.at(-1)}}: ${{series.at(-1)}} Tage`;
      trendValue.textContent = `${{(trend.slope * 10).toFixed(2)}} Tage`;
    }}

    function linearTrend(xs, ys) {{
      const n = xs.length;
      const meanX = xs.reduce((sum, value) => sum + value, 0) / n;
      const meanY = ys.reduce((sum, value) => sum + value, 0) / n;
      let numerator = 0;
      let denominator = 0;
      for (let index = 0; index < n; index += 1) {{
        numerator += (xs[index] - meanX) * (ys[index] - meanY);
        denominator += (xs[index] - meanX) ** 2;
      }}
      const slope = denominator === 0 ? 0 : numerator / denominator;
      return {{ slope, intercept: meanY - slope * meanX }};
    }}

    slider.addEventListener("input", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""
