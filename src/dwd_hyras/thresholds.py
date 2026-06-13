from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from dwd_hyras.process import DEFAULT_VARIABLE_CANDIDATES, normalize_to_celsius


@dataclass(frozen=True)
class ExclusionFeature:
    name: str
    lat: float
    lon: float


DEFAULT_CITY_FEATURES = (
    ExclusionFeature("Berlin", 52.5200, 13.4050),
    ExclusionFeature("Hamburg", 53.5511, 9.9937),
    ExclusionFeature("Munich", 48.1351, 11.5820),
    ExclusionFeature("Cologne", 50.9375, 6.9603),
    ExclusionFeature("Frankfurt am Main", 50.1109, 8.6821),
    ExclusionFeature("Stuttgart", 48.7758, 9.1829),
    ExclusionFeature("Dusseldorf", 51.2277, 6.7735),
    ExclusionFeature("Leipzig", 51.3397, 12.3731),
    ExclusionFeature("Dortmund", 51.5136, 7.4653),
    ExclusionFeature("Essen", 51.4556, 7.0116),
    ExclusionFeature("Bremen", 53.0793, 8.8017),
    ExclusionFeature("Dresden", 51.0504, 13.7373),
    ExclusionFeature("Hanover", 52.3759, 9.7320),
    ExclusionFeature("Nuremberg", 49.4521, 11.0767),
    ExclusionFeature("Duisburg", 51.4344, 6.7623),
)


DEFAULT_AIRPORT_FEATURES = (
    ExclusionFeature("Frankfurt Airport", 50.0379, 8.5622),
    ExclusionFeature("Munich Airport", 48.3538, 11.7861),
    ExclusionFeature("Berlin Brandenburg Airport", 52.3667, 13.5033),
    ExclusionFeature("Dusseldorf Airport", 51.2895, 6.7668),
    ExclusionFeature("Hamburg Airport", 53.6304, 9.9882),
    ExclusionFeature("Cologne Bonn Airport", 50.8659, 7.1427),
    ExclusionFeature("Stuttgart Airport", 48.6899, 9.2219),
    ExclusionFeature("Hanover Airport", 52.4611, 9.6851),
    ExclusionFeature("Nuremberg Airport", 49.4987, 11.0669),
    ExclusionFeature("Leipzig Halle Airport", 51.4239, 12.2364),
    ExclusionFeature("Dortmund Airport", 51.5183, 7.6122),
    ExclusionFeature("Bremen Airport", 53.0475, 8.7867),
    ExclusionFeature("Dresden Airport", 51.1328, 13.7672),
    ExclusionFeature("Muenster Osnabrueck Airport", 52.1346, 7.6848),
    ExclusionFeature("Karlsruhe Baden-Baden Airport", 48.7793, 8.0805),
)


def compute_threshold_day_counts(
    tasmax_c: xr.DataArray,
    thresholds: Iterable[float],
    scenario_masks: dict[str, xr.DataArray] | None = None,
) -> pd.DataFrame:
    """Count days per year where any valid grid cell reaches each threshold."""
    if "time" not in tasmax_c.dims:
        raise ValueError("TASMAX data must include a 'time' dimension.")

    tasmax_c = normalize_to_celsius(tasmax_c)
    spatial_dims = tuple(dim for dim in tasmax_c.dims if dim != "time")
    if not spatial_dims:
        raise ValueError("TASMAX data must include at least one spatial dimension.")

    masks = scenario_masks or {"all": _full_spatial_mask(tasmax_c, spatial_dims)}
    daily_max_by_scenario = {
        scenario: tasmax_c.where(mask).max(dim=spatial_dims, skipna=True).load()
        for scenario, mask in masks.items()
    }
    first_daily_max = next(iter(daily_max_by_scenario.values()))
    years = first_daily_max["time"].dt.year.values.astype(int)
    unique_years = np.unique(years)

    data: dict[str, np.ndarray] = {"year": unique_years}
    for scenario, daily_max in daily_max_by_scenario.items():
        daily_values = daily_max.values
        for threshold in thresholds:
            reached = daily_values >= threshold
            data[_threshold_column(threshold, scenario)] = np.array(
                [int(reached[years == year].sum()) for year in unique_years],
                dtype=int,
            )

    return pd.DataFrame(data).sort_values("year").reset_index(drop=True)


def compute_threshold_day_counts_from_files(
    paths: Iterable[str | Path],
    thresholds: Iterable[float],
    variable: str | None = None,
    city_buffer_km: float = 20.0,
    airport_buffer_km: float = 10.0,
) -> pd.DataFrame:
    """Compute threshold counts by loading NetCDF files one at a time."""
    frames: list[pd.DataFrame] = []
    threshold_list = list(thresholds)
    for path in paths:
        with xr.open_dataset(path) as dataset:
            data_var = variable or _find_tasmax_variable(dataset)
            if data_var not in dataset:
                raise KeyError(f"Variable {data_var!r} not found in {path}.")
            frame = _compute_threshold_day_counts_for_dataset(
                dataset[data_var],
                threshold_list,
                city_buffer_km,
                airport_buffer_km,
            )
            frames.append(frame)

    if not frames:
        raise ValueError("At least one NetCDF path is required.")

    combined = pd.concat(frames, ignore_index=True).sort_values("year").reset_index(drop=True)
    duplicate_years = combined.loc[combined.duplicated(subset=["year"]), "year"].unique().tolist()
    if duplicate_years:
        raise ValueError(
            f"Overlapping years found in input files: {sorted(duplicate_years)}. "
            "Each year must appear in exactly one input file."
        )
    return combined


def _compute_threshold_day_counts_for_dataset(
    tasmax: xr.DataArray,
    thresholds: list[float],
    city_buffer_km: float,
    airport_buffer_km: float,
) -> pd.DataFrame:
    tasmax_c = normalize_to_celsius(tasmax)
    if "time" not in tasmax_c.dims:
        raise ValueError("TASMAX data must include a 'time' dimension.")

    spatial_dims = tuple(dim for dim in tasmax_c.dims if dim != "time")
    masks = build_default_scenario_masks(tasmax_c, city_buffer_km, airport_buffer_km)
    values = tasmax_c.transpose("time", *spatial_dims).values
    flat_values = values.reshape(values.shape[0], -1)
    years = tasmax_c["time"].dt.year.values.astype(int)
    unique_years = np.unique(years)

    data: dict[str, np.ndarray] = {"year": unique_years}
    for scenario, mask in masks.items():
        flat_mask = mask.transpose(*spatial_dims).values.reshape(-1)
        daily_max = np.nanmax(flat_values[:, flat_mask], axis=1)
        for threshold in thresholds:
            reached = daily_max >= threshold
            data[_threshold_column(threshold, scenario)] = np.array(
                [int(reached[years == year].sum()) for year in unique_years],
                dtype=int,
            )

    return pd.DataFrame(data).sort_values("year").reset_index(drop=True)


def write_threshold_counts_csv(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)


def write_threshold_interactive_html(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parsed_columns = [_parse_threshold_column(column) for column in metrics.columns]
    parsed_columns = [item for item in parsed_columns if item is not None]
    scenarios = sorted({scenario for _, scenario in parsed_columns}, key=_scenario_sort_key)
    thresholds = sorted({threshold for threshold, _ in parsed_columns})
    payload = {
        "years": metrics["year"].astype(int).tolist(),
        "thresholds": thresholds,
        "scenarioLabels": {
            "all": "Alle Rasterzellen",
            "no_airports": "Flughafenumfeld ausgeschlossen",
            "no_cities": "Großstadtumfeld ausgeschlossen",
            "no_airports_no_cities": "Flughafen- und Großstadtumfeld ausgeschlossen",
        },
        "series": {
            scenario: {
                _threshold_key(threshold): metrics[_threshold_column(threshold, scenario)].astype(int).tolist()
                for threshold in thresholds
            }
            for scenario in scenarios
        },
    }

    output.write_text(_html_template(payload), encoding="utf-8")


def build_default_scenario_masks(
    tasmax_c: xr.DataArray,
    city_buffer_km: float = 20.0,
    airport_buffer_km: float = 10.0,
) -> dict[str, xr.DataArray]:
    """Build raster masks for all checkbox combinations used by the HTML page."""
    city_exclusion = _feature_exclusion_mask(tasmax_c, DEFAULT_CITY_FEATURES, city_buffer_km)
    airport_exclusion = _feature_exclusion_mask(tasmax_c, DEFAULT_AIRPORT_FEATURES, airport_buffer_km)
    full = xr.ones_like(city_exclusion, dtype=bool)
    return {
        "all": full,
        "no_airports": ~airport_exclusion,
        "no_cities": ~city_exclusion,
        "no_airports_no_cities": ~(airport_exclusion | city_exclusion),
    }


def _full_spatial_mask(tasmax_c: xr.DataArray, spatial_dims: tuple[str, ...]) -> xr.DataArray:
    template = tasmax_c.isel(time=0, drop=True) if "time" in tasmax_c.dims else tasmax_c
    return xr.ones_like(template, dtype=bool).transpose(*spatial_dims)


def _feature_exclusion_mask(
    tasmax_c: xr.DataArray,
    features: Iterable[ExclusionFeature],
    radius_km: float,
) -> xr.DataArray:
    if radius_km < 0:
        raise ValueError("Exclusion buffer radius must not be negative.")
    if "lat" not in tasmax_c.coords or "lon" not in tasmax_c.coords:
        raise ValueError("TASMAX data must include 'lat' and 'lon' coordinates for scenario masks.")

    lat = tasmax_c["lat"]
    lon = tasmax_c["lon"]
    mask = xr.zeros_like(lat, dtype=bool)
    for feature in features:
        mask = mask | (_haversine_km(lat, lon, feature.lat, feature.lon) <= radius_km)
    return mask


def _haversine_km(lat: xr.DataArray, lon: xr.DataArray, center_lat: float, center_lon: float) -> xr.DataArray:
    earth_radius_km = 6371.0088
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    lat2 = np.deg2rad(center_lat)
    lon2 = np.deg2rad(center_lon)
    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def _threshold_column(threshold: float, scenario: str = "all") -> str:
    value = f"{threshold:.1f}".replace(".", "_")
    suffix = "" if scenario == "all" else f"__{scenario}"
    return f"days_ge_{value}c{suffix}"


def _column_threshold(column: str) -> float:
    column = column.split("__", 1)[0]
    value = column.removeprefix("days_ge_").removesuffix("c").replace("_", ".")
    return float(value)


def _parse_threshold_column(column: str) -> tuple[float, str] | None:
    if not column.startswith("days_ge_"):
        return None
    base, _, scenario = column.partition("__")
    if not base.endswith("c"):
        return None
    return _column_threshold(base), scenario or "all"


def _scenario_sort_key(scenario: str) -> tuple[int, str]:
    order = {
        "all": 0,
        "no_airports": 1,
        "no_cities": 2,
        "no_airports_no_cities": 3,
    }
    return order.get(scenario, 99), scenario


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
    .scenario-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 8px 0 16px;
    }}
    .scenario-controls label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
      border: 1px solid #d8d8d2;
      border-radius: 8px;
      padding: 7px 10px;
      background: #fbfbf8;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
      text-transform: none;
      cursor: pointer;
    }}
    .scenario-controls input {{
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }}
    .scenario-note {{
      margin: -4px 0 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .method-note {{
      margin: 0 0 14px;
      padding: 10px 12px;
      border-left: 3px solid var(--accent);
      background: #fff8ef;
      color: #4f4b45;
      font-size: 13px;
      line-height: 1.42;
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
      <div class="scenario-controls" aria-label="Rasterzellen ausschließen">
        <label><input id="excludeAirports" type="checkbox"> Flughafenumfeld ausschließen</label>
        <label><input id="excludeCities" type="checkbox"> Großstadtumfeld ausschließen</label>
      </div>
      <p id="scenarioLabel" class="scenario-note"></p>
      <p class="method-note">Sensitivitätsanalyse: ausgeschlossen werden HYRAS-Rasterzellen im 10-km-Puffer um 15 große Flughäfen und/oder im 20-km-Puffer um 15 Großstädte. Gezählt wird ein Tag weiterhin, sobald außerhalb der ausgeschlossenen Bereiche irgendwo in Deutschland mindestens eine Rasterzelle die gewählte Schwelle erreicht.</p>
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
    const excludeAirports = document.getElementById("excludeAirports");
    const excludeCities = document.getElementById("excludeCities");
    const scenarioLabel = document.getElementById("scenarioLabel");
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

    function activeScenario() {{
      if (excludeAirports.checked && excludeCities.checked) return "no_airports_no_cities";
      if (excludeAirports.checked) return "no_airports";
      if (excludeCities.checked) return "no_cities";
      return "all";
    }}

    function render() {{
      const threshold = thresholdAtSlider();
      const scenario = activeScenario();
      const series = DATA.series[scenario][threshold.toFixed(1)];
      thresholdLabel.textContent = threshold.toFixed(1);
      countLabel.textContent = `Schwellen: ${{DATA.thresholds[0].toFixed(1)}}-${{DATA.thresholds.at(-1).toFixed(1)}} °C`;
      scenarioLabel.textContent = `Aktive Datenbasis: ${{DATA.scenarioLabels[scenario] || scenario}}. Es wird weiterhin nur eine Balkenreihe angezeigt.`;

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
          tooltip.textContent = `${{bar.dataset.year}}: ${{bar.dataset.value}} Tage >= ${{threshold.toFixed(1)}} °C (${{DATA.scenarioLabels[scenario] || scenario}})`;
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
    excludeAirports.addEventListener("change", render);
    excludeCities.addEventListener("change", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""
