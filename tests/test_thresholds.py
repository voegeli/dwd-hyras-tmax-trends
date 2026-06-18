from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dwd_hyras.stations import compute_station_threshold_day_counts_from_files
from dwd_hyras.thresholds import (
    compute_threshold_day_counts,
    compute_threshold_day_counts_from_files,
    write_threshold_interactive_html,
)


def test_compute_threshold_day_counts_handles_cftime_times():
    cftime = pytest.importorskip("cftime")
    times = [cftime.DatetimeGregorian(1951, 6, 1), cftime.DatetimeGregorian(1951, 6, 2)]
    tasmax = xr.DataArray(
        np.array([[31.0, 28.0], [29.0, 36.0]]),
        dims=["time", "x"],
        coords={"time": times},
        attrs={"units": "degC"},
    )
    result = compute_threshold_day_counts(tasmax, thresholds=[30.0])
    assert result.iloc[0]["days_ge_30_0c"] == 2


def test_compute_threshold_day_counts_applies_scenario_masks():
    tasmax = xr.DataArray(
        np.array(
            [
                [[31.0, 29.0], [28.0, 27.0]],
                [[29.0, 28.0], [36.0, 27.0]],
            ]
        ),
        dims=["time", "y", "x"],
        coords={"time": pd.to_datetime(["1951-06-01", "1951-06-02"])},
        attrs={"units": "degC"},
    )
    all_cells = xr.DataArray(
        np.array([[True, True], [True, True]]),
        dims=["y", "x"],
    )
    exclude_hot_cells = xr.DataArray(
        np.array([[False, True], [False, True]]),
        dims=["y", "x"],
    )

    result = compute_threshold_day_counts(
        tasmax,
        thresholds=[30.0],
        scenario_masks={
            "all": all_cells,
            "without_peaks": exclude_hot_cells,
        },
    )

    row = result.iloc[0]
    assert row["days_ge_30_0c"] == 2
    assert row["days_ge_30_0c__without_peaks"] == 0


def test_compute_threshold_day_counts_from_files_raises_on_duplicate_years():
    from unittest.mock import MagicMock, patch

    tasmax = xr.DataArray(
        np.array([[31.0, 32.0]]),
        dims=["time", "x"],
        coords={
            "time": pd.to_datetime(["1951-06-01"]),
            "lat": ("x", np.array([50.0, 51.0])),
            "lon": ("x", np.array([8.0, 9.0])),
        },
        attrs={"units": "degC"},
    )
    fake_ds = tasmax.to_dataset(name="tasmax")

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=fake_ds)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("dwd_hyras.thresholds.xr.open_dataset", return_value=mock_ctx):
        with pytest.raises(ValueError, match="1951"):
            compute_threshold_day_counts_from_files(["fake1.nc", "fake2.nc"], thresholds=[30.0])


def test_threshold_html_renders_station_data_in_the_main_chart(tmp_path):
    metrics = pd.DataFrame(
        {
            "year": [1951, 1952, 1953],
            "days_ge_30_0c": [2, 4, 6],
        }
    )
    station_metrics = pd.DataFrame(
        {
            "year": [1950, 1951, 1952, 1953, 1954],
            "days_ge_30_0c": [1, 3, 5, 7, 9],
            "days_ge_30_0c__no_airports": [1, 2, 4, 6, 8],
        }
    )
    output = tmp_path / "threshold.html"

    write_threshold_interactive_html(metrics, output, station_metrics=station_metrics)

    html = output.read_text(encoding="utf-8")
    assert 'id="stationChart"' not in html
    assert 'id="stationPanel"' not in html
    assert "stationSeriesForYears" in html
    assert "DWD-Stationen" in html
    assert "const years = DATA.years;" in html
    assert '"no_airports":{"30.0":[1,2,4,6,8]}' in html
    assert "HYRAS deckt Deutschland als dichtes Raster ab" in html
    assert 'class="trend station"' in html
    assert "HYRAS-Trend" in html
    assert "DWD-Stationen-Trend" in html


def test_station_threshold_counts_apply_airport_and_city_filters(tmp_path):
    import zipfile

    def write_station_zip(station_id: str, value: float) -> str:
        path = tmp_path / f"tageswerte_KL_{station_id}_hist.zip"
        product = (
            "STATIONS_ID;MESS_DATUM;TXK;\n"
            f"{int(station_id)};19510601;{value};\n"
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"produkt_klima_tag_{station_id}.txt", product)
        return str(path)

    paths = [
        write_station_zip("00001", 35.0),
        write_station_zip("00002", 34.0),
        write_station_zip("00003", 33.0),
    ]
    station_locations = {
        "00001": (48.3538, 11.7861),  # Munich Airport
        "00002": (52.5200, 13.4050),  # Berlin
        "00003": (50.0, 10.0),
    }

    result = compute_station_threshold_day_counts_from_files(
        paths,
        thresholds=[34.0],
        station_locations=station_locations,
        city_buffer_km=20.0,
        airport_buffer_km=10.0,
    )

    row = result.iloc[0]
    assert row["days_ge_34_0c"] == 1
    assert row["days_ge_34_0c__no_airports"] == 1
    assert row["days_ge_34_0c__no_cities"] == 1
    assert row["days_ge_34_0c__no_airports_no_cities"] == 0
