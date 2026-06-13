import numpy as np
import pandas as pd
import xarray as xr

from dwd_hyras.process import compute_annual_metrics, normalize_to_celsius
from dwd_hyras.thresholds import compute_threshold_day_counts


def test_normalize_to_celsius_converts_kelvin_units():
    values = xr.DataArray([273.15, 303.15], dims=["time"], attrs={"units": "K"})

    result = normalize_to_celsius(values)

    np.testing.assert_allclose(result.values, [0.0, 30.0])
    assert result.attrs["units"] == "degC"


def test_compute_annual_metrics_counts_any_cell_hot_days_and_area_shares():
    tasmax = xr.DataArray(
        np.array(
            [
                [[29.0, 31.0], [28.0, np.nan]],
                [[34.0, 36.0], [30.0, 20.0]],
                [[10.0, 11.0], [12.0, 13.0]],
            ]
        ),
        dims=["time", "y", "x"],
        coords={"time": pd.to_datetime(["1951-06-01", "1951-06-02", "1952-07-01"])},
        attrs={"units": "degC"},
        name="tasmax",
    )

    result = compute_annual_metrics(tasmax)

    row_1951 = result.loc[result["year"] == 1951].iloc[0]
    assert row_1951["days_any_hot_30"] == 2
    assert row_1951["days_any_hot_35"] == 1
    assert row_1951["annual_max_tmax_c"] == 36.0
    assert row_1951["annual_mean_tmax_c"] == 29.666666666666664
    assert row_1951["mean_hot_area_share_30"] == 0.5416666666666666
    assert row_1951["max_hot_area_share_30"] == 0.75
    assert row_1951["mean_hot_area_share_35"] == 0.125
    assert row_1951["max_hot_area_share_35"] == 0.25

    row_1952 = result.loc[result["year"] == 1952].iloc[0]
    assert row_1952["days_any_hot_30"] == 0
    assert row_1952["days_any_hot_35"] == 0


def test_compute_threshold_day_counts_counts_local_daily_maxima_by_year():
    tasmax = xr.DataArray(
        np.array(
            [
                [[29.0, 31.0], [28.0, np.nan]],
                [[34.0, 36.0], [30.0, 20.0]],
                [[39.0, 20.0], [18.0, 19.0]],
                [[10.0, 11.0], [12.0, 13.0]],
            ]
        ),
        dims=["time", "y", "x"],
        coords={
            "time": pd.to_datetime(
                ["1951-06-01", "1951-06-02", "1951-07-01", "1952-07-01"]
            )
        },
        attrs={"units": "degC"},
        name="tasmax",
    )

    result = compute_threshold_day_counts(tasmax, thresholds=[30.0, 35.0, 40.0])

    row_1951 = result.loc[result["year"] == 1951].iloc[0]
    assert row_1951["days_ge_30_0c"] == 3
    assert row_1951["days_ge_35_0c"] == 2
    assert row_1951["days_ge_40_0c"] == 0

    row_1952 = result.loc[result["year"] == 1952].iloc[0]
    assert row_1952["days_ge_30_0c"] == 0
