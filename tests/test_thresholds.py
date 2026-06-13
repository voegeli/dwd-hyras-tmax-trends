from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dwd_hyras.thresholds import compute_threshold_day_counts, compute_threshold_day_counts_from_files


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


def test_compute_threshold_day_counts_from_files_raises_on_duplicate_years():
    from unittest.mock import MagicMock, patch

    tasmax = xr.DataArray(
        np.array([[31.0, 32.0]]),
        dims=["time", "x"],
        coords={"time": pd.to_datetime(["1951-06-01"])},
        attrs={"units": "degC"},
    )
    fake_ds = tasmax.to_dataset(name="tasmax")

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=fake_ds)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("dwd_hyras.thresholds.xr.open_dataset", return_value=mock_ctx):
        with pytest.raises(ValueError, match="1951"):
            compute_threshold_day_counts_from_files(["fake1.nc", "fake2.nc"], thresholds=[30.0])
