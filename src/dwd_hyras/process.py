from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr


DEFAULT_VARIABLE_CANDIDATES = ("tasmax", "tasmax_hyras", "temperature", "tmax")

_KELVIN_UNITS = {"k", "kelvin"}
_CELSIUS_UNITS = {"c", "degc", "degree_celsius", "degrees_celsius", "celsius", "°c"}


def normalize_to_celsius(values: xr.DataArray) -> xr.DataArray:
    """Return daily maximum temperature values in degrees Celsius."""
    units = str(values.attrs.get("units", "")).strip().lower()
    if units in _KELVIN_UNITS:
        result = values - 273.15
        result.attrs.update(values.attrs)
        result.attrs["units"] = "degC"
        return result
    if units in _CELSIUS_UNITS:
        result = values.copy(deep=False)
        result.attrs["units"] = "degC"
        return result
    raise ValueError(
        f"Unrecognized temperature units {values.attrs.get('units')!r}. "
        "Expected Kelvin (e.g. 'K') or Celsius (e.g. 'degC'). "
        "Set the 'units' attribute on the DataArray before calling."
    )


def open_tasmax(
    paths: str | Path | Iterable[str | Path],
    variable: str | None = None,
    chunks: dict[str, int] | None = None,
) -> xr.DataArray:
    """Open one or more HYRAS TASMAX NetCDF files and return Celsius values."""
    if isinstance(paths, (str, Path)):
        path_list = [str(paths)]
    else:
        path_list = [str(path) for path in paths]

    if not path_list:
        raise ValueError("At least one NetCDF path is required.")

    dataset = xr.open_mfdataset(
        path_list,
        combine="by_coords",
        chunks=chunks,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    data_var = variable or _find_tasmax_variable(dataset)
    if data_var not in dataset:
        raise KeyError(f"Variable {data_var!r} not found in dataset.")
    return normalize_to_celsius(dataset[data_var])


def compute_annual_metrics(tasmax_c: xr.DataArray) -> pd.DataFrame:
    """Compute annual Germany-wide TASMAX metrics from a time/y/x DataArray."""
    if "time" not in tasmax_c.dims:
        raise ValueError("TASMAX data must include a 'time' dimension.")

    tasmax_c = normalize_to_celsius(tasmax_c)
    spatial_dims = tuple(dim for dim in tasmax_c.dims if dim != "time")
    if not spatial_dims:
        raise ValueError("TASMAX data must include at least one spatial dimension.")

    valid_cells = tasmax_c.notnull().sum(dim=spatial_dims)
    daily_mean = tasmax_c.mean(dim=spatial_dims, skipna=True)
    daily_max = tasmax_c.max(dim=spatial_dims, skipna=True)
    hot_share_30 = (tasmax_c >= 30.0).sum(dim=spatial_dims) / valid_cells
    hot_share_35 = (tasmax_c >= 35.0).sum(dim=spatial_dims) / valid_cells
    any_hot_30 = hot_share_30 > 0
    any_hot_35 = hot_share_35 > 0

    annual = xr.Dataset(
        {
            "annual_mean_tmax_c": daily_mean.groupby("time.year").mean(skipna=True),
            "annual_max_tmax_c": daily_max.groupby("time.year").max(skipna=True),
            "days_any_hot_30": any_hot_30.groupby("time.year").sum(),
            "days_any_hot_35": any_hot_35.groupby("time.year").sum(),
            "mean_hot_area_share_30": hot_share_30.groupby("time.year").mean(skipna=True),
            "max_hot_area_share_30": hot_share_30.groupby("time.year").max(skipna=True),
            "mean_hot_area_share_35": hot_share_35.groupby("time.year").mean(skipna=True),
            "max_hot_area_share_35": hot_share_35.groupby("time.year").max(skipna=True),
        }
    )

    frame = annual.to_dataframe().reset_index()
    frame["year"] = frame["year"].astype(int)
    frame["days_any_hot_30"] = frame["days_any_hot_30"].astype(int)
    frame["days_any_hot_35"] = frame["days_any_hot_35"].astype(int)
    return frame.sort_values("year").reset_index(drop=True)


def write_annual_metrics_csv(metrics: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)


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
