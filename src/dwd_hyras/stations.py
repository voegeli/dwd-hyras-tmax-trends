from __future__ import annotations

from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

from dwd_hyras.thresholds import DEFAULT_AIRPORT_FEATURES, DEFAULT_CITY_FEATURES


def compute_station_threshold_day_counts_from_files(
    paths: Iterable[str | Path],
    thresholds: Iterable[float],
    station_locations: dict[str, tuple[float, float]] | None = None,
    city_buffer_km: float = 20.0,
    airport_buffer_km: float = 10.0,
) -> pd.DataFrame:
    """Count days per year where any DWD station reaches each TXK threshold."""
    threshold_list = list(thresholds)
    path_list = list(paths)
    frames = [
        _read_station_txk(path)
        for path in tqdm(path_list, desc="DWD station ZIP files", unit="file")
    ]
    if not frames:
        raise ValueError("At least one station ZIP path is required.")

    station_values = pd.concat(frames, ignore_index=True)
    station_values = station_values.dropna(subset=["txk_c"])
    if station_values.empty:
        raise ValueError("No valid TXK station values found.")

    station_values["year"] = station_values["date"].dt.year.astype(int)
    years = np.array(sorted(station_values["year"].unique()), dtype=int)

    data: dict[str, np.ndarray] = {"year": years}
    for scenario, scenario_values in _station_scenarios(
        station_values,
        station_locations,
        city_buffer_km,
        airport_buffer_km,
    ).items():
        daily_max = scenario_values.groupby("date", as_index=False)["txk_c"].max()
        daily_max["year"] = daily_max["date"].dt.year.astype(int)
        for threshold in threshold_list:
            reached = daily_max["txk_c"] >= threshold
            data[_threshold_column(threshold, scenario)] = np.array(
                [int(reached[daily_max["year"] == year].sum()) for year in years],
                dtype=int,
            )

    return pd.DataFrame(data).sort_values("year").reset_index(drop=True)


def _station_scenarios(
    station_values: pd.DataFrame,
    station_locations: dict[str, tuple[float, float]] | None,
    city_buffer_km: float,
    airport_buffer_km: float,
) -> dict[str, pd.DataFrame]:
    scenarios = {"all": station_values}
    if not station_locations:
        return scenarios

    station_ids = station_values["station_id"].astype(str).str.zfill(5)
    city_station_ids = {
        station_id
        for station_id, (lat, lon) in station_locations.items()
        if _is_near_any_feature(lat, lon, DEFAULT_CITY_FEATURES, city_buffer_km)
    }
    airport_station_ids = {
        station_id
        for station_id, (lat, lon) in station_locations.items()
        if _is_near_any_feature(lat, lon, DEFAULT_AIRPORT_FEATURES, airport_buffer_km)
    }

    near_city = station_ids.isin(city_station_ids)
    near_airport = station_ids.isin(airport_station_ids)
    scenarios["no_airports"] = station_values.loc[~near_airport]
    scenarios["no_cities"] = station_values.loc[~near_city]
    scenarios["no_airports_no_cities"] = station_values.loc[~(near_airport | near_city)]
    return scenarios


def _is_near_any_feature(lat: float, lon: float, features: Iterable[object], radius_km: float) -> bool:
    return any(_haversine_km(lat, lon, feature.lat, feature.lon) <= radius_km for feature in features)


def _haversine_km(lat: float, lon: float, center_lat: float, center_lon: float) -> float:
    earth_radius_km = 6371.0088
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    lat2 = np.deg2rad(center_lat)
    lon2 = np.deg2rad(center_lon)
    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return float(2.0 * earth_radius_km * np.arcsin(np.sqrt(a)))


def _read_station_txk(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    with ZipFile(path) as archive:
        data_names = [
            name
            for name in archive.namelist()
            if Path(name).name.lower().startswith("produkt_") and name.lower().endswith(".txt")
        ]
        if not data_names:
            raise ValueError(f"No DWD product text file found in {path}.")

        with archive.open(data_names[0]) as handle:
            frame = pd.read_csv(
                handle,
                sep=";",
                encoding="latin1",
                na_values=[-999, "-999"],
                usecols=lambda column: column.strip() in {"STATIONS_ID", "MESS_DATUM", "TXK"},
            )

    frame.columns = [column.strip().lower() for column in frame.columns]
    if "mess_datum" not in frame or "txk" not in frame:
        raise ValueError(f"DWD station file in {path} does not contain MESS_DATUM and TXK.")
    station_id = (
        frame["stations_id"].astype(str).str.strip().str.zfill(5)
        if "stations_id" in frame
        else pd.Series([_station_id_from_path(path)] * len(frame))
    )

    return pd.DataFrame(
        {
            "station_id": station_id,
            "date": pd.to_datetime(frame["mess_datum"].astype(str), format="%Y%m%d"),
            "txk_c": pd.to_numeric(frame["txk"], errors="coerce"),
        }
    )


def _station_id_from_path(path: Path) -> str:
    for part in path.stem.split("_"):
        if part.isdigit():
            return part.zfill(5)
    raise ValueError(f"Could not infer station ID from {path}.")


def _threshold_column(threshold: float, scenario: str = "all") -> str:
    value = f"{threshold:.1f}".replace(".", "_")
    suffix = "" if scenario == "all" else f"__{scenario}"
    return f"days_ge_{value}c{suffix}"
