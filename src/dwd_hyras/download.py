from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


DWD_HYRAS_TASMAX_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/grids_germany/"
    "daily/hyras_de/air_temperature_max/"
)
DWD_DAILY_KL_HISTORICAL_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/daily/kl/historical/"
)
DWD_DAILY_KL_100_YEAR_TXK_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/daily/kl/timeseries_overview/ZeitReihen_klima_tag_GE_100Jahre_TXK.txt"
)
DWD_DAILY_KL_STATION_DESCRIPTION_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/daily/kl/historical/KL_Tageswerte_Beschreibung_Stationen.txt"
)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


def list_netcdf_urls(base_url: str = DWD_HYRAS_TASMAX_URL) -> list[str]:
    return _list_urls(base_url, ".nc")


def list_zip_urls(base_url: str) -> list[str]:
    return _list_urls(base_url, ".zip")


def list_station_ids_from_overview(url: str = DWD_DAILY_KL_100_YEAR_TXK_URL) -> list[str]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return _parse_station_ids(response.content.decode("latin1").splitlines())


def list_station_locations_from_overview(
    url: str = DWD_DAILY_KL_100_YEAR_TXK_URL,
    station_description_url: str = DWD_DAILY_KL_STATION_DESCRIPTION_URL,
) -> dict[str, tuple[float, float]]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    requested_ids = set(_parse_station_ids(response.content.decode("latin1").splitlines()))

    response = requests.get(station_description_url, timeout=60)
    response.raise_for_status()
    station_locations = _parse_station_locations(response.content.decode("latin1").splitlines())
    return {
        station_id: location
        for station_id, location in station_locations.items()
        if station_id in requested_ids
    }


def _parse_station_ids(lines: list[str]) -> list[str]:
    station_ids: list[str] = []
    for line in lines[1:]:
        parts = [part.strip() for part in line.split(";")]
        if parts and parts[0].isdigit():
            station_ids.append(parts[0].zfill(5))
    return station_ids


def _parse_station_locations(lines: list[str]) -> dict[str, tuple[float, float]]:
    station_locations: dict[str, tuple[float, float]] = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        station_locations[parts[0].zfill(5)] = (float(parts[4]), float(parts[5]))
    return station_locations


def _list_urls(base_url: str, suffix: str) -> list[str]:
    response = requests.get(base_url, timeout=60)
    response.raise_for_status()

    parser = _LinkParser()
    parser.feed(response.text)
    urls = [urljoin(base_url, link) for link in parser.links if link.lower().endswith(suffix)]
    return sorted(set(urls))


def download_files(urls: list[str], output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for url in urls:
        target = output / url.rstrip("/").split("/")[-1]
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue

        tmp = target.with_name(target.name + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or "0")
                with tmp.open("wb") as file_obj, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=target.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        file_obj.write(chunk)
                        progress.update(len(chunk))
            tmp.rename(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        downloaded.append(target)

    return downloaded
