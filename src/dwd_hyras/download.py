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
    response = requests.get(base_url, timeout=60)
    response.raise_for_status()

    parser = _LinkParser()
    parser.feed(response.text)
    urls = [urljoin(base_url, link) for link in parser.links if link.lower().endswith(".nc")]
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

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0"))
            with target.open("wb") as file_obj, tqdm(
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
        downloaded.append(target)

    return downloaded
