# AGENTS.md

## Purpose
This repository contains tooling to download, process, and analyze DWD HYRAS daily maximum temperature (TASMAX) datasets.

## Goal
Evaluate long-term temperature trends in Germany using DWD HYRAS data rather than individual weather events.

Key metrics:
- Annual mean daily maximum temperature
- Annual maximum temperature
- Number of hot days (>= 30°C)
- Number of extreme hot days (>= 35°C)
- Fraction of Germany affected by heat events
- Long-term trends since 1951

## Technology Stack
- Python 3.11+
- xarray
- netCDF4
- numpy
- pandas
- matplotlib
- requests
- tqdm

## Workflow

### 1. Download Data
Download HYRAS TASMAX NetCDF files from:

https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_max/

### 2. Load Dataset
Use xarray to load NetCDF files.

### 3. Normalize Units
Convert Kelvin to Celsius if required.

### 4. Compute Metrics
For every day:
- Germany-wide mean Tmax
- Germany-wide maximum Tmax
- Area share >= 30°C
- Area share >= 35°C

Aggregate to annual statistics.

### 5. Generate Outputs
Create:
- CSV summary
- Trend plots
- Heat-day statistics
- Decadal comparisons

## Coding Guidelines
- Prefer vectorized xarray operations.
- Avoid loading unnecessary variables.
- Keep memory usage low.
- Use type hints where practical.
- Separate download, processing, and visualization logic.

## Validation
Cross-check:
- Annual means against DWD publications.
- Heat-day trends against official DWD climate reports.

## Deliverables
- dwd_hyras_tasmax_analysis.csv
- annual_mean_tmax.png
- hot_area_days_30.png
- Optional trend report in Markdown

## Future Extensions
- Regional analysis by Bundesland
- Heatwave duration statistics
- Comparison with station measurements
- Linear trend analysis
- Mann-Kendall significance tests
