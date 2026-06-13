from __future__ import annotations

import pytest

from dwd_hyras.cli import _expand_inputs, _threshold_range


def test_threshold_range_no_duplicate_values_with_fine_step():
    thresholds = _threshold_range(25.0, 27.0, 0.05)
    assert len(thresholds) == len(set(thresholds)), f"Duplicate thresholds: {thresholds}"


def test_threshold_range_includes_endpoints():
    thresholds = _threshold_range(25.0, 30.0, 0.5)
    assert thresholds[0] == 25.0
    assert thresholds[-1] == 30.0


def test_expand_inputs_handles_absolute_path_with_wildcard(tmp_path):
    (tmp_path / "data.nc").write_bytes(b"x")
    pattern = str(tmp_path / "*.nc")
    result = _expand_inputs([pattern])
    assert len(result) == 1
    assert result[0].name == "data.nc"
