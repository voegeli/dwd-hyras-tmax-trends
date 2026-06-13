from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dwd_hyras.download import download_files


def _mock_response(chunks, content_length: str = "100") -> MagicMock:
    r = MagicMock()
    r.__enter__ = lambda s: s
    r.__exit__ = MagicMock(return_value=False)
    r.raise_for_status = MagicMock()
    r.headers = {"content-length": content_length}
    r.iter_content = MagicMock(return_value=iter(chunks))
    return r


def test_interrupted_download_leaves_no_partial_file(tmp_path):
    def failing_chunks():
        yield b"partial data"
        raise ConnectionError("network dropped")

    mock = _mock_response(failing_chunks())

    with patch("dwd_hyras.download.requests.get", return_value=mock):
        with pytest.raises(ConnectionError):
            download_files(["http://example.com/test.nc"], tmp_path)

    assert not any(tmp_path.iterdir()), "Partial file must not be left on disk after a failed download"


def test_empty_content_length_header_does_not_crash(tmp_path):
    mock = _mock_response([b"data"], content_length="")

    with patch("dwd_hyras.download.requests.get", return_value=mock):
        paths = download_files(["http://example.com/test.nc"], tmp_path)

    assert len(paths) == 1
    assert paths[0].exists()
