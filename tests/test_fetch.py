"""Tests for the download/dispatch layer (offline; network is mocked)."""

from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from lambdaquery import _fetch
from lambdaquery._registry import DatasetEntry, Registry

# --- pure path helpers ---------------------------------------------------


def test_local_path_mirrors_data_tree(tmp_path):
    dest = _fetch._local_path("/data/map/dr5/9yr/x.fits", tmp_path)
    assert dest == tmp_path / "data/map/dr5/9yr/x.fits"


def test_lambda_url():
    assert _fetch._lambda_url("/data/map/x.fits") == (
        "https://lambda.gsfc.nasa.gov/data/map/x.fits"
    )


def test_lambda_path_to_s3_wmap_rewrite():
    bucket, key = _fetch._lambda_path_to_s3("WMAP", "/data/map/dr5/9yr/x.fits")
    assert bucket == "nasa-lambda"
    assert key == "wmap/dr5/9yr/x.fits"


def test_lambda_path_to_s3_no_rewrite():
    bucket, key = _fetch._lambda_path_to_s3("COBE/FIRAS", "/data/cobe/firas/x.fits")
    assert (bucket, key) == ("nasa-lambda", "cobe/firas/x.fits")


def test_lambda_path_to_s3_collapses_repeated_slashes():
    # The manifest carries paths like "/data/map/powspec//file.txt"; HTTPS
    # tolerates the doubled slash but S3 keys are literal.
    _, key = _fetch._lambda_path_to_s3("WMAP", "/data/map/powspec//x.txt")
    assert key == "wmap/powspec/x.txt"


def test_local_path_collapses_repeated_slashes(tmp_path):
    # Dedup depends on both spellings landing on one file.
    assert _fetch._local_path(
        "/data/map/powspec//x.txt", tmp_path
    ) == _fetch._local_path("/data/map/powspec/x.txt", tmp_path)


# --- dispatch + dedup ----------------------------------------------------


def _record_downloaders(monkeypatch):
    """Replace the real downloaders with fakes that create the file and log calls."""
    calls = {"s3": [], "url": []}

    def fake_s3(bucket, key, location):
        calls["s3"].append(key)
        location.write_bytes(b"s3-bytes")
        return location

    def fake_url(url, location):
        calls["url"].append(url)
        location.write_bytes(b"url-bytes")
        return location

    monkeypatch.setattr(_fetch, "_download_from_s3", fake_s3)
    monkeypatch.setattr(_fetch, "_download_from_url", fake_url)
    return calls


def test_wmap_routes_to_s3(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)
    entry = DatasetEntry(experiment="WMAP", name="f", path="/data/map/dr5/9yr/x.fits")
    result = _fetch._download(entry, tmp_path)
    assert result == tmp_path / "data/map/dr5/9yr/x.fits"
    assert calls["s3"] == ["wmap/dr5/9yr/x.fits"]
    assert calls["url"] == []


def test_planck_routes_to_url(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)
    entry = DatasetEntry(experiment="PLANCK", name="f", path="/data/planck/x.fits")
    _fetch._download(entry, tmp_path)
    assert calls["url"] == ["https://lambda.gsfc.nasa.gov/data/planck/x.fits"]
    assert calls["s3"] == []


def test_group_download_dedups_shared_leaf(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)
    reg = Registry()  # packaged sample manifest
    entry = reg.get("WMAP", "all9yr")  # references iqumapKa9yr twice

    result = _fetch._download(entry, tmp_path)

    assert isinstance(result, list)
    # The shared leaf's S3 key appears once despite being referenced twice.
    shared_key = (
        "wmap/dr5/skymaps/9yr/forered/wmap_band_forered_iqumap_r9_9yr_Ka_v5.fits"
    )
    assert calls["s3"].count(shared_key) == 1
    assert len(set(calls["s3"])) == 2  # two distinct leaves, each downloaded once
    assert all(p.exists() for p in result)


def _s3_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "nope"}}, "HeadObject")


def test_s3_miss_falls_back_to_https(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)

    def missing(bucket, key, location):
        calls["s3"].append(key)
        raise _s3_error("404")

    monkeypatch.setattr(_fetch, "_download_from_s3", missing)

    # WMAP's 1-year release is not on the mirror, but HTTPS serves it.
    entry = DatasetEntry(experiment="WMAP", name="f", path="/data/map/skymaps/x.fits")
    result = _fetch._download(entry, tmp_path)

    assert calls["s3"] == ["wmap/skymaps/x.fits"]
    assert calls["url"] == ["https://lambda.gsfc.nasa.gov/data/map/skymaps/x.fits"]
    assert isinstance(result, Path)  # leaf entry -> single path
    assert result.read_bytes() == b"url-bytes"


def test_s3_non_404_error_propagates(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)

    def throttled(bucket, key, location):
        raise _s3_error("SlowDown")

    monkeypatch.setattr(_fetch, "_download_from_s3", throttled)

    entry = DatasetEntry(experiment="WMAP", name="f", path="/data/map/dr5/x.fits")
    with pytest.raises(ClientError):
        _fetch._download(entry, tmp_path)
    assert calls["url"] == []  # no silent fallback on a real failure


def test_cache_short_circuits_second_call(tmp_path, monkeypatch):
    calls = _record_downloaders(monkeypatch)
    entry = DatasetEntry(experiment="WMAP", name="f", path="/data/map/x.fits")

    _fetch._download(entry, tmp_path)
    _fetch._download(entry, tmp_path)  # already cached

    assert len(calls["s3"]) == 1


# --- atomic download -----------------------------------------------------


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_from_url_atomic(tmp_path, monkeypatch):
    seen_paths = []

    def fake_get(url, stream):
        assert stream is True
        return _FakeResponse([b"abc", b"def"])

    monkeypatch.setattr(_fetch.requests, "get", fake_get)

    real_replace = _fetch.os.replace

    def spy_replace(src, dst):
        seen_paths.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(_fetch.os, "replace", spy_replace)

    dest = tmp_path / "out.fits"
    _fetch._download_from_url("http://example/x", dest)

    assert dest.read_bytes() == b"abcdef"
    # written via a .part temp then atomically renamed to the final path
    src, dst = seen_paths[0]
    assert src.endswith(".part")
    assert dst == str(dest)
    assert not (tmp_path / "out.fits.part").exists()
