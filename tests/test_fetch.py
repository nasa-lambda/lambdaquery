"""Tests for the download/dispatch layer (offline; network is mocked)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
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


# --- size walk -----------------------------------------------------------


def _record_sizes(monkeypatch, sizes):
    """Replace the size probes with fakes driven by a {key-or-url: size} map.

    A missing entry (or an explicit None) means "size unavailable" -- the same
    signal the real probes return for a 404 or a transport failure.
    """
    calls = {"s3": [], "url": []}

    def fake_s3(bucket, key):
        calls["s3"].append(key)
        return sizes.get(key)

    def fake_url(url):
        calls["url"].append(url)
        return sizes.get(url)

    monkeypatch.setattr(_fetch, "_size_from_s3", fake_s3)
    monkeypatch.setattr(_fetch, "_size_from_url", fake_url)
    return calls


def _group(*names):
    return DatasetEntry(
        experiment="WMAP",
        name="grp",
        children=tuple(
            DatasetEntry(experiment="WMAP", name=n, path=f"/data/map/dr5/{n}")
            for n in names
        ),
    )


def test_total_size_sums_group_leaves(tmp_path, monkeypatch):
    _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100, "wmap/dr5/b.fits": 23})
    assert _fetch._total_size(_group("a.fits", "b.fits"), tmp_path) == 123


def test_total_size_of_leaf(tmp_path, monkeypatch):
    _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100})
    entry = DatasetEntry(experiment="WMAP", name="a", path="/data/map/dr5/a.fits")
    assert _fetch._total_size(entry, tmp_path) == 100


def test_total_size_skips_cached_files(tmp_path, monkeypatch):
    calls = _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100, "wmap/dr5/b.fits": 23})
    cached = tmp_path / "data/map/dr5/a.fits"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"already here")

    # Only the uncached leaf contributes -- and only it is probed.
    assert _fetch._total_size(_group("a.fits", "b.fits"), tmp_path) == 23
    assert calls["s3"] == ["wmap/dr5/b.fits"]


def test_total_size_dedups_shared_leaf(tmp_path, monkeypatch):
    calls = _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100})
    leaf = DatasetEntry(experiment="WMAP", name="a", path="/data/map/dr5/a.fits")
    group = DatasetEntry(experiment="WMAP", name="grp", children=(leaf, leaf))

    # Nothing is written during a size walk, so dedup can't ride on the cache
    # the way _download's does.
    assert _fetch._total_size(group, tmp_path) == 100
    assert calls["s3"] == ["wmap/dr5/a.fits"]


def test_total_size_unknown_leaf_is_skipped(tmp_path, monkeypatch):
    # b.fits is on neither source; a dead link must not fail the whole query.
    _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100})
    assert _fetch._total_size(_group("a.fits", "b.fits"), tmp_path) == 100


def test_total_size_reports_each_leaf_once(tmp_path, monkeypatch):
    _record_sizes(monkeypatch, {"wmap/dr5/a.fits": 100, "wmap/dr5/c.fits": 5})
    cached = tmp_path / "data/map/dr5/c.fits"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"here")

    events = []
    _fetch._total_size(
        _group("a.fits", "b.fits", "c.fits"),
        tmp_path,
        on_file=lambda e, p: events.append((e, p.name)),
    )
    assert events == [
        ("counted", "a.fits"),
        ("unknown", "b.fits"),
        ("cached", "c.fits"),
    ]


# --- leaf names ----------------------------------------------------------


def test_leaf_names_walks_a_group():
    assert _fetch._leaf_names(_group("a.fits", "b.fits")) == ["a.fits", "b.fits"]


def test_leaf_names_of_a_leaf_is_its_own_name():
    entry = DatasetEntry(experiment="WMAP", name="a", path="/data/map/dr5/a.fits")
    assert _fetch._leaf_names(entry) == ["a"]


def test_leaf_names_dedups_shared_leaf():
    # Same dedup key as _total_size, so `files` and `size` agree on the count.
    leaf = DatasetEntry(experiment="WMAP", name="a", path="/data/map/dr5/a.fits")
    group = DatasetEntry(experiment="WMAP", name="grp", children=(leaf, leaf))
    assert _fetch._leaf_names(group) == ["a"]


def test_leaf_names_dedups_across_doubled_slashes():
    plain = DatasetEntry(experiment="WMAP", name="p", path="/data/map/powspec/x.txt")
    doubled = DatasetEntry(experiment="WMAP", name="d", path="/data/map/powspec//x.txt")
    group = DatasetEntry(experiment="WMAP", name="grp", children=(plain, doubled))
    assert _fetch._leaf_names(group) == ["p"]


def test_remote_size_falls_back_to_https_on_s3_miss(monkeypatch):
    calls = _record_sizes(
        monkeypatch,
        {"https://lambda.gsfc.nasa.gov/data/map/skymaps/x.fits": 42},
    )
    # WMAP's 1-year release is not on the mirror, but HTTPS serves it.
    entry = DatasetEntry(experiment="WMAP", name="f", path="/data/map/skymaps/x.fits")

    assert _fetch._remote_size(entry) == 42
    assert calls["s3"] == ["wmap/skymaps/x.fits"]
    assert calls["url"] == ["https://lambda.gsfc.nasa.gov/data/map/skymaps/x.fits"]


def test_remote_size_skips_s3_for_non_mirrored_experiment(monkeypatch):
    calls = _record_sizes(
        monkeypatch, {"https://lambda.gsfc.nasa.gov/data/planck/x.fits": 7}
    )
    entry = DatasetEntry(experiment="PLANCK", name="f", path="/data/planck/x.fits")

    assert _fetch._remote_size(entry) == 7
    assert calls["s3"] == []


# --- size probes ---------------------------------------------------------


def _fake_s3_client(monkeypatch, head_object):
    """Point _s3_client at a stub whose head_object is the given function."""
    monkeypatch.setattr(
        _fetch, "_s3_client", lambda: SimpleNamespace(head_object=head_object)
    )


def test_size_from_s3_reads_content_length(monkeypatch):
    _fake_s3_client(monkeypatch, lambda Bucket, Key: {"ContentLength": 1008000})
    assert _fetch._size_from_s3("nasa-lambda", "wmap/x.fits") == 1008000


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "SlowDown"])
def test_size_from_s3_returns_none_on_client_error(monkeypatch, code):
    def raising(Bucket, Key):
        raise _s3_error(code)

    _fake_s3_client(monkeypatch, raising)
    # Broader than the download path's _S3_MISSING_CODES check on purpose: here
    # the fallback costs one more HEAD, not a re-transferred file.
    assert _fetch._size_from_s3("nasa-lambda", "wmap/x.fits") is None


class _FakeHead:
    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers if headers is not None else {}


@pytest.mark.parametrize(
    "response,expected",
    [
        (_FakeHead(200, {"Content-Length": "1008000"}), 1008000),
        (_FakeHead(404), None),
        (_FakeHead(200, {}), None),  # no Content-Length header
        (_FakeHead(200, {"Content-Length": "not-a-number"}), None),
    ],
)
def test_size_from_url(monkeypatch, response, expected):
    monkeypatch.setattr(
        _fetch.requests, "head", lambda url, allow_redirects, timeout: response
    )
    assert _fetch._size_from_url("https://example/x.fits") == expected


def test_size_from_url_returns_none_on_transport_error(monkeypatch):
    def boom(url, allow_redirects, timeout):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(_fetch.requests, "head", boom)
    assert _fetch._size_from_url("https://example/x.fits") is None
