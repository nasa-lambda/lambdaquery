"""Tests for the cache/verification layer."""

import hashlib

import pytest

from lambdaquery import _cache
from lambdaquery._cache import ChecksumError
from lambdaquery._registry import DatasetEntry


def _leaf(**kwargs) -> DatasetEntry:
    return DatasetEntry(experiment="WMAP", name="f", path="/data/x.fits", **kwargs)


def test_compute_checksum_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    assert _cache.compute_checksum(f, "md5") == hashlib.md5(b"hello world").hexdigest()
    assert _cache.compute_checksum(f, "sha256") == (
        hashlib.sha256(b"hello world").hexdigest()
    )


def test_is_cached_missing_file(tmp_path):
    assert not _cache.is_cached(tmp_path / "nope.fits", _leaf())


def test_is_cached_empty_file(tmp_path):
    f = tmp_path / "empty.fits"
    f.write_bytes(b"")
    assert not _cache.is_cached(f, _leaf())


def test_is_cached_present_no_metadata(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    assert _cache.is_cached(f, _leaf())


def test_is_cached_checks_size(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    assert _cache.is_cached(f, _leaf(size=4))
    assert not _cache.is_cached(f, _leaf(size=99))


def test_is_cached_ignores_checksum(tmp_path):
    # is_cached runs on every fetch and once per leaf in a size walk, so it
    # checks size only -- hashing here would re-read every cached byte just to
    # answer "is it already there?". verify still catches a bad digest.
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    assert _cache.is_cached(f, _leaf(checksum="deadbeef"))


def test_is_cached_does_not_read_the_file(tmp_path, monkeypatch):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")

    def boom(*args, **kwargs):
        raise AssertionError("is_cached must not hash the file")

    monkeypatch.setattr(_cache, "compute_checksum", boom)
    good = hashlib.md5(b"data").hexdigest()
    assert _cache.is_cached(f, _leaf(size=4, checksum=good))


def test_verify_checks_checksum(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    good = hashlib.md5(b"data").hexdigest()
    _cache.verify(f, _leaf(checksum=good))  # bare hex means md5
    _cache.verify(f, _leaf(checksum=f"md5:{good}"))
    assert f.exists()


def test_verify_noop_without_metadata(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    _cache.verify(f, _leaf())  # no raise
    assert f.exists()


def test_verify_size_mismatch_deletes_and_raises(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    with pytest.raises(ChecksumError, match="Size mismatch"):
        _cache.verify(f, _leaf(size=99))
    assert not f.exists()


def test_verify_checksum_mismatch_deletes_and_raises(tmp_path):
    f = tmp_path / "x.fits"
    f.write_bytes(b"data")
    with pytest.raises(ChecksumError, match="Checksum mismatch"):
        _cache.verify(f, _leaf(checksum="deadbeef"))
    assert not f.exists()
