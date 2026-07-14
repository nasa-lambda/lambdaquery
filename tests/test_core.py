"""Registry / public-API tests."""

import textwrap

import pytest

import lambdaquery
from lambdaquery._registry import Registry


def _write_manifest(tmp_path, body: str):
    path = tmp_path / "manifest.yaml"
    path.write_text(textwrap.dedent(body))
    return Registry(manifest_path=path)


def test_public_api_surface():
    assert lambdaquery.__all__ == ["list_experiments", "list_datasets", "fetch_data"]


def test_list_experiments_and_datasets():
    assert "WMAP" in lambdaquery.list_experiments()
    datasets = lambdaquery.list_datasets("WMAP")
    assert "all9yr" in datasets
    assert "iqumapKa9yr" in datasets


def test_get_resolves_leaf(tmp_path):
    reg = _write_manifest(
        tmp_path,
        """
        WMAP:
          a: /data/map/x.fits
        """,
    )
    entry = reg.get("WMAP", "a")
    assert entry.is_file
    assert entry.path == "/data/map/x.fits"
    assert entry.children == ()


def test_get_resolves_group_references(tmp_path):
    reg = _write_manifest(
        tmp_path,
        """
        WMAP:
          a: /data/map/a.fits
          b: /data/map/b.fits
          grp:
            - a
            - b
        """,
    )
    grp = reg.get("WMAP", "grp")
    assert not grp.is_file
    assert [c.name for c in grp.children] == ["a", "b"]
    assert [c.path for c in grp.children] == ["/data/map/a.fits", "/data/map/b.fits"]


def test_get_missing_reference_raises(tmp_path):
    reg = _write_manifest(
        tmp_path,
        """
        WMAP:
          grp:
            - nope
        """,
    )
    with pytest.raises(KeyError):
        reg.get("WMAP", "grp")


def test_get_reference_cycle_raises(tmp_path):
    reg = _write_manifest(
        tmp_path,
        """
        WMAP:
          a:
            - b
          b:
            - a
        """,
    )
    with pytest.raises(ValueError, match="cycle"):
        reg.get("WMAP", "a")


def test_unknown_experiment_raises():
    with pytest.raises(KeyError):
        lambdaquery.list_datasets("NOPE")
