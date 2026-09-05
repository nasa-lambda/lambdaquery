"""Registry / public-API tests."""

import json

import pytest

import lambdaquery
from lambdaquery._fetch import _leaf_names
from lambdaquery._registry import Registry


def _write_manifest(tmp_path, data: dict):
    """Build a Registry over a synthetic manifest given as a plain dict.

    The manifest is JSON on disk, and json.load hands back exactly this shape,
    so fixtures are written as dicts rather than as serialized text.
    """
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return Registry(manifest_path=path)


def test_public_api_surface():
    assert lambdaquery.__all__ == [
        "list_experiments",
        "list_datasets",
        "list_files",
        "fetch_data",
        "get_download_size",
    ]


def test_list_experiments_and_datasets():
    """Smoke test on the packaged manifest: it loads and lists sensibly.

    Asserts properties that survive regeneration rather than specific entry
    names, which come and go as the catalog is rebuilt -- the exception is
    BICEP2/BKP, pinned because tests/test_fetch.py's dedup test depends on it.
    """
    experiments = lambdaquery.list_experiments()
    assert experiments == sorted(experiments)
    assert {"WMAP", "COBE/DIRBE", "BICEP2", "Planck"} <= set(experiments)

    datasets = lambdaquery.list_datasets("WMAP")
    assert datasets == sorted(datasets)
    assert len(datasets) > 1000  # WMAP dominates the catalog
    assert "map_ilc_yr1_v1.fits" in datasets

    assert "BKP" in lambdaquery.list_datasets("BICEP2")


def test_packaged_manifest_entries_all_resolve():
    """Every entry in the shipped manifest builds without error.

    Catches a regenerated manifest with a dangling reference, a cycle, or a
    malformed mapping entry -- all of which would otherwise surface as a
    KeyError/ValueError in a user's first call rather than in CI.
    """
    for experiment in lambdaquery.list_experiments():
        for name in lambdaquery.list_datasets(experiment):
            lambdaquery.list_files(experiment, name)


def test_get_resolves_leaf(tmp_path):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": "/data/map/x.fits"}})
    entry = reg.get("WMAP", "a")
    assert entry.is_file
    assert entry.path == "/data/map/x.fits"
    assert entry.children == ()


def test_get_resolves_group_references(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": "/data/map/a.fits",
                "b": "/data/map/b.fits",
                "grp": ["a", "b"],
            }
        },
    )
    grp = reg.get("WMAP", "grp")
    assert not grp.is_file
    assert [c.name for c in grp.children] == ["a", "b"]
    assert [c.path for c in grp.children] == ["/data/map/a.fits", "/data/map/b.fits"]


def test_get_missing_reference_raises(tmp_path):
    reg = _write_manifest(tmp_path, {"WMAP": {"grp": ["nope"]}})
    with pytest.raises(KeyError):
        reg.get("WMAP", "grp")


def test_get_reference_cycle_raises(tmp_path):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": ["b"], "b": ["a"]}})
    with pytest.raises(ValueError, match="cycle"):
        reg.get("WMAP", "a")


def test_unknown_experiment_raises():
    with pytest.raises(KeyError):
        lambdaquery.list_datasets("NOPE")


# --- mapping-form entries ------------------------------------------------
#
# A leaf may be written "name": "/path" or "name": {"path": ..., "size": ...};
# a group "name": [child, ...] or "name": {"children": [...], ...}. Which key
# is present -- path or children -- decides which kind of entry it is.


def test_mapping_leaf_carries_metadata(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": {
                    "path": "/data/map/a.fits",
                    "size": 50342400,
                    "checksum": "md5:9f2c1b8e04a7d3f61c8b0e2a5d7f4361",
                    "description": "A map",
                }
            }
        },
    )
    entry = reg.get("WMAP", "a")
    assert entry.is_file
    assert entry.path == "/data/map/a.fits"
    assert entry.size == 50342400
    assert entry.checksum == "md5:9f2c1b8e04a7d3f61c8b0e2a5d7f4361"
    assert entry.description == "A map"


def test_mapping_leaf_metadata_is_optional(tmp_path):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": {"path": "/data/map/a.fits"}}})
    entry = reg.get("WMAP", "a")
    assert entry.size is None
    assert entry.checksum is None
    assert entry.description == ""


def test_bare_digest_checksum_survives_round_trip(tmp_path):
    # JSON strings are unambiguously quoted, so an all-digit 32-char digest
    # stays a string -- the type coercion that made the "md5:" prefix
    # load-bearing under YAML cannot happen here.
    digest = "1" * 32
    reg = _write_manifest(
        tmp_path,
        {"WMAP": {"a": {"path": "/data/map/a.fits", "checksum": digest}}},
    )
    assert reg.get("WMAP", "a").checksum == digest


def test_shorthand_and_mapping_leaves_resolve_alike(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": "/data/map/a.fits",
                "b": {"path": "/data/map/b.fits"},
                "grp": ["a", "b"],
            }
        },
    )
    grp = reg.get("WMAP", "grp")
    assert [c.path for c in grp.children] == ["/data/map/a.fits", "/data/map/b.fits"]
    assert all(c.is_file for c in grp.children)


def test_mapping_group_carries_description(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": "/data/map/a.fits",
                "b": "/data/map/b.fits",
                "grp": {"description": "Two maps", "children": ["a", "b"]},
            }
        },
    )
    grp = reg.get("WMAP", "grp")
    assert not grp.is_file
    assert grp.description == "Two maps"
    assert [c.name for c in grp.children] == ["a", "b"]


def test_mapping_group_nests(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": "/data/map/a.fits",
                "inner": {"children": ["a"]},
                "outer": {"children": ["inner"]},
            }
        },
    )
    outer = reg.get("WMAP", "outer")
    assert [c.name for c in outer.children] == ["inner"]
    assert [c.name for c in outer.children[0].children] == ["a"]


def test_mapping_group_cycle_raises(tmp_path):
    # The cycle check must still be threaded through the mapping form.
    reg = _write_manifest(
        tmp_path,
        {"WMAP": {"a": {"children": ["b"]}, "b": {"children": ["a"]}}},
    )
    with pytest.raises(ValueError, match="cycle"):
        reg.get("WMAP", "a")


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"path": "/data/x.fits", "children": ["b"]}, "both"),
        ({"description": "neither"}, "either"),
        ({"path": "/data/x.fits", "checkum": "deadbeef"}, "unknown key"),
        ({"children": [], "size": 5}, "unknown key"),
        ({"path": "/data/x.fits", "size": -1}, "negative"),
    ],
)
def test_mapping_entry_validation_errors(tmp_path, entry, match):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": entry}})
    with pytest.raises(ValueError, match=match):
        reg.get("WMAP", "a")


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"path": "/data/x.fits", "size": "big"}, "size"),
        ({"path": "/data/x.fits", "size": True}, "size"),
        ({"path": "/data/x.fits", "size": 1.5}, "size"),
        ({"path": 42}, "path"),
        ({"path": "/data/x.fits", "checksum": 42}, "checksum"),
        ({"children": "not-a-list"}, "children"),
        (42, "must be a path string"),
    ],
)
def test_mapping_entry_type_errors(tmp_path, entry, match):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": entry}})
    with pytest.raises(TypeError, match=match):
        reg.get("WMAP", "a")


# --- list_files ----------------------------------------------------------
#
# lambdaquery.list_files reads the module-level registry singleton, so these
# drive _leaf_names off a synthetic Registry directly rather than monkeypatching
# it. The public wrapper is one line over the same call.


def _files(reg, experiment, dataset):
    return _leaf_names(reg.get(experiment, dataset))


def test_list_files_leaf_returns_itself(tmp_path):
    reg = _write_manifest(tmp_path, {"WMAP": {"a": "/data/map/a.fits"}})
    assert _files(reg, "WMAP", "a") == ["a"]


def test_list_files_group_returns_members_in_order(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "b": "/data/map/b.fits",
                "a": "/data/map/a.fits",
                "grp": ["b", "a"],
            }
        },
    )
    # Manifest order, not sorted -- these mirror what fetch_data downloads.
    assert _files(reg, "WMAP", "grp") == ["b", "a"]


def test_list_files_flattens_nested_groups(tmp_path):
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "a": "/data/map/a.fits",
                "b": "/data/map/b.fits",
                "c": "/data/map/c.fits",
                "inner": ["a", "b"],
                "outer": ["inner", "c"],
            }
        },
    )
    assert _files(reg, "WMAP", "outer") == ["a", "b", "c"]


def test_list_files_dedups_leaf_shared_by_two_groups(tmp_path):
    # BICEP2's BKP and BKJan2015 share a bandpowers file; a listing must not
    # double-count what fetch_data downloads once.
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "shared": "/data/map/shared.fits",
                "a": "/data/map/a.fits",
                "one": ["a", "shared"],
                "two": ["shared"],
                "both": ["one", "two"],
            }
        },
    )
    assert _files(reg, "WMAP", "both") == ["a", "shared"]


def test_list_files_dedups_across_doubled_slashes(tmp_path):
    # The manifest has 1,265 doubled-slash paths; both spellings are one file.
    reg = _write_manifest(
        tmp_path,
        {
            "WMAP": {
                "plain": "/data/map/x.fits",
                "doubled": "/data/map//x.fits",
                "grp": ["plain", "doubled"],
            }
        },
    )
    assert _files(reg, "WMAP", "grp") == ["plain"]


def test_list_files_public_wrapper_hits_real_manifest():
    names = lambdaquery.list_files("BICEP2", "BK14")
    assert len(names) == 4
    assert "BK14_cosmomc.tgz" in names
