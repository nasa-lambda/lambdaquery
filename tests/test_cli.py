"""Tests for the command-line interface (offline; network is mocked).

``main`` takes an ``argv`` parameter precisely so these can call it directly --
no ``subprocess``. ``_cli`` binds the public functions into its own namespace via
``from . import ...``, so ``monkeypatch.setattr(_cli, "fetch_data", ...)`` is the
seam for stubbing them.
"""

from pathlib import Path

import pytest
import requests
from botocore.exceptions import ClientError

import lambdaquery
from lambdaquery import _cli, _fetch
from lambdaquery._cache import ChecksumError
from lambdaquery._registry import DatasetEntry


def _stub_fetch(monkeypatch, result, *, on_file_events=()):
    """Replace ``_cli.fetch_data`` with a fake: records call args, replays events."""
    calls = []

    def fake(experiment, dataset, location, *, on_file=None):
        calls.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "location": location,
                "on_file": on_file,
            }
        )
        for event, path in on_file_events:
            if on_file is not None:
                on_file(event, path)
        return result

    monkeypatch.setattr(_cli, "fetch_data", fake)
    return calls


def _stub_size(monkeypatch, total, *, on_file_events=()):
    """Replace ``_cli.get_download_size`` with a fake: records args, replays events."""
    calls = []

    def fake(experiment, dataset, location, *, on_file=None):
        calls.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "location": location,
                "on_file": on_file,
            }
        )
        for event, path in on_file_events:
            if on_file is not None:
                on_file(event, path)
        return total

    monkeypatch.setattr(_cli, "get_download_size", fake)
    return calls


def _raising_fetch(monkeypatch, exc):
    def fake(experiment, dataset, location, *, on_file=None):
        raise exc

    monkeypatch.setattr(_cli, "fetch_data", fake)


# --- listing commands ----------------------------------------------------


def test_experiments_prints_one_per_line(capsys, monkeypatch):
    monkeypatch.setattr(_cli, "list_experiments", lambda: ["A", "B"])
    assert _cli.main(["experiments"]) == 0
    assert capsys.readouterr().out == "A\nB\n"


def _stub_listing(monkeypatch, members):
    """Stub the pair ``datasets`` uses: names come from ``members``' keys."""
    seen = []

    def fake_datasets(experiment):
        seen.append(experiment)
        return list(members)

    monkeypatch.setattr(_cli, "list_datasets", fake_datasets)
    monkeypatch.setattr(_cli, "list_files", lambda experiment, name: members[name])
    return seen


def test_datasets_plain_prints_one_per_line(capsys, monkeypatch):
    seen = _stub_listing(monkeypatch, {"x.fits": ["x.fits"], "y.fits": ["y.fits"]})
    assert _cli.main(["datasets", "WMAP", "--plain"]) == 0
    assert capsys.readouterr().out == "x.fits\ny.fits\n"
    assert seen == ["WMAP"]


def test_datasets_plain_skips_resolving_entries(capsys, monkeypatch):
    # --plain is the scripting path: it must not resolve every entry to count
    # files, which for WMAP means 6,860 lookups the caller did not ask for.
    monkeypatch.setattr(_cli, "list_datasets", lambda experiment: ["x.fits"])

    def boom(experiment, name):
        raise AssertionError("list_files must not be called for --plain")

    monkeypatch.setattr(_cli, "list_files", boom)
    assert _cli.main(["datasets", "WMAP", "--plain"]) == 0
    assert capsys.readouterr().out == "x.fits\n"


def test_datasets_annotates_multi_file_entries(capsys, monkeypatch):
    _stub_listing(
        monkeypatch,
        {
            "BK14": ["a.txt", "b.txt", "c.txt"],
            "z.fits": ["z.fits"],
        },
    )
    assert _cli.main(["datasets", "BICEP2"]) == 0
    out = capsys.readouterr().out
    assert out == "BK14  (3 files)\nz.fits\n"
    # Leaf rows stay bare -- no padding, so no trailing whitespace.
    assert not any(line.endswith(" ") for line in out.splitlines())


def test_datasets_aligns_on_widest_group_name(capsys, monkeypatch):
    # The long name here is a leaf: padding to it would strand the annotations
    # far to the right for no gain.
    _stub_listing(
        monkeypatch,
        {
            "B2": ["a.txt", "b.txt"],
            "BKJan2015": ["c.txt", "d.txt"],
            "a_very_long_leaf_file_name.fits": ["a_very_long_leaf_file_name.fits"],
        },
    )
    assert _cli.main(["datasets", "BICEP2"]) == 0
    assert capsys.readouterr().out == (
        "B2         (2 files)\nBKJan2015  (2 files)\na_very_long_leaf_file_name.fits\n"
    )


def test_datasets_single_child_group_reads_as_a_leaf(capsys, monkeypatch):
    # "Multi-file" is the question being asked; "(1 files)" would be noise.
    _stub_listing(monkeypatch, {"solo": ["only.txt"]})
    assert _cli.main(["datasets", "BICEP2"]) == 0
    assert capsys.readouterr().out == "solo\n"


def test_empty_listing_prints_nothing(capsys, monkeypatch):
    # print(*[], sep="\n") is bare print() -- it would emit a stray newline.
    monkeypatch.setattr(_cli, "list_experiments", lambda: [])
    assert _cli.main(["experiments"]) == 0
    assert capsys.readouterr().out == ""


def test_files_prints_one_member_per_line(capsys, monkeypatch):
    seen = []

    def fake(experiment, dataset):
        seen.append((experiment, dataset))
        return ["a.txt", "b.txt", "c.txt"]

    monkeypatch.setattr(_cli, "list_files", fake)
    assert _cli.main(["files", "BICEP2", "BK14"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "a.txt\nb.txt\nc.txt\n"
    assert captured.err == ""  # a listing command: the answer, nothing else
    assert seen == [("BICEP2", "BK14")]


def test_files_on_a_leaf_prints_the_single_name(capsys, monkeypatch):
    monkeypatch.setattr(_cli, "list_files", lambda experiment, dataset: [dataset])
    assert _cli.main(["files", "WMAP", "x.fits"]) == 0
    assert capsys.readouterr().out == "x.fits\n"


def test_files_unknown_dataset_exits_1(capsys):
    # No stub: the real registry raises KeyError with the available entries.
    assert _cli.main(["files", "BICEP2", "NOPE"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "not found" in captured.err


def test_unknown_experiment_exits_1(capsys):
    # No stub: the real registry raises KeyError with the available names.
    assert _cli.main(["datasets", "NOPE"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Available experiments" in captured.err
    # err.args[0], not str(KeyError) -- the latter wraps the message in quotes.
    assert not captured.err.startswith('error: "')


# --- fetch: output shape -------------------------------------------------


def test_fetch_single_path_printed(capsys, monkeypatch, tmp_path):
    # A file leaf yields a bare Path, not a list; it must not print as a repr.
    dest = tmp_path / "data" / "map" / "x.fits"
    _stub_fetch(monkeypatch, dest)

    assert _cli.main(["fetch", "WMAP", "x.fits", "-o", str(tmp_path), "-q"]) == 0
    assert capsys.readouterr().out == f"{dest}\n"


def test_fetch_group_prints_each_path(capsys, monkeypatch, tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    _stub_fetch(monkeypatch, [a, b])

    assert _cli.main(["fetch", "WMAP", "grp", "-o", str(tmp_path), "-q"]) == 0
    assert capsys.readouterr().out == f"{a}\n{b}\n"


def test_fetch_paths_go_to_stdout_progress_to_stderr(capsys, monkeypatch, tmp_path):
    # Keeps `lambdaquery fetch ... > files.txt` free of progress chatter.
    dest = tmp_path / "x.fits"
    dest.write_bytes(b"abc")
    _stub_fetch(monkeypatch, dest, on_file_events=[("start", dest), ("done", dest)])

    assert _cli.main(["fetch", "WMAP", "x.fits", "-o", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{dest}\n"
    assert "downloading" in captured.err
    assert "1 downloaded" in captured.err


# --- fetch: arguments ----------------------------------------------------


def test_output_defaults_to_cwd(monkeypatch, tmp_path):
    calls = _stub_fetch(monkeypatch, tmp_path / "x.fits")
    assert _cli.main(["fetch", "WMAP", "x.fits", "-q"]) == 0
    assert calls[0]["location"] == Path(".")


def test_output_flag_is_a_path(monkeypatch, tmp_path):
    calls = _stub_fetch(monkeypatch, tmp_path / "x.fits")
    assert _cli.main(["fetch", "WMAP", "x.fits", "-o", str(tmp_path), "-q"]) == 0
    assert calls[0]["location"] == tmp_path
    assert isinstance(calls[0]["location"], Path)  # argparse type=Path


def test_fetch_forwards_experiment_and_dataset(monkeypatch, tmp_path):
    calls = _stub_fetch(monkeypatch, tmp_path / "x.fits")
    _cli.main(["fetch", "COBE/FIRAS", "spec.fits", "-o", str(tmp_path), "-q"])
    assert calls[0]["experiment"] == "COBE/FIRAS"
    assert calls[0]["dataset"] == "spec.fits"


def test_no_command_exits_2():
    # subparsers(required=True): argparse exits before any handler runs.
    with pytest.raises(SystemExit) as excinfo:
        _cli.main([])
    assert excinfo.value.code == 2


# --- fetch: progress and -q ----------------------------------------------


def test_quiet_passes_no_callback(capsys, monkeypatch, tmp_path):
    dest = tmp_path / "x.fits"
    calls = _stub_fetch(monkeypatch, dest)

    assert _cli.main(["fetch", "WMAP", "x.fits", "-o", str(tmp_path), "-q"]) == 0
    captured = capsys.readouterr()
    assert calls[0]["on_file"] is None
    assert captured.err == ""
    assert captured.out == f"{dest}\n"  # paths still printed


def test_progress_summary_counts(capsys, monkeypatch, tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    a.write_bytes(b"abc")
    _stub_fetch(
        monkeypatch,
        [a, b],
        on_file_events=[("start", a), ("done", a), ("cached", b)],
    )

    assert _cli.main(["fetch", "WMAP", "grp", "-o", str(tmp_path)]) == 0
    summary = capsys.readouterr().err.strip().splitlines()[-1]
    assert "2 file(s)" in summary
    assert "1 downloaded" in summary
    assert "1 cached" in summary


def test_progress_class_writes_to_stderr(capsys, tmp_path):
    dest = tmp_path / "x.fits"
    dest.write_bytes(b"a" * 2048)  # "done" stats the file, so it must exist

    progress = _cli._Progress()
    progress("start", dest)
    progress("done", dest)
    progress("cached", tmp_path / "y.fits")

    captured = capsys.readouterr()
    assert progress.downloaded == 1
    assert progress.cached == 1
    assert "2.00 KB" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, "0.00 B"),
        (1023, "1023.00 B"),
        (1024, "1.00 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
    ],
)
def test_format_size(size, expected):
    assert _cli._format_size(size) == expected


# --- size ----------------------------------------------------------------


def test_size_prints_total_to_stdout(capsys, monkeypatch, tmp_path):
    _stub_size(monkeypatch, 1008000)
    assert _cli.main(["size", "WMAP", "x.fits", "-o", str(tmp_path), "-q"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "984.38 KB\n"
    assert captured.err == ""


def test_size_summary_goes_to_stderr(capsys, monkeypatch, tmp_path):
    # Keeps `lambdaquery size ... > total.txt` free of chatter.
    a, b, c = tmp_path / "a.fits", tmp_path / "b.fits", tmp_path / "c.fits"
    _stub_size(
        monkeypatch,
        1024,
        on_file_events=[("counted", a), ("cached", b), ("unknown", c)],
    )

    assert _cli.main(["size", "WMAP", "grp", "-o", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "1.00 KB\n"
    summary = captured.err.strip().splitlines()[-1]
    assert "3 file(s)" in summary
    assert "1 to download" in summary
    assert "1 cached" in summary
    assert "1 of unknown size" in summary


def test_size_names_unknown_files_on_stderr(capsys, monkeypatch, tmp_path):
    # A short total must never be silent about what it left out.
    missing = tmp_path / "gone.fits"
    _stub_size(monkeypatch, 0, on_file_events=[("unknown", missing)])

    assert _cli.main(["size", "WMAP", "gone.fits", "-o", str(tmp_path)]) == 0
    assert str(missing) in capsys.readouterr().err


def test_size_quiet_passes_no_callback(capsys, monkeypatch, tmp_path):
    calls = _stub_size(monkeypatch, 512)
    assert _cli.main(["size", "WMAP", "x.fits", "-o", str(tmp_path), "-q"]) == 0
    captured = capsys.readouterr()
    assert calls[0]["on_file"] is None
    assert captured.err == ""
    assert captured.out == "512.00 B\n"  # the total is still printed


def test_size_forwards_arguments(monkeypatch, tmp_path):
    calls = _stub_size(monkeypatch, 0)
    assert _cli.main(["size", "COBE/FIRAS", "spec.fits", "-o", str(tmp_path)]) == 0
    assert calls[0]["experiment"] == "COBE/FIRAS"
    assert calls[0]["dataset"] == "spec.fits"
    assert calls[0]["location"] == tmp_path
    assert isinstance(calls[0]["location"], Path)  # argparse type=Path


def test_size_output_defaults_to_cwd(monkeypatch):
    calls = _stub_size(monkeypatch, 0)
    assert _cli.main(["size", "WMAP", "x.fits", "-q"]) == 0
    assert calls[0]["location"] == Path(".")


def test_size_unknown_experiment_exits_1(capsys):
    # No stub: the real registry raises KeyError with the available names.
    assert _cli.main(["size", "NOPE", "x.fits"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Available experiments" in captured.err


# --- error mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ChecksumError("Checksum mismatch"),
        requests.HTTPError("404 Not Found"),
        OSError("disk full"),
        ClientError({"Error": {"Code": "SlowDown", "Message": "nope"}}, "GetObject"),
        ValueError("Reference cycle in manifest: a -> b -> a"),
    ],
)
def test_error_exits_1(capsys, monkeypatch, tmp_path, exc):
    _raising_fetch(monkeypatch, exc)
    assert _cli.main(["fetch", "WMAP", "x.fits", "-o", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")


# --- integration: the callback reaches the download layer ----------------


def test_progress_reaches_download_layer(capsys, monkeypatch, tmp_path):
    """End-to-end guard on on_file: main -> fetch_data -> _download -> _download_file.

    A dropped keyword anywhere in that chain turns this red. Only the two
    downloaders are faked, so the real registry and dispatch code run.
    """

    def fake_s3(bucket, key, location):
        location.write_bytes(b"s3-bytes")
        return location

    def fake_url(url, location):
        location.write_bytes(b"url-bytes")
        return location

    monkeypatch.setattr(_fetch, "_download_from_s3", fake_s3)
    monkeypatch.setattr(_fetch, "_download_from_url", fake_url)

    name = lambdaquery.list_datasets("WMAP")[0]  # don't hardcode a manifest name
    assert _cli.main(["fetch", "WMAP", name, "-o", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "downloading" in captured.err
    assert "1 downloaded" in captured.err
    assert captured.out.strip().startswith(str(tmp_path))


def test_progress_reports_cached_on_second_run(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(
        _fetch,
        "_download_from_s3",
        lambda bucket, key, location: location.write_bytes(b"s3-bytes"),
    )
    monkeypatch.setattr(
        _fetch,
        "_download_from_url",
        lambda url, location: location.write_bytes(b"url-bytes"),
    )

    name = lambdaquery.list_datasets("WMAP")[0]
    argv = ["fetch", "WMAP", name, "-o", str(tmp_path)]
    assert _cli.main(argv) == 0
    capsys.readouterr()  # discard the first run's output

    assert _cli.main(argv) == 0
    err = capsys.readouterr().err
    assert "cached" in err
    assert "0 downloaded, 1 cached" in err


def test_group_progress_reports_every_child(tmp_path, monkeypatch):
    """The recursion in _download must forward on_file to every child.

    Lives here rather than in test_fetch.py's CLI-free world only because the
    packaged manifest has no group entries to drive it from the command line;
    the entry is synthetic for the same reason.
    """
    monkeypatch.setattr(
        _fetch,
        "_download_from_s3",
        lambda bucket, key, location: location.write_bytes(b"s3-bytes"),
    )
    events = []

    group = DatasetEntry(
        experiment="WMAP",
        name="grp",
        children=(
            DatasetEntry(experiment="WMAP", name="a", path="/data/map/dr5/a.fits"),
            DatasetEntry(experiment="WMAP", name="b", path="/data/map/dr5/b.fits"),
        ),
    )
    _fetch._download(group, tmp_path, on_file=lambda e, p: events.append((e, p.name)))

    assert events == [
        ("start", "a.fits"),
        ("done", "a.fits"),
        ("start", "b.fits"),
        ("done", "b.fits"),
    ]


def test_size_callback_reaches_fetch_layer(capsys, monkeypatch, tmp_path):
    """End-to-end guard on on_file: main -> get_download_size -> _total_size.

    Only the size probes are faked, so the real registry and dispatch run.
    """
    monkeypatch.setattr(_fetch, "_size_from_s3", lambda bucket, key: 2048)
    monkeypatch.setattr(_fetch, "_size_from_url", lambda url: None)

    name = lambdaquery.list_datasets("WMAP")[0]  # don't hardcode a manifest name
    assert _cli.main(["size", "WMAP", name, "-o", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "2.00 KB\n"
    assert "1 file(s): 1 to download" in captured.err
