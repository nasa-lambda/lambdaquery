from pathlib import Path

from ._fetch import ProgressCallback, _download, _leaf_names, _total_size
from ._registry import _registry


def list_experiments() -> list[str]:
    return _registry.list_experiments()


def list_datasets(experiment: str) -> list[str]:
    return _registry.list_datasets(experiment)


def list_files(experiment: str, dataset: str) -> list[str]:
    """Entry names of every file ``fetch_data(experiment, dataset, ...)`` downloads.

    A single-file dataset returns ``[dataset]``. Group members come back in tree
    order with duplicates removed. Reads the manifest only -- no network access.
    """
    entry = _registry.get(experiment, dataset)
    return _leaf_names(entry)


def fetch_data(
    experiment: str,
    dataset: str,
    location: str | Path,
    *,
    on_file: ProgressCallback | None = None,
) -> Path | list[Path]:
    entry = _registry.get(experiment, dataset)
    return _download(entry, Path(location), on_file=on_file)


def get_download_size(
    experiment: str,
    dataset: str,
    location: str | Path,
    *,
    on_file: ProgressCallback | None = None,
) -> int:
    """Total bytes ``fetch_data(experiment, dataset, location)`` would download.

    Files already cached under ``location`` are excluded, as is a duplicate
    reference to a leaf a dataset reaches more than once. Sizes come from the
    network (the manifest carries none), so this makes one lightweight request
    per uncached file.
    """
    entry = _registry.get(experiment, dataset)
    return _total_size(entry, Path(location), on_file=on_file)


__all__ = [
    "list_experiments",
    "list_datasets",
    "list_files",
    "fetch_data",
    "get_download_size",
]
