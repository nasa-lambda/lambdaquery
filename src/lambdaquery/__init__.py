from pathlib import Path

from ._fetch import ProgressCallback, _download, _total_size
from ._registry import _registry


def list_experiments() -> list[str]:
    return _registry.list_experiments()


def list_datasets(experiment: str) -> list[str]:
    return _registry.list_datasets(experiment)


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


__all__ = ["list_experiments", "list_datasets", "fetch_data", "get_download_size"]
