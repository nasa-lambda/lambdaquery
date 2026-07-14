from dataclasses import dataclass, field
from pathlib import Path

import yaml

_MANIFEST_PATH = Path(__file__).parent / "data" / "manifest.yaml"


@dataclass(frozen=True)
class DatasetEntry:
    """A resolved node in the dataset tree.

    A node is either a *file leaf* (``path`` is set) or a *group* (``children``
    is populated). ``_fetch``/``_cache`` depend only on this normalized shape --
    the manifest layout can change without touching them.
    """

    experiment: str  # top-level experiment; drives S3 dispatch
    name: str  # dataset / group / entry name
    description: str = ""
    path: str | None = None  # set on FILE leaves; "/data/.../file.fits"
    checksum: str | None = None  # optional, "<algo>:<hex>" or bare md5 hex
    size: int | None = None  # optional expected byte size
    children: tuple["DatasetEntry", ...] = field(default_factory=tuple)

    @property
    def is_file(self) -> bool:
        return self.path is not None


class Registry:
    def __init__(self, manifest_path: Path = _MANIFEST_PATH):
        with open(manifest_path, "r") as f:
            self._data = yaml.safe_load(f)

    def list_experiments(self) -> list[str]:
        return sorted(self._data.keys())

    def list_datasets(self, experiment: str) -> list[str]:
        self._check_experiment_exists(experiment)
        return sorted(self._data[experiment].keys())

    def get(self, experiment: str, dataset: str) -> DatasetEntry:
        self._check_experiment_exists(experiment)
        return self._build_entry(experiment, dataset)

    # --- manifest parsing -------------------------------------------------
    # Everything below understands the *raw manifest shape* and is the only
    # part that changes when the manifest changes. Under each experiment an
    # entry is either "name: /data/.../file.fits" (a file leaf) or
    # "name: [other_entry_name, ...]" (a group referencing sibling entries by
    # name). References are resolved recursively.

    def _build_entry(
        self,
        experiment: str,
        name: str,
        _chain: tuple[str, ...] = (),
    ) -> DatasetEntry:
        if name in _chain:
            cycle = " -> ".join([*_chain, name])
            raise ValueError(f"Reference cycle in manifest: {cycle}")

        entries = self._data[experiment]
        if name not in entries:
            raise KeyError(
                f"Entry '{name}' not found in experiment '{experiment}'. "
                f"Available entries: {sorted(entries.keys())}"
            )

        value = entries[name]
        if isinstance(value, str):
            return DatasetEntry(experiment=experiment, name=name, path=value)

        if isinstance(value, list):
            children = tuple(
                self._build_entry(experiment, child, (*_chain, name)) for child in value
            )
            return DatasetEntry(experiment=experiment, name=name, children=children)

        raise TypeError(
            f"Entry '{experiment}/{name}' must be a path string or a list of "
            f"entry names, got {type(value).__name__}"
        )

    def _check_experiment_exists(self, experiment: str):
        if experiment not in self._data:
            raise KeyError(
                f"Experiment '{experiment}' not found. "
                f"Available experiments: {self.list_experiments()}"
            )


_registry = Registry()
