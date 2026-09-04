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
    # entry value takes one of four forms:
    #
    #   name: /data/.../file.fits          a file leaf (shorthand)
    #   name: [other_entry_name, ...]      a group of siblings (shorthand)
    #   name: {path: ..., size: ...}       a file leaf carrying metadata
    #   name: {children: [...], ...}       a group carrying metadata
    #
    # The mapping forms are distinguished by which key is present: 'path' means
    # a leaf, 'children' means a group. References are resolved recursively.

    _LEAF_KEYS = frozenset({"path", "size", "checksum", "description"})
    _GROUP_KEYS = frozenset({"children", "description"})

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
            return self._build_group(experiment, name, value, _chain)

        if isinstance(value, dict):
            return self._build_mapping(experiment, name, value, _chain)

        raise TypeError(
            f"Entry '{experiment}/{name}' must be a path string, a list of entry "
            f"names, or a mapping, got {type(value).__name__}"
        )

    def _build_group(
        self,
        experiment: str,
        name: str,
        children: list,
        _chain: tuple[str, ...],
        description: str = "",
    ) -> DatasetEntry:
        return DatasetEntry(
            experiment=experiment,
            name=name,
            description=description,
            children=tuple(
                self._build_entry(experiment, child, (*_chain, name))
                for child in children
            ),
        )

    def _build_mapping(
        self,
        experiment: str,
        name: str,
        value: dict,
        _chain: tuple[str, ...],
    ) -> DatasetEntry:
        """Build an entry from the mapping form, validating its keys.

        Unknown keys are rejected rather than ignored: a typo like 'checkum'
        would otherwise silently disable verification for that file.
        """
        where = f"{experiment}/{name}"
        has_path = "path" in value
        has_children = "children" in value
        if has_path and has_children:
            raise ValueError(
                f"Entry '{where}' sets both 'path' and 'children'; an entry is "
                f"either a file or a group, not both"
            )
        if not has_path and not has_children:
            raise ValueError(
                f"Entry '{where}' must set either 'path' (a file) or 'children' "
                f"(a group); got keys {sorted(value)}"
            )

        allowed = self._LEAF_KEYS if has_path else self._GROUP_KEYS
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"Entry '{where}' has unknown key(s) {unknown}; "
                f"allowed here: {sorted(allowed)}"
            )

        description = self._as_str(value.get("description", ""), where, "description")

        if has_children:
            children = value["children"]
            if not isinstance(children, list):
                raise TypeError(
                    f"Entry '{where}' key 'children' must be a list of entry "
                    f"names, got {type(children).__name__}"
                )
            return self._build_group(
                experiment, name, children, _chain, description=description
            )

        return DatasetEntry(
            experiment=experiment,
            name=name,
            description=description,
            path=self._as_str(value["path"], where, "path"),
            checksum=self._optional_str(value.get("checksum"), where, "checksum"),
            size=self._as_size(value.get("size"), where),
        )

    # --- mapping-form field validation ------------------------------------
    # Validate here so a malformed manifest fails in Registry.get, naming the
    # offending entry, rather than deep inside _cache._check at download time.

    @staticmethod
    def _as_str(value, where: str, key: str) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"Entry '{where}' key '{key}' must be a string, "
                f"got {type(value).__name__}"
            )
        return value

    @classmethod
    def _optional_str(cls, value, where: str, key: str) -> str | None:
        return None if value is None else cls._as_str(value, where, key)

    @staticmethod
    def _as_size(value, where: str) -> int | None:
        if value is None:
            return None
        # bool is an int subclass; "size: true" is a mistake, not a size.
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Entry '{where}' key 'size' must be an integer number of bytes, "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"Entry '{where}' key 'size' must not be negative")
        return value

    def _check_experiment_exists(self, experiment: str):
        if experiment not in self._data:
            raise KeyError(
                f"Experiment '{experiment}' not found. "
                f"Available experiments: {self.list_experiments()}"
            )


_registry = Registry()
