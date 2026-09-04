"""Download layer for lambdaquery.

Walks a resolved :class:`~lambdaquery._registry.DatasetEntry` tree and downloads
every file leaf, from either the LAMBDA HTTPS server or the public ``nasa-lambda``
S3 mirror. Files are stored at a canonical location derived from their ``/data/``
path, so a file referenced by several datasets is downloaded and stored once.

The same tree walk also backs a download-free size query: :func:`_total_size`
asks each source how big a leaf is instead of fetching it.
"""

import functools
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from . import _cache
from ._registry import DatasetEntry

ProgressCallback = Callable[[str, Path], None]

# Experiments whose files are also mirrored in the public S3 bucket. Names must
# match the manifest's experiment keys exactly -- the COBE instruments are keyed
# "COBE/<instrument>". The bucket mirrors two top-level trees, cobe/ and wmap/.
s3_exps = {"COBE/DIRBE", "COBE/DMR", "COBE/FIRAS", "WMAP"}

LAMBDA_BASE = "https://lambda.gsfc.nasa.gov"
S3_BUCKET = "nasa-lambda"

# Per-experiment S3 key rewrites: the LAMBDA path's first segment differs from
# the S3 key's first segment. (first, replacement) applied after stripping /data/.
S3_KEY_REWRITE = {"WMAP": ("map/", "wmap/")}

_CHUNK_SIZE = 1 << 20  # 1 MiB

# S3 error codes meaning "this object is not on the mirror" (as opposed to a
# transport/permission failure, which must not be swallowed).
_S3_MISSING_CODES = {"404", "NoSuchKey"}

_HEAD_TIMEOUT = 30  # seconds; size probes must not hang a listing


def _download(
    entry: DatasetEntry,
    location: Path,
    on_file: ProgressCallback | None = None,
) -> Path | list[Path]:
    """Download ``entry`` under ``location``.

    Returns the file path for a leaf, or a flat list of every downloaded file
    (tree order) for a group.
    """
    if entry.path is not None:
        dest = _local_path(entry.path, location)
        return _download_file(entry, dest, on_file=on_file)

    paths: list[Path] = []
    for child in entry.children:
        result = _download(child, location, on_file=on_file)
        if isinstance(result, list):
            paths.extend(result)
        else:
            paths.append(result)
    return paths


def _iter_leaves(entry: DatasetEntry) -> Iterator[DatasetEntry]:
    """Yield every file leaf under ``entry``, in tree order."""
    if entry.path is not None:
        yield entry
        return
    for child in entry.children:
        yield from _iter_leaves(child)


def _leaf_names(entry: DatasetEntry) -> list[str]:
    """Names of every distinct file leaf under ``entry``, in tree order.

    Deduped on the same key ``_total_size`` uses, so a listing and a size query
    agree on the file count and the manifest's doubled slashes normalize away.
    """
    seen: set[Path] = set()
    names: list[str] = []
    for leaf in _iter_leaves(entry):
        assert leaf.path is not None  # _iter_leaves only yields file leaves
        key = _local_path(leaf.path, Path("."))
        if key in seen:
            continue
        seen.add(key)
        names.append(leaf.name)
    return names


def _notify(on_file: ProgressCallback | None, event: str, path: Path) -> None:
    if on_file is not None:
        on_file(event, path)


def _local_path(path: str, location: Path) -> Path:
    """Canonical on-disk location for a ``/data/...`` file path.

    Mirrors the LAMBDA directory tree under ``location`` so the same path always
    maps to the same file (enabling dedup across datasets).
    """
    return location / path.lstrip("/")


def _download_file(
    entry: DatasetEntry,
    dest: Path,
    on_file: ProgressCallback | None = None,
) -> Path:
    assert entry.path is not None  # leaf invariant; enforced by callers
    if _cache.is_cached(dest, entry):
        _notify(on_file, "cached", dest)
        return dest

    _notify(on_file, "start", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _lambda_url(entry.path)
    if entry.experiment in s3_exps:
        bucket, key = _lambda_path_to_s3(entry.experiment, entry.path)
        try:
            _download_from_s3(bucket, key, dest)
        except ClientError as err:
            # The S3 mirror is partial -- e.g. WMAP's 1-year release and much of
            # its TOD/sim data are HTTPS-only. Fall back rather than fail; other
            # S3 errors (throttling, transport) still propagate.
            code = err.response.get("Error", {}).get("Code")
            if code not in _S3_MISSING_CODES:
                raise
            _download_from_url(url, dest)
    else:
        _download_from_url(url, dest)

    _cache.verify(dest, entry)

    _notify(on_file, "done", dest)
    return dest


def _total_size(
    entry: DatasetEntry,
    location: Path,
    on_file: ProgressCallback | None = None,
) -> int:
    """Bytes that ``_download(entry, location)`` would actually transfer.

    Files already cached under ``location`` count as zero, and a leaf reachable
    by several paths through the tree counts once. Leaves whose size cannot be
    determined are skipped rather than failing the whole query.

    Each distinct leaf is reported through ``on_file`` exactly once, as
    ``("counted" | "cached" | "unknown", path)``.
    """
    total = 0
    seen: set[Path] = set()
    for leaf in _iter_leaves(entry):
        assert leaf.path is not None  # _iter_leaves yields file leaves only
        dest = _local_path(leaf.path, location)
        # _download gets its dedup for free -- the second visit to a shared leaf
        # finds the file already on disk. Nothing is written here, so the walk
        # has to remember. _local_path collapses the manifest's doubled slashes,
        # so both spellings of a path land on one key.
        if dest in seen:
            continue
        seen.add(dest)

        if _cache.is_cached(dest, leaf):
            _notify(on_file, "cached", dest)
            continue

        size = _remote_size(leaf)
        if size is None:
            _notify(on_file, "unknown", dest)
            continue
        total += size
        _notify(on_file, "counted", dest)
    return total


def _remote_size(entry: DatasetEntry) -> int | None:
    """Byte size for a file leaf, or None if it cannot be determined.

    Prefers the manifest's own ``size:`` when present, so a fully-populated
    dataset answers offline with no requests at all; entries without one still
    fall back to asking S3 and then HTTPS.
    """
    assert entry.path is not None
    if entry.size is not None:
        return entry.size
    if entry.experiment in s3_exps:
        bucket, key = _lambda_path_to_s3(entry.experiment, entry.path)
        size = _size_from_s3(bucket, key)
        if size is not None:
            return size
    return _size_from_url(_lambda_url(entry.path))


def _size_from_s3(bucket: str, key: str) -> int | None:
    try:
        resp = _s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError:
        # Deliberately broader than _download_file's _S3_MISSING_CODES check.
        # There, falling back means re-transferring a whole file, so masking a
        # throttling error as a slow path is expensive. Here the fallback is one
        # cheap HEAD against HTTPS, which serves the full archive anyway -- so
        # any S3 failure just defers to it, and only a failure of both is fatal
        # to the estimate.
        return None
    return resp["ContentLength"]


def _size_from_url(url: str) -> int | None:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=_HEAD_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    length = resp.headers.get("Content-Length")
    if length is None or not length.isdigit():
        return None
    return int(length)


def _download_from_url(url: str, location: Path) -> Path:
    """Stream ``url`` to ``location`` via a temp file, then atomically rename."""
    tmp = _tmp_path(location)
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                f.write(chunk)
    os.replace(tmp, location)
    return location


def _download_from_s3(bucket: str, key: str, location: Path) -> Path:
    """Download an S3 object to a temp file, then atomically rename.

    Uses unsigned requests -- the ``nasa-lambda`` bucket is public.
    """
    tmp = _tmp_path(location)
    _s3_client().download_file(bucket, key, str(tmp))
    os.replace(tmp, location)
    return location


@functools.cache
def _s3_client():
    """The shared unsigned S3 client -- the ``nasa-lambda`` bucket is public.

    Cached because a size walk over a large group would otherwise pay client
    construction once per leaf.
    """
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _lambda_url(path: str) -> str:
    return LAMBDA_BASE + path


def _lambda_path_to_s3(experiment: str, path: str) -> tuple[str, str]:
    """Map a ``/data/...`` path to an ``(bucket, key)`` on the S3 mirror.

    Repeated slashes are collapsed: the manifest contains paths like
    ``/data/map/powspec//file.txt``, which the HTTPS server tolerates but which
    S3 would treat as a distinct (nonexistent) key.
    """
    key = re.sub(r"/+", "/", path).removeprefix("/data/")
    rewrite = S3_KEY_REWRITE.get(experiment)
    if rewrite is not None:
        first, replacement = rewrite
        if key.startswith(first):
            key = replacement + key[len(first) :]
    return S3_BUCKET, key


def _tmp_path(location: Path) -> Path:
    return location.with_suffix(location.suffix + ".part")
