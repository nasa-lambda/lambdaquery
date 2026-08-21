"""Download layer for lambdaquery.

Walks a resolved :class:`~lambdaquery._registry.DatasetEntry` tree and downloads
every file leaf, from either the LAMBDA HTTPS server or the public ``nasa-lambda``
S3 mirror. Files are stored at a canonical location derived from their ``/data/``
path, so a file referenced by several datasets is downloaded and stored once.
"""

import os
import re
from collections.abc import Callable
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
        if on_file is not None:
            on_file("cached", dest)
        return dest

    if on_file is not None:
        on_file("start", dest)

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

    if on_file is not None:
        on_file("done", dest)
    return dest


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
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    client.download_file(bucket, key, str(tmp))
    os.replace(tmp, location)
    return location


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
