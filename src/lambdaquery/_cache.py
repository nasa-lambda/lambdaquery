"""Cache management for lambdaquery.

Handles deciding whether a file is already present locally and verifying that a
downloaded file is complete/correct.
"""

import hashlib
from pathlib import Path

from ._registry import DatasetEntry

_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_ALGO = "md5"  # LAMBDA publishes md5 checksums


class ChecksumError(Exception):
    """Raised when a downloaded file fails size/checksum verification."""


def is_cached(path: Path, entry: DatasetEntry) -> bool:
    """Return True if ``path`` already holds a valid copy of ``entry``.

    When the manifest supplies a size it must match; otherwise a file is
    considered cached as long as it exists and is non-empty.

    Deliberately checks the size but *not* the checksum. This runs on every
    ``fetch_data`` and once per leaf in a size walk, so hashing here would mean
    re-reading every cached byte just to answer "is it already there?" -- for a
    large group, minutes of disk I/O to conclude there is nothing to download.
    Size catches truncation, the realistic cache corruption, in one stat; the
    checksum still gates every file at the moment it lands, via `verify`.
    """
    if not path.is_file():
        return False
    try:
        _check(path, entry, checksum=False)
    except ChecksumError:
        return False
    return path.stat().st_size > 0


def verify(path: Path, entry: DatasetEntry) -> None:
    """Verify a freshly downloaded file, deleting and raising on mismatch.

    Checks size *and* checksum -- unlike `is_cached`, this runs once per actual
    download, so the file has just been read anyway. No-op when the manifest
    provides neither.
    """
    try:
        _check(path, entry)
    except ChecksumError:
        path.unlink(missing_ok=True)
        raise


def compute_checksum(path: Path, algo: str = _DEFAULT_ALGO) -> str:
    """Return the hex digest of ``path`` using ``algo`` (streamed)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, entry: DatasetEntry, checksum: bool = True) -> None:
    """Compare ``path`` against the size/checksum in ``entry``.

    Raises ChecksumError on mismatch; returns silently when there is nothing to
    check. ``checksum=False`` skips the (whole-file) digest and checks size only.
    """
    if entry.size is not None and path.stat().st_size != entry.size:
        raise ChecksumError(
            f"Size mismatch for {path}: expected {entry.size}, "
            f"got {path.stat().st_size}"
        )

    if checksum and entry.checksum is not None:
        algo, _, expected = entry.checksum.rpartition(":")
        algo = algo or _DEFAULT_ALGO
        actual = compute_checksum(path, algo)
        if actual != expected:
            raise ChecksumError(
                f"Checksum mismatch for {path}: expected {algo}:{expected}, "
                f"got {algo}:{actual}"
            )
