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

    When the manifest supplies a size and/or checksum they must match; otherwise
    a file is considered cached as long as it exists and is non-empty.
    """
    if not path.is_file():
        return False
    try:
        _check(path, entry)
    except ChecksumError:
        return False
    return path.stat().st_size > 0


def verify(path: Path, entry: DatasetEntry) -> None:
    """Verify a freshly downloaded file, deleting and raising on mismatch.

    No-op when the manifest provides neither a size nor a checksum.
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


def _check(path: Path, entry: DatasetEntry) -> None:
    """Compare ``path`` against the size/checksum in ``entry``.

    Raises ChecksumError on mismatch; returns silently when there is nothing to
    check.
    """
    if entry.size is not None and path.stat().st_size != entry.size:
        raise ChecksumError(
            f"Size mismatch for {path}: expected {entry.size}, "
            f"got {path.stat().st_size}"
        )

    if entry.checksum is not None:
        algo, _, expected = entry.checksum.rpartition(":")
        algo = algo or _DEFAULT_ALGO
        actual = compute_checksum(path, algo)
        if actual != expected:
            raise ChecksumError(
                f"Checksum mismatch for {path}: expected {algo}:{expected}, "
                f"got {algo}:{actual}"
            )
