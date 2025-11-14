"""
Hashing utilities for SAP_LLM.

Provides functions for computing hashes of files and data for caching and deduplication.
"""

import hashlib
from pathlib import Path
from typing import Union


def compute_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """
    Compute hash of data.

    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm (sha256, md5, sha1)

    Returns:
        Hex digest of hash

    Example:
        >>> compute_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)

    Returns:
        Hex digest of hash

    Example:
        >>> compute_file_hash("document.pdf")
        'a3b5c7d9e1f2a4b6c8d0e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e8f0a2b4'
    """
    file_path = Path(file_path)

    hasher = hashlib.new(algorithm)

    # Read file in chunks to handle large files
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)  # 8KB chunks
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()
