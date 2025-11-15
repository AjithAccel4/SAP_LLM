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

    SECURITY: Only cryptographically secure hash algorithms are allowed.
    MD5 and SHA1 are deprecated due to collision vulnerabilities.

    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm (sha256, sha384, sha512, sha3_256, sha3_512)
                  Default: sha256 (recommended)

    Returns:
        Hex digest of hash

    Raises:
        ValueError: If insecure hash algorithm is specified

    Example:
        >>> compute_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    # SECURITY: Whitelist of cryptographically secure algorithms
    SECURE_ALGORITHMS = {
        'sha256', 'sha384', 'sha512',
        'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }

    # SECURITY: Block insecure algorithms
    INSECURE_ALGORITHMS = {'md5', 'sha1'}

    if algorithm.lower() in INSECURE_ALGORITHMS:
        raise ValueError(
            f"Insecure hash algorithm '{algorithm}' is not allowed. "
            f"MD5 and SHA1 are vulnerable to collision attacks. "
            f"Use SHA-256 or stronger: {', '.join(SECURE_ALGORITHMS)}"
        )

    if algorithm.lower() not in SECURE_ALGORITHMS:
        raise ValueError(
            f"Unknown or unsupported hash algorithm '{algorithm}'. "
            f"Supported algorithms: {', '.join(SECURE_ALGORITHMS)}"
        )

    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of file.

    SECURITY: Only cryptographically secure hash algorithms are allowed.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, sha384, sha512, sha3_256, sha3_512)
                  Default: sha256 (recommended)

    Returns:
        Hex digest of hash

    Raises:
        ValueError: If insecure hash algorithm is specified

    Example:
        >>> compute_file_hash("document.pdf")
        'a3b5c7d9e1f2a4b6c8d0e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e8f0a2b4'
    """
    # SECURITY: Whitelist of cryptographically secure algorithms
    SECURE_ALGORITHMS = {
        'sha256', 'sha384', 'sha512',
        'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }

    # SECURITY: Block insecure algorithms
    INSECURE_ALGORITHMS = {'md5', 'sha1'}

    if algorithm.lower() in INSECURE_ALGORITHMS:
        raise ValueError(
            f"Insecure hash algorithm '{algorithm}' is not allowed. "
            f"MD5 and SHA1 are vulnerable to collision attacks. "
            f"Use SHA-256 or stronger: {', '.join(SECURE_ALGORITHMS)}"
        )

    if algorithm.lower() not in SECURE_ALGORITHMS:
        raise ValueError(
            f"Unknown or unsupported hash algorithm '{algorithm}'. "
            f"Supported algorithms: {', '.join(SECURE_ALGORITHMS)}"
        )

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
