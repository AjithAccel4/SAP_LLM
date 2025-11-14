"""
APOP Signature - ECDSA signing and verification for envelopes

Provides cryptographic signing of APOP envelopes to ensure integrity
and authenticity.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SignatureManager:
    """
    Manager for APOP envelope signatures.

    Uses ECDSA (Elliptic Curve Digital Signature Algorithm) with secp256r1 curve.
    """

    def __init__(
        self,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        """
        Initialize signature manager.

        Args:
            private_key_path: Path to private key PEM file
            public_key_path: Path to public key PEM file
        """
        self.private_key = None
        self.public_key = None

        # Load keys if paths provided
        if private_key_path:
            self.load_private_key(private_key_path)

        if public_key_path:
            self.load_public_key(public_key_path)

    def generate_key_pair(
        self,
        private_key_path: str,
        public_key_path: str,
    ) -> None:
        """
        Generate new ECDSA key pair.

        Args:
            private_key_path: Path to save private key
            public_key_path: Path to save public key
        """
        # Generate private key
        private_key = ec.generate_private_key(
            ec.SECP256R1(),
            default_backend(),
        )

        # Get public key
        public_key = private_key.public_key()

        # Save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        Path(private_key_path).parent.mkdir(parents=True, exist_ok=True)
        with open(private_key_path, "wb") as f:
            f.write(private_pem)

        # Save public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        Path(public_key_path).parent.mkdir(parents=True, exist_ok=True)
        with open(public_key_path, "wb") as f:
            f.write(public_pem)

        logger.info(f"Generated key pair: {private_key_path}, {public_key_path}")

        self.private_key = private_key
        self.public_key = public_key

    def load_private_key(self, path: str) -> None:
        """Load private key from PEM file."""
        try:
            with open(path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend(),
                )
            logger.debug(f"Loaded private key from {path}")
        except FileNotFoundError:
            logger.warning(f"Private key not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")

    def load_public_key(self, path: str) -> None:
        """Load public key from PEM file."""
        try:
            with open(path, "rb") as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend(),
                )
            logger.debug(f"Loaded public key from {path}")
        except FileNotFoundError:
            logger.warning(f"Public key not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load public key: {e}")

    def sign_envelope(self, envelope_dict: Dict[str, Any]) -> str:
        """
        Sign APOP envelope.

        Args:
            envelope_dict: Envelope as dictionary (without signature field)

        Returns:
            Hex-encoded signature

        Raises:
            ValueError: If private key not loaded
        """
        if self.private_key is None:
            raise ValueError("Private key not loaded")

        # Create canonical representation
        canonical = self._canonicalize(envelope_dict)

        # Sign
        signature = self.private_key.sign(
            canonical,
            ec.ECDSA(hashes.SHA256()),
        )

        # Return as hex string
        return signature.hex()

    def verify_signature(
        self,
        envelope_dict: Dict[str, Any],
        signature_hex: str,
    ) -> bool:
        """
        Verify envelope signature.

        Args:
            envelope_dict: Envelope dictionary (without signature field)
            signature_hex: Hex-encoded signature

        Returns:
            True if signature is valid, False otherwise
        """
        if self.public_key is None:
            logger.warning("Public key not loaded, cannot verify signature")
            return False

        try:
            # Create canonical representation
            canonical = self._canonicalize(envelope_dict)

            # Decode signature
            signature = bytes.fromhex(signature_hex)

            # Verify
            self.public_key.verify(
                signature,
                canonical,
                ec.ECDSA(hashes.SHA256()),
            )

            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def _canonicalize(self, envelope_dict: Dict[str, Any]) -> bytes:
        """
        Create canonical representation of envelope for signing.

        Removes signature field and serializes deterministically.

        Args:
            envelope_dict: Envelope dictionary

        Returns:
            Canonical byte representation
        """
        # Remove signature if present
        envelope_copy = {
            k: v for k, v in envelope_dict.items()
            if k != "signature"
        }

        # Serialize with sorted keys for determinism
        canonical_json = json.dumps(
            envelope_copy,
            sort_keys=True,
            separators=(",", ":"),  # No whitespace
        )

        # Encode to bytes
        return canonical_json.encode("utf-8")


# Module-level signature manager (singleton)
_signature_manager: Optional[SignatureManager] = None


def get_signature_manager(
    private_key_path: Optional[str] = None,
    public_key_path: Optional[str] = None,
) -> SignatureManager:
    """
    Get or create signature manager.

    Args:
        private_key_path: Optional path to private key
        public_key_path: Optional path to public key

    Returns:
        SignatureManager instance
    """
    global _signature_manager

    if _signature_manager is None:
        _signature_manager = SignatureManager(
            private_key_path=private_key_path,
            public_key_path=public_key_path,
        )

    return _signature_manager


def sign_envelope(envelope_dict: Dict[str, Any]) -> str:
    """
    Sign envelope using module-level signature manager.

    Args:
        envelope_dict: Envelope dictionary

    Returns:
        Hex-encoded signature
    """
    manager = get_signature_manager()
    return manager.sign_envelope(envelope_dict)


def verify_signature(
    envelope_dict: Dict[str, Any],
    signature_hex: str,
) -> bool:
    """
    Verify envelope signature using module-level signature manager.

    Args:
        envelope_dict: Envelope dictionary
        signature_hex: Hex-encoded signature

    Returns:
        True if valid, False otherwise
    """
    manager = get_signature_manager()
    return manager.verify_signature(envelope_dict, signature_hex)
