# aingram/trust/signing.py
from __future__ import annotations

import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def generate_keypair() -> tuple[Ed25519PrivateKey, str]:
    """Generate an Ed25519 keypair. Returns (private_key, pubkey_hex)."""
    private_key = Ed25519PrivateKey.generate()
    pubkey_bytes = private_key.public_key().public_bytes_raw()
    return private_key, pubkey_bytes.hex()


def compute_entry_id(
    content_data: bytes,
    parent_ids: list[str],
    pubkey_hex: str,
) -> str:
    """SHA-256(content + sorted_parents + pubkey). Content-addressed DAG node.

    content_data: raw canonical JSON bytes (from canonicalize_content).
    parent_ids: list of parent entry_id hex strings (sorted internally).
    pubkey_hex: author's Ed25519 public key as hex string.
    """
    h = hashlib.sha256(content_data)
    for pid in sorted(parent_ids):
        h.update(bytes.fromhex(pid))
    h.update(bytes.fromhex(pubkey_hex))
    return h.hexdigest()


def sign_entry(private_key: Ed25519PrivateKey, entry_id: str) -> str:
    """Sign an entry_id with Ed25519. Returns 128-char hex signature."""
    sig_bytes = private_key.sign(entry_id.encode('utf-8'))
    return sig_bytes.hex()


def verify_signature(pubkey_hex: str, entry_id: str, signature_hex: str) -> bool:
    """Verify an Ed25519 signature. Returns True if valid, False otherwise."""
    try:
        pub_bytes = bytes.fromhex(pubkey_hex)
        sig_bytes = bytes.fromhex(signature_hex)
        public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        public_key.verify(sig_bytes, entry_id.encode('utf-8'))
        return True
    except (InvalidSignature, ValueError):
        return False
