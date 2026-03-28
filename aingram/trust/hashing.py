# aingram/trust/hashing.py
from __future__ import annotations

import hashlib

import rfc8785


def canonicalize_content(content: str | dict) -> bytes:
    """Canonicalize content via RFC 8785 (JCS).

    String content is wrapped as {"text": "..."}.
    Dict content is used directly.
    Returns canonical bytes.
    """
    if isinstance(content, str):
        payload = {'text': content}
    else:
        payload = content
    return rfc8785.dumps(payload)


def compute_content_hash(content_payload: dict, entry_type: str) -> str:
    """Compute content_hash with Git-style type-length header.

    SHA-256(entry_type + ' ' + byte_length + '\\0' + canonical_bytes)
    """
    canonical_bytes = rfc8785.dumps(content_payload)
    header = f'{entry_type} {len(canonical_bytes)}\0'.encode()
    return hashlib.sha256(header + canonical_bytes).hexdigest()
