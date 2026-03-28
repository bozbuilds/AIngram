"""AIngram trust layer — content addressing, signing, verification."""

from aingram.trust.hashing import canonicalize_content, compute_content_hash
from aingram.trust.session import SessionManager
from aingram.trust.signing import (
    compute_entry_id,
    generate_keypair,
    sign_entry,
    verify_signature,
)

__all__ = [
    'canonicalize_content',
    'compute_content_hash',
    'compute_entry_id',
    'generate_keypair',
    'SessionManager',
    'sign_entry',
    'verify_signature',
]
