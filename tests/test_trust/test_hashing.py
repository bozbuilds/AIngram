# tests/test_trust/test_hashing.py
import hashlib

import rfc8785

from aingram.trust.hashing import canonicalize_content, compute_content_hash


def test_canonicalize_string_wraps_as_text():
    result = canonicalize_content('hello world')
    assert result == rfc8785.dumps({'text': 'hello world'})


def test_canonicalize_dict_uses_directly():
    result = canonicalize_content({'b': 2, 'a': 1})
    assert result == rfc8785.dumps({'a': 1, 'b': 2})


def test_content_hash_has_type_length_header():
    """Git-style: SHA-256(entry_type + space + byte_length + NUL + canonical_bytes)."""
    payload = {'text': 'hello'}
    canonical_bytes = rfc8785.dumps(payload)
    header = f'observation {len(canonical_bytes)}\0'.encode()
    expected = hashlib.sha256(header + canonical_bytes).hexdigest()

    assert compute_content_hash(payload, 'observation') == expected


def test_different_types_different_hashes():
    payload = {'text': 'same content'}
    h1 = compute_content_hash(payload, 'observation')
    h2 = compute_content_hash(payload, 'hypothesis')
    assert h1 != h2


def test_content_hash_deterministic():
    payload = {'text': 'deterministic'}
    assert compute_content_hash(payload, 'observation') == compute_content_hash(
        payload, 'observation'
    )


def test_content_hash_is_64_hex_chars():
    result = compute_content_hash({'text': 'test'}, 'observation')
    assert len(result) == 64
    int(result, 16)  # valid hex
