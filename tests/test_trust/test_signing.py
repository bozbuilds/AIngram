# tests/test_trust/test_signing.py
from aingram.trust.signing import (
    compute_entry_id,
    generate_keypair,
    sign_entry,
    verify_signature,
)


def test_generate_keypair_pubkey_is_64_hex():
    _private_key, pubkey_hex = generate_keypair()
    assert len(pubkey_hex) == 64
    int(pubkey_hex, 16)  # valid hex


class TestDAGHash:
    def test_deterministic(self):
        content = b'{"text":"hello"}'
        parents = ['aabb' * 8]
        pubkey = 'cc' * 32
        id1 = compute_entry_id(content, parents, pubkey)
        id2 = compute_entry_id(content, parents, pubkey)
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex

    def test_different_parents_different_hash(self):
        content = b'{"text":"hello"}'
        pubkey = 'cc' * 32
        id1 = compute_entry_id(content, ['aa' * 32], pubkey)
        id2 = compute_entry_id(content, ['bb' * 32], pubkey)
        assert id1 != id2

    def test_genesis_no_parents(self):
        content = b'{"text":"genesis"}'
        pubkey = 'cc' * 32
        entry_id = compute_entry_id(content, [], pubkey)
        assert len(entry_id) == 64

    def test_parent_order_irrelevant(self):
        """Parents are sorted internally — different input order, same hash."""
        content = b'data'
        pubkey = 'cc' * 32
        p1, p2 = 'aa' * 32, 'bb' * 32
        id_ab = compute_entry_id(content, [p1, p2], pubkey)
        id_ba = compute_entry_id(content, [p2, p1], pubkey)
        assert id_ab == id_ba

    def test_sign_verify_with_dag_id(self):
        privkey, pubkey_hex = generate_keypair()
        entry_id = compute_entry_id(b'content', [], pubkey_hex)
        sig = sign_entry(privkey, entry_id)
        assert verify_signature(pubkey_hex, entry_id, sig)

    def test_different_content_different_hash(self):
        pubkey = 'cc' * 32
        id1 = compute_entry_id(b'content A', [], pubkey)
        id2 = compute_entry_id(b'content B', [], pubkey)
        assert id1 != id2


def test_sign_and_verify_roundtrip():
    private_key, pubkey_hex = generate_keypair()
    entry_id = 'c' * 64
    signature_hex = sign_entry(private_key, entry_id)
    assert len(signature_hex) == 128  # Ed25519 sig = 64 bytes = 128 hex
    assert verify_signature(pubkey_hex, entry_id, signature_hex)


def test_verify_rejects_tampered_entry_id():
    private_key, pubkey_hex = generate_keypair()
    sig = sign_entry(private_key, 'c' * 64)
    assert not verify_signature(pubkey_hex, 'd' * 64, sig)


def test_verify_rejects_wrong_key():
    pk1, pub1 = generate_keypair()
    _pk2, pub2 = generate_keypair()
    sig = sign_entry(pk1, 'c' * 64)
    assert verify_signature(pub1, 'c' * 64, sig)
    assert not verify_signature(pub2, 'c' * 64, sig)
