# tests/test_trust/test_session.py
from aingram.trust.session import SessionManager


def test_session_creates_with_keypair():
    sm = SessionManager(agent_name='test-agent')
    assert sm.session_id
    assert len(sm.public_key_hex) == 64
    assert sm.agent_name == 'test-agent'


def test_session_starts_at_sequence_one():
    sm = SessionManager(agent_name='test')
    assert sm.next_sequence_num == 1
    assert sm.prev_entry_id is None


def test_advance_increments_sequence():
    sm = SessionManager(agent_name='test')
    assert sm.next_sequence_num == 1
    sm.advance('entry_001')
    assert sm.next_sequence_num == 2
    assert sm.prev_entry_id == 'entry_001'


def test_advance_chains_entries():
    sm = SessionManager(agent_name='test')
    sm.advance('entry_001')
    sm.advance('entry_002')
    assert sm.next_sequence_num == 3
    assert sm.prev_entry_id == 'entry_002'


def test_to_agent_session():
    sm = SessionManager(agent_name='researcher')
    agent_session = sm.to_agent_session()
    assert agent_session.session_id == sm.session_id
    assert agent_session.agent_name == 'researcher'
    assert agent_session.public_key == sm.public_key_hex
    assert agent_session.parent_session_id is None


def test_session_with_parent():
    sm = SessionManager(agent_name='test', parent_session_id='old-session')
    assert sm.to_agent_session().parent_session_id == 'old-session'


def test_session_with_metadata():
    sm = SessionManager(agent_name='test', metadata={'model': 'sonnet'})
    assert sm.to_agent_session().metadata == {'model': 'sonnet'}


def test_sign_produces_verifiable_signature():
    """SessionManager.sign() should produce a signature verifiable with the public key."""
    from aingram.trust.signing import verify_signature

    sm = SessionManager(agent_name='test')
    entry_id = 'a' * 64
    sig = sm.sign(entry_id)
    assert len(sig) == 128  # 64 bytes hex
    assert verify_signature(sm.public_key_hex, entry_id, sig)


def test_no_private_key_property():
    """Private key should not be directly accessible as a property."""
    sm = SessionManager(agent_name='test')
    assert not hasattr(sm, 'private_key')
