"""Ownership check tests for reference() and verify()."""

import pytest

from tests.conftest import MockEmbedder


@pytest.fixture
def store(tmp_path):
    from aingram.store import MemoryStore

    db = tmp_path / 'owner.db'
    s = MemoryStore(str(db), agent_name='owner-test', embedder=MockEmbedder())
    yield s
    s.close()


class TestReferenceOwnership:
    def test_contributor_can_reference_own_entry(self, store):
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        e1 = store.remember('Entry one')
        e2 = store.remember('Entry two')
        caller = CallerContext(
            agent_id='test',
            session_id=store._session.session_id,
            role=Role.CONTRIBUTOR,
        )
        store.reference(source_id=e1, target_id=e2, reference_type='supports', caller=caller)

    def test_contributor_cannot_reference_other_session(self, store):
        from aingram.exceptions import AuthorizationError
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        e1 = store.remember('Entry one')
        e2 = store.remember('Entry two')
        caller = CallerContext(
            agent_id='other',
            session_id='different-session',
            role=Role.CONTRIBUTOR,
        )
        with pytest.raises(AuthorizationError):
            store.reference(source_id=e1, target_id=e2, reference_type='supports', caller=caller)

    def test_admin_can_reference_any_session(self, store):
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        e1 = store.remember('Entry one')
        e2 = store.remember('Entry two')
        caller = CallerContext(
            agent_id='admin',
            session_id='admin-session',
            role=Role.ADMIN,
        )
        store.reference(source_id=e1, target_id=e2, reference_type='supports', caller=caller)

    def test_no_caller_no_restriction(self, store):
        e1 = store.remember('Entry one')
        e2 = store.remember('Entry two')
        store.reference(source_id=e1, target_id=e2, reference_type='supports')


class TestVerifyOwnership:
    def test_contributor_can_verify_own_session(self, store):
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        store.remember('Test')
        caller = CallerContext(
            agent_id='test',
            session_id=store._session.session_id,
            role=Role.CONTRIBUTOR,
        )
        result = store.verify(caller=caller)
        assert result.valid is True

    def test_contributor_cannot_verify_other_session(self, store):
        from aingram.exceptions import AuthorizationError
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        store.remember('Test')
        caller = CallerContext(
            agent_id='other',
            session_id='other-session',
            role=Role.CONTRIBUTOR,
        )
        with pytest.raises(AuthorizationError):
            store.verify(session_id='some-session', caller=caller)

    def test_admin_can_verify_any_session(self, store):
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        store.remember('Test')
        caller = CallerContext(
            agent_id='admin',
            session_id='admin-session',
            role=Role.ADMIN,
        )
        result = store.verify(session_id=store._session.session_id, caller=caller)
        assert result.valid is True

    def test_no_caller_no_restriction(self, store):
        store.remember('Test')
        result = store.verify()
        assert result.valid is True
