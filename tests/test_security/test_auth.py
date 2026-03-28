"""Token authentication and agent management tests."""

import pytest


@pytest.fixture
def engine(tmp_path):
    from aingram.storage.engine import StorageEngine

    eng = StorageEngine(str(tmp_path / 'auth.db'))
    yield eng
    eng.close()


class TestAgentTokenCRUD:
    def test_create_agent_token(self, engine):
        result = engine.create_agent_token(
            agent_name='researcher-1',
            role='contributor',
        )
        assert 'agent_id' in result
        assert 'token' in result
        assert len(result['token']) > 20

    def test_verify_valid_token(self, engine):
        result = engine.create_agent_token(agent_name='test', role='reader')
        agent = engine.verify_agent_token(result['token'])
        assert agent is not None
        assert agent['agent_name'] == 'test'
        assert agent['role'] == 'reader'

    def test_verify_invalid_token(self, engine):
        agent = engine.verify_agent_token('bogus-token')
        assert agent is None

    def test_verify_revoked_token(self, engine):
        result = engine.create_agent_token(agent_name='temp', role='contributor')
        engine.revoke_agent_token('temp')
        agent = engine.verify_agent_token(result['token'])
        assert agent is None

    def test_list_agents(self, engine):
        engine.create_agent_token(agent_name='a', role='reader')
        engine.create_agent_token(agent_name='b', role='admin')
        agents = engine.list_agent_tokens()
        names = {a['agent_name'] for a in agents}
        assert names == {'a', 'b'}

    def test_duplicate_agent_name_raises(self, engine):
        engine.create_agent_token(agent_name='unique', role='reader')
        with pytest.raises(Exception):
            engine.create_agent_token(agent_name='unique', role='admin')

    def test_create_with_pubkey(self, engine):
        from aingram.trust.signing import generate_keypair

        _, pubkey_hex = generate_keypair()
        result = engine.create_agent_token(
            agent_name='signed-agent', role='contributor', public_key=pubkey_hex
        )
        agent = engine.verify_agent_token(result['token'])
        assert agent['public_key'] == pubkey_hex

    def test_invalid_role_raises(self, engine):
        with pytest.raises(ValueError, match='role'):
            engine.create_agent_token(agent_name='bad', role='superadmin')


class TestTokenAuthenticator:
    def test_authenticate_valid_token(self, engine):
        from aingram.security.auth import TokenAuthenticator

        result = engine.create_agent_token(agent_name='test', role='contributor')
        auth = TokenAuthenticator(engine)
        ctx = auth.authenticate(result['token'])
        assert ctx.agent_id == result['agent_id']
        assert ctx.role.value == 'contributor'

    def test_authenticate_invalid_raises(self, engine):
        from aingram.exceptions import AuthenticationError
        from aingram.security.auth import TokenAuthenticator

        auth = TokenAuthenticator(engine)
        with pytest.raises(AuthenticationError):
            auth.authenticate('bad-token')

    def test_authenticate_revoked_raises(self, engine):
        from aingram.exceptions import AuthenticationError
        from aingram.security.auth import TokenAuthenticator

        result = engine.create_agent_token(agent_name='revokee', role='reader')
        engine.revoke_agent_token('revokee')
        auth = TokenAuthenticator(engine)
        with pytest.raises(AuthenticationError):
            auth.authenticate(result['token'])

    def test_caller_context_is_frozen(self, engine):
        from aingram.security.auth import TokenAuthenticator

        result = engine.create_agent_token(agent_name='freeze', role='admin')
        auth = TokenAuthenticator(engine)
        ctx = auth.authenticate(result['token'])
        with pytest.raises(AttributeError):
            ctx.role = 'reader'  # type: ignore[misc]
