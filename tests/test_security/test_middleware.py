"""SecurityMiddleware integration tests."""

import pytest


@pytest.fixture
def engine(tmp_path):
    from aingram.storage.engine import StorageEngine

    eng = StorageEngine(str(tmp_path / 'mw.db'))
    yield eng
    eng.close()


class TestSecurityMiddleware:
    def test_valid_token_returns_context(self, engine):
        from aingram.security.middleware import SecurityMiddleware

        result = engine.create_agent_token(agent_name='mw-test', role='contributor')
        mw = SecurityMiddleware(engine)
        ctx = mw.process('remember', result['token'], {'content': 'hello'})
        assert ctx.agent_id == result['agent_id']

    def test_invalid_token_raises(self, engine):
        from aingram.exceptions import AuthenticationError
        from aingram.security.middleware import SecurityMiddleware

        mw = SecurityMiddleware(engine)
        with pytest.raises(AuthenticationError):
            mw.process('recall', 'bad-token', {})

    def test_unauthorized_tool_raises(self, engine):
        from aingram.exceptions import AuthorizationError
        from aingram.security.middleware import SecurityMiddleware

        result = engine.create_agent_token(agent_name='reader', role='reader')
        mw = SecurityMiddleware(engine)
        with pytest.raises(AuthorizationError):
            mw.process('remember', result['token'], {'content': 'hello'})

    def test_bounds_clamped(self, engine):
        from aingram.security.middleware import SecurityMiddleware

        result = engine.create_agent_token(agent_name='test', role='reader')
        mw = SecurityMiddleware(engine)
        params = {'limit': 500}
        mw.process('recall', result['token'], params)
        assert params['limit'] == 100

    def test_oversized_content_rejected(self, engine):
        from aingram.exceptions import InputBoundsError
        from aingram.security.middleware import SecurityMiddleware

        result = engine.create_agent_token(agent_name='writer', role='contributor')
        mw = SecurityMiddleware(engine)
        with pytest.raises(InputBoundsError):
            mw.process('remember', result['token'], {'content': 'x' * 65_537})

    def test_full_pipeline_order(self, engine):
        from aingram.security.middleware import SecurityMiddleware

        result = engine.create_agent_token(agent_name='full', role='contributor')
        mw = SecurityMiddleware(engine)
        ctx = mw.process('remember', result['token'], {'content': 'ok'})
        assert ctx.role.value == 'contributor'
