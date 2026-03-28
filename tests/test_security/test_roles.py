"""RBAC permission tests."""

import pytest


class TestRoleAuthorizer:
    def _make_ctx(self, role_str):
        from aingram.security.auth import CallerContext
        from aingram.security.roles import Role

        return CallerContext(
            agent_id='test-id',
            session_id='test-session',
            role=Role(role_str),
        )

    def test_reader_can_recall(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('reader'), 'recall')

    def test_reader_cannot_remember(self):
        from aingram.exceptions import AuthorizationError
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        with pytest.raises(AuthorizationError):
            auth.check(self._make_ctx('reader'), 'remember')

    def test_contributor_can_remember(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('contributor'), 'remember')

    def test_contributor_cannot_consolidate(self):
        from aingram.exceptions import AuthorizationError
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        with pytest.raises(AuthorizationError):
            auth.check(self._make_ctx('contributor'), 'consolidate')

    def test_admin_can_consolidate(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('admin'), 'consolidate')

    def test_all_roles_can_verify(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        for role in ('reader', 'contributor', 'admin'):
            auth.check(self._make_ctx(role), 'verify')

    def test_all_roles_can_get_experiment_context(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        for role in ('reader', 'contributor', 'admin'):
            auth.check(self._make_ctx(role), 'get_experiment_context')

    def test_reader_cannot_reference(self):
        from aingram.exceptions import AuthorizationError
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        with pytest.raises(AuthorizationError):
            auth.check(self._make_ctx('reader'), 'reference')

    def test_reader_cannot_get_related(self):
        from aingram.exceptions import AuthorizationError
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        with pytest.raises(AuthorizationError):
            auth.check(self._make_ctx('reader'), 'get_related')

    def test_contributor_can_review_memory(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('contributor'), 'review_memory')

    def test_reader_can_get_due_reviews(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('reader'), 'get_due_reviews')

    def test_reader_can_get_surprise(self):
        from aingram.security.roles import RoleAuthorizer

        auth = RoleAuthorizer()
        auth.check(self._make_ctx('reader'), 'get_surprise')
