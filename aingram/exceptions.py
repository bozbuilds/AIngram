# aingram/exceptions.py


class AIngramError(Exception):
    """Base exception for all aingram errors."""


class DatabaseError(AIngramError):
    """Raised for SQLite and storage-related errors."""


class ModelNotFoundError(AIngramError):
    """Raised when a required ML model cannot be found or downloaded."""


class EmbeddingError(AIngramError):
    """Raised for embedding generation or dimension mismatch errors."""


class TrustError(AIngramError):
    """Raised for cryptographic trust failures (signing, verification, chain integrity)."""


class VerificationError(TrustError):
    """Raised when entry or chain verification fails."""


class AuthenticationError(AIngramError):
    """Raised when authentication fails (bad token, revoked, missing)."""


class AuthorizationError(AIngramError):
    """Raised when an authenticated caller lacks permission for the requested action."""


class RateLimitError(AIngramError):
    """Raised when a caller exceeds their rate limit."""

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f'Rate limit exceeded. Retry after {retry_after_seconds:.1f}s')


class InputBoundsError(AIngramError):
    """Raised when input exceeds allowed bounds (content size, title length)."""
