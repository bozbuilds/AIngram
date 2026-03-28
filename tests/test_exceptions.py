# tests/test_exceptions.py
from aingram.exceptions import AIngramError, DatabaseError, EmbeddingError, ModelNotFoundError


def test_database_error_is_aingram_error():
    err = DatabaseError('db locked')
    assert isinstance(err, AIngramError)
    assert str(err) == 'db locked'


def test_model_not_found_error_is_aingram_error():
    err = ModelNotFoundError('nomic-embed-text')
    assert isinstance(err, AIngramError)


def test_embedding_error_is_aingram_error():
    err = EmbeddingError('dimension mismatch')
    assert isinstance(err, AIngramError)


def test_exception_hierarchy_catchable_as_base():
    for exc_class in (DatabaseError, ModelNotFoundError, EmbeddingError):
        try:
            raise exc_class('test')
        except AIngramError:
            pass  # expected
