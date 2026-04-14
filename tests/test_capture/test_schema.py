import sqlite3

from aingram.capture.schema import CAPTURE_SCHEMA_VERSION, apply_capture_schema


class TestCaptureSchema:
    def test_creates_tables(self, tmp_queue_db):
        conn = sqlite3.connect(tmp_queue_db)
        apply_capture_schema(conn)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert 'capture_queue' in tables
        assert 'toggle_state' in tables
        assert 'capture_schema_version' in tables
        conn.close()

    def test_schema_version_set(self, tmp_queue_db):
        conn = sqlite3.connect(tmp_queue_db)
        apply_capture_schema(conn)
        row = conn.execute('SELECT version FROM capture_schema_version').fetchone()
        assert row[0] == CAPTURE_SCHEMA_VERSION
        conn.close()

    def test_idempotent(self, tmp_queue_db):
        conn = sqlite3.connect(tmp_queue_db)
        apply_capture_schema(conn)
        apply_capture_schema(conn)
        conn.close()
