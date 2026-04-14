import pytest

from aingram.capture.config import CaptureConfig


@pytest.fixture
def tmp_queue_db(tmp_path):
    return str(tmp_path / 'test_capture_queue.db')


@pytest.fixture
def capture_config():
    return CaptureConfig()
