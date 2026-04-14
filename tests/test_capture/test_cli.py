from typer.testing import CliRunner

from aingram.cli import app

runner = CliRunner()


class TestCaptureCliHelp:
    def test_capture_help(self):
        result = runner.invoke(app, ['capture', '--help'])
        assert result.exit_code == 0
        assert 'start' in result.output
        assert 'stop' in result.output
        assert 'status' in result.output

    def test_capture_install_claude(self):
        result = runner.invoke(app, ['capture', 'install', 'claude_code'])
        assert result.exit_code == 0
        assert 'localhost:7749' in result.output

    def test_capture_install_unknown_tool(self):
        result = runner.invoke(app, ['capture', 'install', 'nonexistent'])
        assert result.exit_code != 0
