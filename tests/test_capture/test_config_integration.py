import textwrap

from aingram.config import AIngramConfig, load_merged_config


class TestCaptureConfigIntegration:
    def test_default_capture_is_none(self):
        config = AIngramConfig()
        assert config.capture is None

    def test_load_capture_from_toml(self, tmp_path):
        toml_file = tmp_path / 'config.toml'
        toml_file.write_text(
            textwrap.dedent("""\
            [capture]
            enabled = true
            port = 8888
            memory_mode = "isolated"

            [capture.tools.claude_code]
            enabled = true
            container_tag = "my-claude"

            [capture.tools.chatgpt]
            enabled = false
            container_tag = "chatgpt"
        """)
        )

        config = load_merged_config(config_file=toml_file)
        assert config.capture is not None
        assert config.capture.enabled is True
        assert config.capture.port == 8888
        assert config.capture.memory_mode == 'isolated'
        assert config.capture.tools['claude_code'].container_tag == 'my-claude'

    def test_capture_env_var_override(self):
        env = {
            'AINGRAM_CAPTURE_ENABLED': 'true',
            'AINGRAM_CAPTURE_PORT': '9999',
        }
        config = load_merged_config(env=env)
        assert config.capture is not None
        assert config.capture.enabled is True
        assert config.capture.port == 9999

    def test_no_capture_section_leaves_none(self, tmp_path):
        toml_file = tmp_path / 'config.toml'
        toml_file.write_text('log_level = "DEBUG"\n')
        config = load_merged_config(config_file=toml_file)
        assert config.capture is None
