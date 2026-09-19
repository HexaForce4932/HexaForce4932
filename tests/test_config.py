import textwrap

from drone_detector.config import AppConfig, load_config


def test_load_config_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            audio:
              sample_rate: 8000
              channels: 2
              mic_array:
                enabled: true
                mic_spacing_m: 0.06
            dsp:
              harmonicity_threshold: 0.4
            alerts:
              gpio_led:
                enabled: true
                pin: 27
            """
        )
    )

    config = load_config(str(cfg_path))

    assert isinstance(config, AppConfig)
    assert config.audio.sample_rate == 8000
    assert config.audio.channels == 2
    assert config.audio.mic_array.enabled is True
    assert config.audio.mic_array.mic_spacing_m == 0.06
    assert config.dsp.harmonicity_threshold == 0.4
    assert config.alerts.gpio_led.enabled is True
    assert config.alerts.gpio_led.pin == 27

    # Untouched sections should keep their defaults.
    assert config.ml.enabled is True
    assert config.detection.cooldown_seconds == 5.0
    assert config.alerts.webhook.enabled is False


def test_default_app_config_has_sane_defaults():
    config = AppConfig()
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.dsp.fundamental_min_hz < config.dsp.fundamental_max_hz
