"""Tests for settings loading and validation."""

import pytest
from pydantic import ValidationError

from gateway.config import Settings, get_settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.similarity_threshold == 0.90
    assert settings.batch_timeout_ms == 20
    assert settings.max_batch_size == 8


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MAX_BATCH_SIZE", "16")
    assert Settings().max_batch_size == 16


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(similarity_threshold=1.5)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
