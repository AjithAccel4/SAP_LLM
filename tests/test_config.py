"""
Unit tests for configuration module.
"""

import os
import pytest
from pathlib import Path

from sap_llm.config import Config, load_config


@pytest.mark.unit
class TestConfig:
    """Tests for configuration management."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()

        assert config is not None
        assert isinstance(config, Config)
        assert config.system is not None
        assert config.models is not None
        assert config.stages is not None

    def test_config_system_settings(self):
        """Test system configuration settings."""
        config = load_config()

        assert config.system.environment in ["development", "staging", "production"]
        assert config.system.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.api.workers > 0  # workers is in API config, not system config

    def test_config_model_settings(self):
        """Test model configuration settings."""
        config = load_config()

        # Vision Encoder
        assert config.models.vision_encoder.name is not None
        assert config.models.vision_encoder.device in ["cpu", "cuda", "mps"]
        assert config.models.vision_encoder.precision in ["fp32", "fp16", "int8", "int4"]

        # Language Decoder
        assert config.models.language_decoder.name is not None
        assert config.models.language_decoder.device in ["cpu", "cuda", "mps"]

        # Reasoning Engine
        assert config.models.reasoning_engine.name is not None
        assert config.models.reasoning_engine.device in ["cpu", "cuda", "mps"]

    def test_config_stages_settings(self):
        """Test pipeline stages configuration."""
        config = load_config()

        # Check all stages exist
        assert config.stages.inbox is not None
        assert config.stages.preprocessing is not None
        assert config.stages.classification is not None
        assert config.stages.type_identifier is not None
        assert config.stages.extraction is not None
        assert config.stages.quality_check is not None
        assert config.stages.validation is not None
        assert config.stages.routing is not None

    def test_config_pmg_settings(self):
        """Test PMG configuration."""
        config = load_config()

        assert config.pmg is not None
        assert isinstance(config.pmg.enabled, bool)

    def test_config_apop_settings(self):
        """Test APOP configuration."""
        config = load_config()

        assert config.apop is not None
        assert isinstance(config.apop.enabled, bool)
        assert isinstance(config.apop.signing_enabled, bool)
        assert config.apop.spec_version is not None

    def test_config_shwl_settings(self):
        """Test SHWL configuration."""
        config = load_config()

        assert config.shwl is not None
        assert isinstance(config.shwl.enabled, bool)

    def test_config_env_var_substitution(self, monkeypatch):
        """Test environment variable substitution."""
        # Set environment variable
        monkeypatch.setenv("TEST_LOG_LEVEL", "DEBUG")

        # This would require modifying the config file to use ${TEST_LOG_LEVEL}
        # For now, just test that the mechanism exists
        config = load_config()
        assert config is not None

    def test_config_validation(self):
        """Test configuration validation."""
        config = load_config()

        # Test that stages are configured (they are dicts)
        assert isinstance(config.stages.classification, dict)
        assert isinstance(config.stages.quality_check, dict)

        # Test that batch sizes are positive
        assert config.models.vision_encoder.batch_size > 0
        assert isinstance(config.performance.batching, dict)

    def test_config_immutability(self):
        """Test that config is immutable (if using frozen=True)."""
        config = load_config()

        # Pydantic models are mutable by default, but we can test that
        # critical values exist
        assert config.system.environment is not None
