"""
Advanced configuration tests for edge cases and error handling.
"""

import pytest
from pathlib import Path

from sap_llm.config import Config, load_config, save_config


@pytest.mark.unit
class TestConfigAdvanced:
    """Advanced configuration tests."""

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_config_structure(self, temp_dir):
        """Test that invalid config structure raises ValueError."""
        # Create config with missing required fields
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("system:\n  incomplete: true\n")

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config(str(invalid_config))

    def test_save_config(self, temp_dir):
        """Test saving configuration to file."""
        config = load_config()
        output_file = temp_dir / "output" / "saved_config.yaml"

        # Save config (should create parent directories)
        save_config(config, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Load saved config and verify
        loaded_config = load_config(str(output_file))
        assert loaded_config.system.name == config.system.name
        assert loaded_config.system.version == config.system.version

    def test_config_model_dump(self):
        """Test config can be converted to dict."""
        config = load_config()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "system" in config_dict
        assert "models" in config_dict
        assert "stages" in config_dict

    def test_config_env_var_with_default(self, temp_dir, monkeypatch):
        """Test environment variable substitution with default value."""
        config_content = """
system:
  name: ${APP_NAME:-SAP_LLM}
  version: "1.0.0"
  environment: development
  debug: false
  log_level: ${LOG_LEVEL:-INFO}
models:
  vision_encoder:
    name: "microsoft/layoutlmv3-base"
    path: "models/vision_encoder"
    parameters: 300000000
    device: "cpu"
    precision: "fp16"
    batch_size: 1
  language_decoder:
    name: "meta-llama/Llama-2-7b-hf"
    path: "models/language_decoder"
    parameters: 7000000000
    device: "cpu"
    precision: "fp16"
    batch_size: 1
  reasoning_engine:
    name: "mistralai/Mixtral-8x7B-v0.1"
    path: "models/reasoning_engine"
    parameters: 46700000000
    device: "cpu"
    precision: "fp16"
    batch_size: 1
  unified_model:
    enabled: true
training:
  data_dir: "data/training"
  checkpoint_dir: "checkpoints"
stages:
  inbox: {}
  preprocessing: {}
  classification: {}
  type_identifier: {}
  extraction: {}
  quality_check: {}
  validation: {}
  routing: {}
knowledge_base:
  sap_api_count: 400
  vector_store: {}
  field_mappings_path: "config/field_mappings.json"
  business_rules_path: "config/business_rules.json"
  transformation_functions_path: "config/transformations.py"
pmg:
  enabled: true
  backend: "cosmos_gremlin"
  cosmos: {}
  neo4j: {}
  continuous_learning: {}
apop:
  enabled: true
  spec_version: "1.0"
  source: "sap_llm"
  content_type: "application/json"
  signing_enabled: true
  algorithm: "ECDSA"
  private_key_path: "keys/private.pem"
  service_bus: {}
  agents: []
shwl:
  enabled: true
  clustering: {}
  rule_generation: {}
  approval: {}
  schedule: {}
databases:
  redis: {}
  mongodb: {}
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors: {}
  rate_limit: {}
  auth: {}
performance:
  cache: {}
  batching: {}
  optimization: {}
monitoring:
  enabled: true
  prometheus: {}
  opentelemetry: {}
  logging: {}
testing:
  test_data_path: "data/test"
  benchmark_path: "data/benchmarks"
  thresholds: {}
document_types: ["invoice"]
languages: ["en"]
"""
        config_file = temp_dir / "env_config.yaml"
        config_file.write_text(config_content)

        # Test with default values (no env vars set)
        config = load_config(str(config_file))
        assert config.system.name == "SAP_LLM"
        assert config.system.log_level == "INFO"

        # Test with env vars set
        monkeypatch.setenv("APP_NAME", "TEST_APP")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = load_config(str(config_file))
        assert config.system.name == "TEST_APP"
        assert config.system.log_level == "DEBUG"

    def test_config_validation_constraints(self):
        """Test configuration validation constraints."""
        config = load_config()

        # Test system constraints
        assert config.system.environment in ["development", "staging", "production"]
        assert config.system.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Test model constraints
        assert config.models.vision_encoder.device in ["cpu", "cuda", "mps"]
        assert config.models.vision_encoder.precision in ["fp32", "fp16", "int8", "int4"]
        assert 1 <= config.models.vision_encoder.batch_size <= 128

        # Test training constraints
        assert 0.0 <= config.training.train_split <= 1.0
        assert 0.0 <= config.training.val_split <= 1.0
        assert 0.0 <= config.training.test_split <= 1.0
        assert config.training.num_workers >= 1
        assert config.training.learning_rate > 0

        # Test API constraints
        assert 1024 <= config.api.port <= 65535
        assert 1 <= config.api.workers <= 32

    def test_config_pmg_constraints(self):
        """Test PMG configuration constraints."""
        config = load_config()

        assert config.pmg.backend in ["cosmos_gremlin", "neo4j"]
        assert 1 <= config.pmg.max_similar_docs <= 100
        assert 0.0 <= config.pmg.similarity_threshold <= 1.0
        assert 128 <= config.pmg.embedding_dimension <= 2048

    def test_config_web_search_optional(self):
        """Test that web_search config is optional."""
        config = load_config()

        # web_search is optional, may be None or a WebSearchConfig
        if config.web_search is not None:
            assert hasattr(config.web_search, 'enabled')
            assert isinstance(config.web_search.enabled, bool)
