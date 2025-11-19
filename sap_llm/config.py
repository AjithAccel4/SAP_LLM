"""
Configuration management for SAP_LLM.

Handles loading, validation, and access to configuration settings from YAML files
and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import logging

import yaml
from pydantic import BaseModel, Field, field_validator, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    """System-level configuration."""

    name: str = "SAP_LLM"
    version: str = "1.0.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = False
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class ModelConfig(BaseModel):
    """Individual model configuration."""

    name: str
    path: str
    parameters: int
    device: str = Field(default="cuda", pattern="^(cuda|cpu|mps)$")
    precision: str = Field(default="fp16", pattern="^(fp32|fp16|int8|int4)$")
    batch_size: int = Field(default=1, ge=1, le=128)
    max_length: Optional[int] = Field(default=None, ge=128, le=8192)


class ModelsConfig(BaseModel):
    """Models configuration."""

    vision_encoder: ModelConfig
    language_decoder: ModelConfig
    reasoning_engine: ModelConfig
    unified_model: Dict[str, Any]


class TrainingConfig(BaseModel):
    """Training configuration."""

    data_dir: str
    train_split: float = Field(default=0.7, ge=0.0, le=1.0)
    val_split: float = Field(default=0.15, ge=0.0, le=1.0)
    test_split: float = Field(default=0.15, ge=0.0, le=1.0)
    num_workers: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=5e-5, ge=1e-7, le=1e-2)
    warmup_steps: int = Field(default=1000, ge=0)
    max_steps: int = Field(default=50000, ge=100)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    mixed_precision: str = Field(default="fp16", pattern="^(no|fp16|bf16)$")
    optimizer: str = Field(default="adamw", pattern="^(adam|adamw|sgd|adagrad)$")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    distributed: bool = False
    world_size: int = Field(default=1, ge=1, le=64)
    backend: str = Field(default="nccl", pattern="^(nccl|gloo|mpi)$")
    checkpoint_dir: str
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)


class StageConfig(BaseModel):
    """Individual pipeline stage configuration."""

    enabled: bool = True


class StagesConfig(BaseModel):
    """All pipeline stages configuration."""

    inbox: Dict[str, Any]
    preprocessing: Dict[str, Any]
    classification: Dict[str, Any]
    type_identifier: Dict[str, Any]
    extraction: Dict[str, Any]
    quality_check: Dict[str, Any]
    validation: Dict[str, Any]
    routing: Dict[str, Any]


class KnowledgeBaseConfig(BaseModel):
    """SAP Knowledge Base configuration."""

    sap_api_count: int = Field(default=400, ge=1)
    vector_store: Dict[str, Any]
    field_mappings_path: str
    business_rules_path: str
    transformation_functions_path: str


class PMGConfig(BaseModel):
    """Process Memory Graph configuration."""

    enabled: bool = True
    backend: str = Field(default="cosmos_gremlin", pattern="^(cosmos_gremlin|neo4j)$")
    cosmos: Dict[str, Any]
    neo4j: Dict[str, Any]
    max_similar_docs: int = Field(default=20, ge=1, le=100)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    embedding_dimension: int = Field(default=768, ge=128, le=2048)
    continuous_learning: Dict[str, Any]


class APOPConfig(BaseModel):
    """Agentic Process Orchestration Protocol configuration."""

    enabled: bool = True
    spec_version: str = "1.0"
    source: str = "sap_llm"
    content_type: str = "application/json"
    signing_enabled: bool = True
    algorithm: str = "ECDSA"
    private_key_path: str
    service_bus: Dict[str, Any]
    agents: List[Dict[str, Any]]


class SHWLConfig(BaseModel):
    """Self-Healing Workflow Loop configuration."""

    enabled: bool = True
    clustering: Dict[str, Any]
    rule_generation: Dict[str, Any]
    approval: Dict[str, Any]
    schedule: Dict[str, Any]


class DatabasesConfig(BaseModel):
    """Databases configuration."""

    redis: Dict[str, Any]
    mongodb: Dict[str, Any]


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    reload: bool = False
    cors: Dict[str, Any]
    rate_limit: Dict[str, Any]
    auth: Dict[str, Any]


class CORSSettings(BaseSettings):
    """
    CORS (Cross-Origin Resource Sharing) Configuration.

    Security-first CORS configuration with validation to prevent common vulnerabilities.
    Loads from environment variables with proper defaults.
    """

    # SECURITY: Default to empty string (deny all cross-origin requests) for maximum security
    # Must be explicitly configured for production use
    CORS_ALLOWED_ORIGINS: str = Field(
        default="",
        description="Comma-separated list of allowed origins for CORS. Empty string to deny all cross-origin requests."
    )

    # Load environment from system settings
    ENVIRONMENT: str = Field(default="development")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }

    def __init__(self, **data):
        """Initialize and parse CORS origins."""
        super().__init__(**data)
        # Parse and validate origins after initialization
        self._validated_origins = self._parse_and_validate_origins()

    def _parse_and_validate_origins(self) -> List[str]:
        """
        Parse and validate CORS origins for security compliance.

        Security checks:
        1. No wildcard (*) in production
        2. Valid URL format for all origins
        3. HTTPS requirement in production
        4. Warning for too many origins (>5)

        Returns:
            List[str]: Validated list of origins
        """
        # Parse comma-separated string
        if not self.CORS_ALLOWED_ORIGINS.strip():
            return []

        origins = [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

        if not origins:
            return []

        # SECURITY CHECK 1: No wildcards in production
        if "*" in origins and self.ENVIRONMENT == "production":
            raise ValueError(
                "SECURITY VIOLATION: CORS wildcard (*) is not allowed in production. "
                "Specify explicit allowed origins in CORS_ALLOWED_ORIGINS environment variable. "
                "Example: CORS_ALLOWED_ORIGINS=https://app.example.com,https://api.example.com"
            )

        # SECURITY CHECK 2: Validate URL format for non-wildcard origins
        validated_origins = []
        for origin in origins:
            if origin == "*":
                # Wildcard allowed in non-production
                if self.ENVIRONMENT != "production":
                    logger.warning(
                        "SECURITY WARNING: CORS wildcard (*) detected in %s environment. "
                        "This allows ANY domain to access your API. Use explicit origins in production.",
                        self.ENVIRONMENT
                    )
                    validated_origins.append(origin)
                continue

            # Parse and validate URL
            try:
                parsed = urlparse(origin)

                # Must have scheme and netloc
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(
                        f"Invalid CORS origin '{origin}': Must be a complete URL with scheme "
                        f"(e.g., https://example.com, not example.com)"
                    )

                # SECURITY CHECK 3: HTTPS requirement in production
                if self.ENVIRONMENT == "production" and parsed.scheme != "https":
                    raise ValueError(
                        f"SECURITY VIOLATION: CORS origin '{origin}' must use HTTPS in production. "
                        f"HTTP origins are not secure and will expose your API to man-in-the-middle attacks."
                    )

                # Normalize: remove trailing slash for consistency
                normalized_origin = f"{parsed.scheme}://{parsed.netloc}"

                validated_origins.append(normalized_origin)

            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Invalid CORS origin '{origin}': {str(e)}"
                )

        # SECURITY CHECK 4: Warn about too many origins (potential misconfiguration)
        if len(validated_origins) > 5:
            logger.warning(
                "SECURITY WARNING: %d CORS origins configured. "
                "Large numbers of allowed origins may indicate misconfiguration. "
                "Consider using a more restrictive CORS policy or implementing dynamic origin validation.",
                len(validated_origins)
            )

        return validated_origins

    def get_origins(self) -> List[str]:
        """Get the list of validated origins."""
        return self._validated_origins

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def validate_for_production(self) -> None:
        """
        Validate CORS configuration for production deployment.

        Raises:
            ValueError: If configuration is not production-ready
        """
        if not self.is_production():
            return

        origins = self.get_origins()

        if not origins:
            raise ValueError(
                "SECURITY VIOLATION: CORS_ALLOWED_ORIGINS cannot be empty in production. "
                "You must specify at least one allowed origin. "
                "Set the CORS_ALLOWED_ORIGINS environment variable with your frontend domain(s)."
            )

        if "*" in origins:
            raise ValueError(
                "SECURITY VIOLATION: CORS wildcard (*) is not allowed in production."
            )

        # Verify all origins are HTTPS
        for origin in origins:
            if not origin.startswith("https://"):
                raise ValueError(
                    f"SECURITY VIOLATION: All CORS origins must use HTTPS in production. "
                    f"Found: {origin}"
                )


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    cache: Dict[str, Any]
    batching: Dict[str, Any]
    optimization: Dict[str, Any]


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enabled: bool = True
    prometheus: Dict[str, Any]
    opentelemetry: Dict[str, Any]
    logging: Dict[str, Any]


class TestingConfig(BaseModel):
    """Testing configuration."""

    test_data_path: str
    benchmark_path: str
    thresholds: Dict[str, float]


class WebSearchConfig(BaseModel):
    """Web search configuration."""

    enabled: bool = True
    offline_mode: bool = False
    cache_enabled: bool = True
    providers: Dict[str, Any]
    provider_priority: List[str]
    rate_limits: Dict[str, int]
    cache: Dict[str, Any]
    trusted_domains: List[str]
    blocked_domains: List[str]
    min_relevance_score: float = Field(default=0.3, ge=0.0, le=1.0)
    features: Dict[str, bool]
    integrations: Dict[str, Any]


class Config(BaseModel):
    """Main configuration object."""

    system: SystemConfig
    models: ModelsConfig
    training: TrainingConfig
    stages: StagesConfig
    knowledge_base: KnowledgeBaseConfig
    pmg: PMGConfig
    apop: APOPConfig
    shwl: SHWLConfig
    databases: DatabasesConfig
    api: APIConfig
    performance: PerformanceConfig
    monitoring: MonitoringConfig
    testing: TestingConfig
    document_types: List[str]
    languages: List[str]
    web_search: Optional[WebSearchConfig] = None

    @field_validator("system")
    def validate_splits(cls, v: SystemConfig, info) -> SystemConfig:
        """Validate that data splits sum to 1.0."""
        if hasattr(info, "data") and "training" in info.data:
            training = info.data["training"]
            total = training.train_split + training.val_split + training.test_split
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Train/val/test splits must sum to 1.0, got {total:.2f}"
                )
        return v


def load_config(config_path: str | None = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to YAML configuration file. If None, uses default config.

    Returns:
        Config: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Determine config path
    if config_path is None:
        # Use default config
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "configs" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Substitute environment variables
    config_dict = _substitute_env_vars(config_dict)

    # Validate and create Config object
    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    return config


def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in configuration.

    Supports syntax: ${VAR_NAME} or ${VAR_NAME:-default_value}

    Args:
        config_dict: Configuration dictionary

    Returns:
        Dictionary with environment variables substituted
    """
    import re

    def substitute_value(value: Any) -> Any:
        if isinstance(value, str):
            # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
            pattern = r'\$\{([^:}]+)(?::-(.*?))?\}'

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2)
                return os.environ.get(var_name, default_value or "")

            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    return substitute_value(config_dict)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = config.model_dump()

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
