"""
Enterprise Configuration Management for Self-Correction System.

Provides centralized configuration with environment-based overrides,
validation, and secure handling of sensitive settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrectionConfig:
    """Configuration for the self-correction system."""

    # Error Detection Settings
    confidence_threshold_critical: float = 0.85
    confidence_threshold_required: float = 0.75
    confidence_threshold_optional: float = 0.70

    # Correction Engine Settings
    max_correction_attempts: int = 3
    overall_confidence_threshold: float = 0.80
    enable_pattern_learning: bool = True
    enable_human_escalation: bool = True

    # Strategy Settings
    enable_rule_based: bool = True
    enable_rerun_extraction: bool = True
    enable_context_enhancement: bool = True
    enable_human_loop: bool = True

    # Escalation Settings
    escalation_confidence_threshold: float = 0.70
    escalation_max_auto_attempts: int = 3

    # SLA Settings (in hours)
    sla_urgent: int = 2
    sla_high: int = 8
    sla_normal: int = 24
    sla_low: int = 72

    # Analytics Settings
    analytics_retention_days: int = 90
    analytics_export_format: str = "json"

    # Pattern Learning Settings
    pattern_storage_enabled: bool = True
    pattern_storage_path: Optional[str] = None
    pattern_min_confidence: float = 0.80

    # Performance Settings
    correction_timeout_seconds: int = 30
    max_parallel_corrections: int = 10

    # Logging Settings
    log_level: str = "INFO"
    enable_audit_logging: bool = True
    mask_sensitive_fields: bool = True
    sensitive_fields: list = field(default_factory=lambda: [
        "vendor_id", "customer_id", "account_number",
        "tax_id", "ssn", "credit_card"
    ])

    # Monitoring Settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True

    # Retry Settings
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    retry_initial_delay_ms: int = 1000

    # Circuit Breaker Settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_half_open_attempts: int = 1


class ConfigurationManager:
    """
    Manages configuration loading, validation, and environment overrides.

    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Configuration validation
    - Secure handling of sensitive settings
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        logger.info("Configuration loaded successfully")

    def _load_config(self) -> CorrectionConfig:
        """Load configuration from file and environment variables."""
        try:
            # Start with default config
            config_dict = {}

            # Load from file if provided
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config and 'correction' in file_config:
                        config_dict = file_config['correction']
                logger.info(f"Loaded configuration from {self.config_path}")

            # Override with environment variables
            env_overrides = self._load_env_overrides()
            config_dict.update(env_overrides)

            # Create config object
            return CorrectionConfig(**config_dict)

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            logger.warning("Using default configuration")
            return CorrectionConfig()

    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}

        # Map environment variables to config fields
        env_mappings = {
            'CORRECTION_MAX_ATTEMPTS': ('max_correction_attempts', int),
            'CORRECTION_CONFIDENCE_THRESHOLD': ('overall_confidence_threshold', float),
            'CORRECTION_ENABLE_LEARNING': ('enable_pattern_learning', self._parse_bool),
            'CORRECTION_LOG_LEVEL': ('log_level', str),
            'CORRECTION_TIMEOUT_SECONDS': ('correction_timeout_seconds', int),
            'CORRECTION_ENABLE_METRICS': ('enable_metrics', self._parse_bool),
            'CORRECTION_METRICS_PORT': ('metrics_port', int),
        }

        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    overrides[config_key] = converter(value)
                    logger.debug(f"Environment override: {config_key}={value}")
                except ValueError as e:
                    logger.warning(f"Invalid environment variable {env_var}={value}: {e}")

        return overrides

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean value from string."""
        return value.lower() in ('true', '1', 'yes', 'on')

    def _validate_config(self):
        """Validate configuration values."""
        errors = []

        # Validate thresholds (0-1 range)
        if not 0 <= self.config.confidence_threshold_critical <= 1:
            errors.append("confidence_threshold_critical must be between 0 and 1")

        if not 0 <= self.config.overall_confidence_threshold <= 1:
            errors.append("overall_confidence_threshold must be between 0 and 1")

        # Validate positive integers
        if self.config.max_correction_attempts < 1:
            errors.append("max_correction_attempts must be >= 1")

        if self.config.correction_timeout_seconds < 1:
            errors.append("correction_timeout_seconds must be >= 1")

        # Validate SLA times
        if self.config.sla_urgent >= self.config.sla_high:
            errors.append("sla_urgent must be < sla_high")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info("Configuration validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self.config, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }

    def export(self, output_path: str):
        """Export configuration to YAML file."""
        try:
            config_dict = {'correction': self.to_dict()}

            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}", exc_info=True)
            raise


# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """
    Get or create global configuration manager.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigurationManager instance
    """
    global _config_manager

    if _config_manager is None:
        # Try default locations
        default_paths = [
            os.getenv('CORRECTION_CONFIG_PATH'),
            './config/correction.yaml',
            './correction.yaml',
            str(Path.home() / '.sap_llm' / 'correction.yaml')
        ]

        config_file = config_path
        if not config_file:
            for path in default_paths:
                if path and Path(path).exists():
                    config_file = path
                    break

        _config_manager = ConfigurationManager(config_file)

    return _config_manager


def reset_config_manager():
    """Reset global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
