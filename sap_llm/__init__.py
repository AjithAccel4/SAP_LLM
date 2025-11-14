"""
SAP_LLM: 100% Autonomous Document Processing System
Zero 3rd Party LLM Dependencies

This package provides a complete document processing pipeline for SAP integration,
handling all 8 stages from ingestion to routing without external LLM APIs.

Architecture:
- Vision Encoder: LayoutLMv3-base (300M params)
- Language Decoder: LLaMA-2-7B
- Reasoning Engine: Mixtral-8x7B
- Total: 13.8B parameters

Stages:
1. Inbox - Document ingestion & routing
2. Preprocessing - OCR, image enhancement, text extraction
3. Classification - Document type identification
4. Type Identifier - 35+ invoice/PO subtypes
5. Extraction - Field-level data extraction (180+ fields)
6. Data Quality Check - Confidence scoring & validation
7. Validation - Business rules & tolerance checks
8. Routing - SAP API endpoint selection & payload generation
"""

__version__ = "1.0.0"
__author__ = "QorSync AI Team"
__email__ = "ai@qorsync.com"

from sap_llm.config import Config, load_config
from sap_llm.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Module exports
__all__ = [
    "__version__",
    "Config",
    "load_config",
    "get_logger",
]


def get_version() -> str:
    """Get the current version of SAP_LLM."""
    return __version__


def initialize(config_path: str | None = None) -> Config:
    """
    Initialize SAP_LLM with configuration.

    Args:
        config_path: Path to configuration file (YAML). If None, uses default config.

    Returns:
        Config: Loaded configuration object

    Example:
        >>> from sap_llm import initialize
        >>> config = initialize("configs/production.yaml")
        >>> print(config.models.vision_encoder.name)
    """
    logger.info(f"Initializing SAP_LLM v{__version__}")

    # Load configuration
    config = load_config(config_path)

    logger.info(f"Configuration loaded: environment={config.system.environment}")
    logger.info(f"PMG enabled: {config.pmg.enabled}")
    logger.info(f"APOP enabled: {config.apop.enabled}")
    logger.info(f"SHWL enabled: {config.shwl.enabled}")

    return config
