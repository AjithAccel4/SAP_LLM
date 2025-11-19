"""
Field Mapping Manager

Manages SAP field mappings for document transformations.
Loads mappings from JSON configuration files and provides
transformation and validation services.
"""

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class FieldMappingManager:
    """
    Manages field mappings for SAP document transformations.

    Features:
    - Loads mappings from JSON configuration files
    - Caches mappings in memory for performance
    - Applies field transformations and validations
    - Supports nested object and array mappings
    - Handles up to 5 levels of nesting
    """

    # ISO 4217 currency codes (common subset)
    VALID_CURRENCIES = {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "SEK", "NOK", "DKK", "SGD", "HKD", "KRW", "CNY", "INR",
        "BRL", "MXN", "ZAR", "RUB", "TRY", "PLN", "THB", "IDR"
    }

    def __init__(self, mappings_dir: Optional[str] = None):
        """
        Initialize Field Mapping Manager.

        Args:
            mappings_dir: Directory containing mapping JSON files.
                         Defaults to data/field_mappings/
        """
        if mappings_dir is None:
            # Default to data/field_mappings relative to project root
            project_root = Path(__file__).parent.parent.parent
            mappings_dir = project_root / "data" / "field_mappings"

        self.mappings_dir = Path(mappings_dir)
        self._mappings_cache: Dict[str, Dict[str, Any]] = {}
        self._load_all_mappings()

        logger.info(f"Field Mapping Manager initialized with {len(self._mappings_cache)} mappings")

    def _load_all_mappings(self) -> None:
        """Load all mapping files from the mappings directory."""
        if not self.mappings_dir.exists():
            logger.warning(f"Mappings directory not found: {self.mappings_dir}")
            return

        for mapping_file in self.mappings_dir.glob("*.json"):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)

                # Create cache key from document_type and subtype
                doc_type = mapping_data.get("document_type", "")
                subtype = mapping_data.get("subtype", "Standard")
                cache_key = self._get_cache_key(doc_type, subtype)

                self._mappings_cache[cache_key] = mapping_data
                logger.debug(f"Loaded mapping: {mapping_file.name} -> {cache_key}")

            except Exception as e:
                logger.error(f"Failed to load mapping file {mapping_file}: {e}")

    @staticmethod
    def _get_cache_key(document_type: str, subtype: str = "Standard") -> str:
        """
        Generate cache key for a mapping.

        Args:
            document_type: Document type (e.g., "PurchaseOrder")
            subtype: Document subtype (e.g., "Standard", "Service")

        Returns:
            Cache key string
        """
        return f"{document_type}:{subtype}".upper()

    @lru_cache(maxsize=128)
    def get_mapping(
        self,
        document_type: str,
        subtype: str = "Standard",
        target_format: str = "SAP_API"
    ) -> Optional[Dict[str, Any]]:
        """
        Get field mapping for a document type.

        Args:
            document_type: Document type (e.g., "PurchaseOrder", "SupplierInvoice")
            subtype: Document subtype (e.g., "Standard", "Service")
            target_format: Target format (currently unused, for future extension)

        Returns:
            Mapping dictionary or None if not found
        """
        cache_key = self._get_cache_key(document_type, subtype)
        mapping = self._mappings_cache.get(cache_key)

        if not mapping:
            logger.warning(f"No mapping found for {cache_key}")
            return None

        return mapping

    def validate_mapping(
        self,
        data: Dict[str, Any],
        mapping: Dict[str, Any],
        strict: Optional[bool] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against a mapping configuration.

        Args:
            data: Data to validate
            mapping: Mapping configuration
            strict: Strict validation mode (uses mapping config if not specified)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Get config settings
        config = mapping.get("config", {})
        if strict is None:
            strict = config.get("strict_validation", False)

        field_mappings = mapping.get("mappings", {})

        # Check required fields
        for source_field, field_config in field_mappings.items():
            if isinstance(field_config, dict) and field_config.get("required", False):
                sap_field = field_config.get("sap_field", source_field)

                if source_field not in data:
                    errors.append(f"Required field missing: {source_field} (SAP: {sap_field})")

        # Validate field values
        for source_field, value in data.items():
            if source_field in field_mappings:
                field_config = field_mappings[source_field]

                if not isinstance(field_config, dict):
                    continue

                # Validate against regex pattern
                validation_pattern = field_config.get("validation")
                if validation_pattern and value is not None:
                    if not re.match(validation_pattern, str(value)):
                        errors.append(
                            f"Field '{source_field}' value '{value}' "
                            f"does not match pattern '{validation_pattern}'"
                        )

                # Validate max length
                max_length = field_config.get("max_length")
                if max_length and value is not None:
                    if len(str(value)) > max_length:
                        errors.append(
                            f"Field '{source_field}' exceeds max length {max_length}: "
                            f"'{value}' (length: {len(str(value))})"
                        )

        is_valid = len(errors) == 0 or not strict

        if errors:
            logger.warning(f"Validation errors: {errors}")

        return is_valid, errors

    def apply_transformations(
        self,
        value: Any,
        transformations: List[str],
        field_name: str = ""
    ) -> Any:
        """
        Apply transformation functions to a value.

        Supported transformations:
        - uppercase, lowercase, trim
        - parse_date, format_date:FORMAT
        - parse_amount, format_decimal:PLACES
        - pad_left:LENGTH:CHAR, pad_right:LENGTH:CHAR
        - validate_iso_currency
        - parse_integer
        - negate

        Args:
            value: Value to transform
            transformations: List of transformation strings
            field_name: Field name (for error reporting)

        Returns:
            Transformed value
        """
        result = value

        for transformation in transformations:
            try:
                result = self._apply_single_transformation(result, transformation)

            except Exception as e:
                logger.error(
                    f"Transformation '{transformation}' failed for field '{field_name}' "
                    f"with value '{value}': {e}"
                )
                # Return original value on error
                return value

        return result

    def _apply_single_transformation(self, value: Any, transformation: str) -> Any:
        """Apply a single transformation to a value."""
        # String transformations
        if transformation == "uppercase":
            return str(value).upper()

        elif transformation == "lowercase":
            return str(value).lower()

        elif transformation == "trim":
            return str(value).strip()

        # Date transformations
        elif transformation == "parse_date":
            return self._parse_date_value(value)

        elif transformation.startswith("format_date:"):
            date_format = transformation.split(":", 1)[1]
            return self._format_date_value(value, date_format)

        # Number transformations
        elif transformation == "parse_amount":
            return self._parse_amount_value(value)

        elif transformation == "parse_integer":
            return self._parse_integer_value(value)

        elif transformation.startswith("format_decimal:"):
            places = int(transformation.split(":", 1)[1])
            return round(float(value), places)

        elif transformation == "negate":
            return -float(value)

        # Padding transformations
        elif transformation.startswith("pad_left:"):
            parts = transformation.split(":")
            length = int(parts[1])
            char = parts[2] if len(parts) > 2 else "0"
            return str(value).rjust(length, char)

        elif transformation.startswith("pad_right:"):
            parts = transformation.split(":")
            length = int(parts[1])
            char = parts[2] if len(parts) > 2 else " "
            return str(value).ljust(length, char)

        # Currency validation
        elif transformation == "validate_iso_currency":
            return self._validate_currency_value(value)

        else:
            logger.warning(f"Unknown transformation: {transformation}")
            return value

    def _parse_date_value(self, date_value: Any) -> datetime:
        """
        Parse date from various formats.

        Args:
            date_value: Date value (string, datetime, or timestamp)

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date cannot be parsed
        """
        if isinstance(date_value, datetime):
            return date_value

        # Try common date formats
        date_str = str(date_value).strip()

        date_formats = [
            "%Y-%m-%d",           # ISO format: 2024-01-15
            "%d/%m/%Y",           # European: 15/01/2024
            "%m/%d/%Y",           # US: 01/15/2024
            "%Y%m%d",             # SAP format: 20240115
            "%d.%m.%Y",           # German: 15.01.2024
            "%Y-%m-%dT%H:%M:%S",  # ISO with time
            "%Y-%m-%d %H:%M:%S",  # SQL timestamp
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all formats fail, raise error
        raise ValueError(f"Cannot parse date: {date_value}")

    def _format_date_value(self, date_value: Any, format_str: str) -> str:
        """
        Format date to specified format.

        Args:
            date_value: Date value (datetime or string)
            format_str: Target format (YYYYMMDD, YYYY-MM-DD, DD/MM/YYYY, or strftime format)

        Returns:
            Formatted date string
        """
        if isinstance(date_value, str):
            date_value = self._parse_date_value(date_value)

        if format_str == "YYYYMMDD":
            return date_value.strftime("%Y%m%d")
        elif format_str == "YYYY-MM-DD":
            return date_value.strftime("%Y-%m-%d")
        elif format_str == "DD/MM/YYYY":
            return date_value.strftime("%d/%m/%Y")
        elif format_str == "MM/DD/YYYY":
            return date_value.strftime("%m/%d/%Y")
        else:
            # Custom format string
            return date_value.strftime(format_str)

    def _parse_amount_value(self, amount_value: Any) -> float:
        """
        Parse amount from various formats.

        Handles currency symbols, thousand separators, etc.

        Args:
            amount_value: Amount value (string, int, or float)

        Returns:
            Parsed float value
        """
        if isinstance(amount_value, (int, float)):
            return float(amount_value)

        # Remove common currency symbols and formatting
        amount_str = str(amount_value).strip()

        # Remove currency symbols
        amount_str = re.sub(r'[$€£¥₹]', '', amount_str)

        # Remove thousand separators (commas, spaces)
        amount_str = amount_str.replace(',', '').replace(' ', '')

        # Handle parentheses for negative numbers
        if amount_str.startswith('(') and amount_str.endswith(')'):
            amount_str = '-' + amount_str[1:-1]

        try:
            return float(amount_str)
        except ValueError:
            raise ValueError(f"Cannot parse amount: {amount_value}")

    def _parse_integer_value(self, int_value: Any) -> int:
        """
        Parse integer from various formats.

        Args:
            int_value: Integer value (string or number)

        Returns:
            Parsed integer value
        """
        if isinstance(int_value, int):
            return int_value

        # Remove formatting
        int_str = str(int_value).strip()
        int_str = int_str.replace(',', '').replace(' ', '')

        try:
            return int(float(int_str))  # Use float() to handle "10.0" -> 10
        except ValueError:
            raise ValueError(f"Cannot parse integer: {int_value}")

    def _validate_currency_value(self, currency_value: Any) -> str:
        """
        Validate and normalize currency code.

        Args:
            currency_value: Currency code

        Returns:
            Validated currency code (uppercase)

        Raises:
            ValueError: If currency code is invalid
        """
        currency = str(currency_value).strip().upper()

        if currency not in self.VALID_CURRENCIES:
            logger.warning(f"Invalid or uncommon currency code: {currency}")
            # Don't raise error, just warn - allow uncommon currencies

        return currency

    def transform_data(
        self,
        source_data: Dict[str, Any],
        mapping: Dict[str, Any],
        max_nesting_level: int = 5
    ) -> Dict[str, Any]:
        """
        Transform data using a mapping configuration.

        Supports nested objects and arrays up to specified depth.

        Args:
            source_data: Source data dictionary
            mapping: Mapping configuration
            max_nesting_level: Maximum nesting depth to process

        Returns:
            Transformed data dictionary
        """
        target_data = {}

        # Get config settings
        config = mapping.get("config", {})
        field_mappings = mapping.get("mappings", {})
        nested_mappings = mapping.get("nested_mappings", {})

        # Apply field mappings
        for source_field, field_config in field_mappings.items():
            if source_field not in source_data:
                # Check for default value
                if isinstance(field_config, dict):
                    default = field_config.get("default")
                    if default is not None:
                        target_field = field_config.get("sap_field", source_field)
                        target_data[target_field] = default
                continue

            source_value = source_data[source_field]

            # Handle both simple string mappings and dict configs
            if isinstance(field_config, str):
                # Simple field name mapping
                target_field = field_config
                transformations = []
            else:
                # Full config with transformations
                target_field = field_config.get("sap_field", source_field)
                transformations = field_config.get("transformations", [])

            # Apply transformations
            transformed_value = self.apply_transformations(
                source_value,
                transformations,
                source_field
            )

            target_data[target_field] = transformed_value

        # Handle nested mappings (arrays and objects)
        if max_nesting_level > 0:
            for nested_field, nested_config in nested_mappings.items():
                if nested_field not in source_data:
                    continue

                nested_source = source_data[nested_field]
                sap_collection = nested_config.get("sap_collection", nested_field)
                nested_field_mappings = nested_config.get("mappings", {})

                # Create a temporary mapping for nested transformation
                nested_mapping = {
                    "config": config,
                    "mappings": nested_field_mappings
                }

                # Handle arrays
                if isinstance(nested_source, list):
                    target_data[sap_collection] = [
                        self.transform_data(
                            item,
                            nested_mapping,
                            max_nesting_level - 1
                        )
                        for item in nested_source
                    ]
                # Handle single objects
                elif isinstance(nested_source, dict):
                    target_data[sap_collection] = self.transform_data(
                        nested_source,
                        nested_mapping,
                        max_nesting_level - 1
                    )

        # Copy unmapped fields if configured
        if config.get("copy_unmapped", False):
            for key, value in source_data.items():
                if key not in field_mappings and key not in target_data:
                    target_data[key] = value

        return target_data

    def get_document_types(self) -> List[str]:
        """
        Get list of available document types.

        Returns:
            List of document type names
        """
        doc_types = set()
        for mapping in self._mappings_cache.values():
            doc_type = mapping.get("document_type")
            if doc_type:
                doc_types.add(doc_type)
        return sorted(list(doc_types))

    def get_subtypes(self, document_type: str) -> List[str]:
        """
        Get list of subtypes for a document type.

        Args:
            document_type: Document type name

        Returns:
            List of subtype names
        """
        subtypes = []
        for mapping in self._mappings_cache.values():
            if mapping.get("document_type") == document_type:
                subtype = mapping.get("subtype", "Standard")
                subtypes.append(subtype)
        return sorted(subtypes)
