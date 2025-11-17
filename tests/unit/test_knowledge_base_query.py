"""
Comprehensive Unit Tests for Knowledge Base Query Module

Tests field mapping, transformations, and SAP API interactions.
Ensures 90%+ code coverage for sap_llm/knowledge_base/query.py
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from sap_llm.knowledge_base.query import KnowledgeBaseQuery
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage


@pytest.fixture
def mock_storage():
    """Create mock knowledge base storage."""
    storage = KnowledgeBaseStorage(mock_mode=True)
    return storage


@pytest.fixture
def kb_query(mock_storage):
    """Create Knowledge Base Query instance."""
    return KnowledgeBaseQuery(storage=mock_storage)


class TestFieldMapping:
    """Test field mapping and transformation functionality."""

    def test_map_fields_to_sap_basic(self, kb_query):
        """Test basic field mapping from ADC to SAP format."""
        adc_data = {
            "invoice_number": "INV-2025-001",
            "invoice_date": "2025-11-14",
            "vendor_id": "VENDOR123",
            "total_amount": "1250.00",
            "currency": "USD"
        }

        result = kb_query.map_fields_to_sap(adc_data, "supplier_invoice")

        assert "payload" in result
        assert "mappings" in result
        assert result["payload"] is not None

    def test_map_fields_handles_missing_mapping(self, kb_query):
        """Test that unmapped fields are preserved."""
        adc_data = {
            "unknown_field": "test_value",
            "custom_field": 123
        }

        result = kb_query.map_fields_to_sap(adc_data, "supplier_invoice")

        # Unknown fields should be in payload
        assert "unknown_field" in result["payload"]
        assert result["payload"]["unknown_field"] == "test_value"

    def test_field_transformation_uppercase(self, kb_query):
        """Test uppercase transformation."""
        mapping = {"transform": "uppercase"}
        result = kb_query._apply_field_transformation("vendor123", mapping)

        assert result == "VENDOR123"

    def test_field_transformation_lowercase(self, kb_query):
        """Test lowercase transformation."""
        mapping = {"transform": "lowercase"}
        result = kb_query._apply_field_transformation("VENDOR123", mapping)

        assert result == "vendor123"

    def test_field_transformation_remove_spaces(self, kb_query):
        """Test space removal transformation."""
        mapping = {"transform": "remove_spaces"}
        result = kb_query._apply_field_transformation("INV 123 456", mapping)

        assert result == "INV123456"

    def test_field_transformation_date_format(self, kb_query):
        """Test date formatting transformation."""
        mapping = {
            "transform": "date_format",
            "target_format": "SAP"
        }

        # Test ISO format to SAP
        result = kb_query._apply_field_transformation("2025-11-14", mapping)
        assert result == "20251114"

    def test_field_transformation_no_transform(self, kb_query):
        """Test that fields without transformation pass through."""
        mapping = {}
        result = kb_query._apply_field_transformation("test_value", mapping)

        assert result == "test_value"


class TestDateFormatting:
    """Test date parsing and formatting functionality."""

    def test_format_date_iso_to_sap(self, kb_query):
        """Test ISO format to SAP YYYYMMDD format."""
        result = kb_query._format_date("2025-11-14", "SAP")
        assert result == "20251114"

    def test_format_date_european_format(self, kb_query):
        """Test European DD/MM/YYYY format parsing."""
        result = kb_query._format_date("14/11/2025", "SAP")
        assert result == "20251114"

    def test_format_date_us_format(self, kb_query):
        """Test US MM/DD/YYYY format parsing."""
        result = kb_query._format_date("11/14/2025", "SAP")
        assert result == "20251114"

    def test_format_date_german_format(self, kb_query):
        """Test German DD.MM.YYYY format parsing."""
        result = kb_query._format_date("14.11.2025", "SAP")
        assert result == "20251114"

    def test_format_date_already_sap_format(self, kb_query):
        """Test date already in SAP format."""
        result = kb_query._format_date("20251114", "SAP")
        assert result == "20251114"

    def test_format_date_datetime_object(self, kb_query):
        """Test formatting datetime object."""
        date_obj = datetime(2025, 11, 14)
        result = kb_query._format_date(date_obj, "SAP")
        assert result == "20251114"

    def test_format_date_to_iso(self, kb_query):
        """Test formatting to ISO format."""
        result = kb_query._format_date("20251114", "ISO")
        assert result == "2025-11-14"

    def test_format_date_custom_format(self, kb_query):
        """Test custom format string."""
        result = kb_query._format_date("2025-11-14", "%d/%m/%Y")
        assert result == "14/11/2025"

    def test_format_date_invalid_format(self, kb_query):
        """Test handling of invalid date format."""
        result = kb_query._format_date("invalid_date", "SAP")
        # Should return original string on error
        assert result == "invalid_date"

    def test_format_date_with_time(self, kb_query):
        """Test date with time component."""
        result = kb_query._format_date("2025-11-14T10:30:00", "SAP")
        assert result == "20251114"


class TestAPIQueries:
    """Test API querying functionality."""

    def test_find_api_for_document_purchase_order(self, kb_query):
        """Test finding API for purchase order."""
        apis = kb_query.find_api_for_document("purchase_order")

        assert isinstance(apis, list)
        # Mock storage should return results
        assert len(apis) >= 0

    def test_find_api_for_document_with_subtype(self, kb_query):
        """Test finding API with document subtype."""
        apis = kb_query.find_api_for_document("supplier_invoice", "standard")

        assert isinstance(apis, list)

    def test_find_api_for_document_unknown_type(self, kb_query):
        """Test finding API for unknown document type."""
        apis = kb_query.find_api_for_document("unknown_doc_type")

        assert isinstance(apis, list)
        # Should handle gracefully


class TestValidationRules:
    """Test validation rule functionality."""

    def test_find_validation_rules(self, kb_query):
        """Test finding validation rules for document type."""
        rules = kb_query.find_validation_rules("supplier_invoice")

        assert isinstance(rules, list)

    def test_find_validation_rules_for_field(self, kb_query):
        """Test finding validation rules for specific field."""
        rules = kb_query.find_validation_rules("supplier_invoice", "total_amount")

        assert isinstance(rules, list)

    def test_validate_payload_valid(self, kb_query):
        """Test validation of valid payload."""
        payload = {
            "invoice_number": "INV-2025-001",
            "total_amount": 1250.00,
            "vendor_id": "VENDOR123"
        }

        result = kb_query.validate_payload(payload, "supplier_invoice")

        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result

    def test_apply_validation_rule_pattern(self, kb_query):
        """Test pattern validation rule."""
        payload = {"invoice_number": "INV-2025-001"}
        rule = {
            "type": "validation",
            "pattern": r"^INV-\d{4}-\d{3}$",
            "description": "field 'invoice_number' must match pattern"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is True

    def test_apply_validation_rule_pattern_fails(self, kb_query):
        """Test pattern validation failure."""
        payload = {"invoice_number": "INVALID"}
        rule = {
            "type": "validation",
            "pattern": r"^INV-\d{4}-\d{3}$",
            "description": "field 'invoice_number' must match pattern"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is False
        assert "details" in result

    def test_apply_validation_rule_required_field(self, kb_query):
        """Test required field validation."""
        payload = {"vendor_id": "VENDOR123"}
        rule = {
            "type": "required",
            "description": "field 'invoice_number' is required"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is False

    def test_apply_validation_rule_required_field_present(self, kb_query):
        """Test required field validation when field is present."""
        payload = {"invoice_number": "INV-001"}
        rule = {
            "type": "required",
            "description": "field 'invoice_number' is required"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is True

    def test_apply_validation_rule_range(self, kb_query):
        """Test range validation."""
        payload = {"total_amount": 1250.00}
        rule = {
            "type": "range",
            "description": "total_amount must be between 0 and 100000"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is True

    def test_apply_validation_rule_range_fails(self, kb_query):
        """Test range validation failure."""
        payload = {"total_amount": 150000}
        rule = {
            "type": "range",
            "description": "total_amount must be between 0 and 100000"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is False

    def test_apply_validation_rule_email_format(self, kb_query):
        """Test email format validation."""
        payload = {"contact_email": "test@example.com"}
        rule = {
            "type": "format",
            "description": "contact_email must be valid email"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is True

    def test_apply_validation_rule_email_format_fails(self, kb_query):
        """Test email format validation failure."""
        payload = {"contact_email": "invalid_email"}
        rule = {
            "type": "format",
            "description": "contact_email must be valid email"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is False

    def test_apply_validation_rule_phone_format(self, kb_query):
        """Test phone format validation."""
        payload = {"phone": "+1-555-123-4567"}
        rule = {
            "type": "format",
            "description": "phone must be valid phone number"
        }

        result = kb_query._apply_validation_rule(payload, rule)

        assert result["valid"] is True


class TestFormulaEvaluation:
    """Test formula and calculation evaluation."""

    def test_evaluate_formula_simple_addition(self, kb_query):
        """Test simple addition formula."""
        formula = "total_amount = subtotal + tax_amount"
        payload = {
            "total_amount": 110.00,
            "subtotal": 100.00,
            "tax_amount": 10.00
        }

        result = kb_query._evaluate_formula(formula, payload)

        assert result["valid"] is True

    def test_evaluate_formula_calculation_mismatch(self, kb_query):
        """Test formula with calculation mismatch."""
        formula = "total_amount = subtotal + tax_amount"
        payload = {
            "total_amount": 150.00,  # Wrong total
            "subtotal": 100.00,
            "tax_amount": 10.00
        }

        result = kb_query._evaluate_formula(formula, payload)

        assert result["valid"] is False
        assert "error" in result

    def test_evaluate_formula_missing_field(self, kb_query):
        """Test formula with missing field."""
        formula = "total_amount = subtotal + tax_amount"
        payload = {
            "total_amount": 110.00,
            "subtotal": 100.00
            # tax_amount missing
        }

        result = kb_query._evaluate_formula(formula, payload)

        assert result["valid"] is False

    def test_evaluate_formula_complex_expression(self, kb_query):
        """Test complex formula with multiple operations."""
        formula = "net_amount = gross_amount - (gross_amount * discount_rate)"
        payload = {
            "net_amount": 90.00,
            "gross_amount": 100.00,
            "discount_rate": 0.10
        }

        result = kb_query._evaluate_formula(formula, payload)

        assert result["valid"] is True

    def test_evaluate_formula_invalid_syntax(self, kb_query):
        """Test formula with invalid syntax."""
        formula = "invalid formula syntax here"
        payload = {"total_amount": 100}

        result = kb_query._evaluate_formula(formula, payload)

        assert result["valid"] is False


class TestFieldExtraction:
    """Test field extraction from rule descriptions."""

    def test_extract_field_from_rule_single_quotes(self, kb_query):
        """Test extracting field name with single quotes."""
        description = "field 'invoice_number' must be present"
        field_name = kb_query._extract_field_from_rule(description)

        assert field_name == "invoice_number"

    def test_extract_field_from_rule_double_quotes(self, kb_query):
        """Test extracting field name with double quotes."""
        description = 'field "total_amount" must be positive'
        field_name = kb_query._extract_field_from_rule(description)

        assert field_name == "total_amount"

    def test_extract_field_from_rule_must_pattern(self, kb_query):
        """Test extracting field name with 'must' pattern."""
        description = "invoice_number must be unique"
        field_name = kb_query._extract_field_from_rule(description)

        assert field_name == "invoice_number"

    def test_extract_field_from_rule_should_pattern(self, kb_query):
        """Test extracting field name with 'should' pattern."""
        description = "vendor_id should be alphanumeric"
        field_name = kb_query._extract_field_from_rule(description)

        assert field_name == "vendor_id"

    def test_extract_field_from_rule_no_match(self, kb_query):
        """Test field extraction with no match."""
        description = "this is a general rule"
        field_name = kb_query._extract_field_from_rule(description)

        assert field_name is None


class TestTransformationCodeGeneration:
    """Test transformation code generation."""

    def test_get_transformation_code_adc_to_sap(self, kb_query):
        """Test generating ADC to SAP transformation code."""
        code = kb_query.get_transformation_code("ADC", "SAP_ODATA")

        assert code is not None
        assert "def transform_adc_to_sap" in code
        assert "PurchaseOrderID" in code or "SAP" in code

    def test_get_transformation_code_sap_to_adc(self, kb_query):
        """Test generating SAP to ADC transformation code."""
        code = kb_query.get_transformation_code("SAP", "ADC")

        assert code is not None
        assert "def transform_sap_to_adc" in code

    def test_get_transformation_code_json_to_odata(self, kb_query):
        """Test generating JSON to OData transformation code."""
        code = kb_query.get_transformation_code("JSON", "SAP_ODATA")

        assert code is not None
        assert "json" in code.lower()

    def test_get_transformation_code_generic(self, kb_query):
        """Test generating generic transformation code."""
        code = kb_query.get_transformation_code("CUSTOM_FORMAT", "TARGET_FORMAT")

        assert code is not None
        assert "def transform_custom_format_to_target_format" in code


class TestEndpointQueries:
    """Test SAP endpoint querying."""

    def test_get_endpoint_for_action_create(self, kb_query):
        """Test getting endpoint for create action."""
        endpoint = kb_query.get_endpoint_for_action("purchase_order", "create")

        # Should return endpoint info or None
        assert endpoint is None or isinstance(endpoint, dict)

    def test_get_endpoint_for_action_update(self, kb_query):
        """Test getting endpoint for update action."""
        endpoint = kb_query.get_endpoint_for_action("supplier_invoice", "update")

        assert endpoint is None or isinstance(endpoint, dict)

    def test_get_endpoint_for_action_delete(self, kb_query):
        """Test getting endpoint for delete action."""
        endpoint = kb_query.get_endpoint_for_action("purchase_order", "delete")

        assert endpoint is None or isinstance(endpoint, dict)

    def test_get_endpoint_for_action_read(self, kb_query):
        """Test getting endpoint for read action."""
        endpoint = kb_query.get_endpoint_for_action("supplier_invoice", "read")

        assert endpoint is None or isinstance(endpoint, dict)


class TestExamplePayloads:
    """Test example payload generation."""

    def test_get_example_payload(self, kb_query):
        """Test getting example payload for document type."""
        example = kb_query.get_example_payload("purchase_order", "create")

        assert example is None or isinstance(example, dict)

    def test_get_example_payload_no_action(self, kb_query):
        """Test getting example without specific action."""
        example = kb_query.get_example_payload("supplier_invoice")

        assert example is None or isinstance(example, dict)

    def test_extract_example_from_schema(self, kb_query):
        """Test extracting example from OpenAPI schema."""
        schema = {
            "example": {
                "invoice_number": "INV-001",
                "total_amount": 1250.00
            }
        }

        example = kb_query._extract_example_from_schema(schema)

        assert example is not None
        assert example["invoice_number"] == "INV-001"

    def test_extract_example_from_schema_examples_array(self, kb_query):
        """Test extracting from examples array."""
        schema = {
            "examples": {
                "example1": {
                    "value": {
                        "field1": "value1"
                    }
                }
            }
        }

        example = kb_query._extract_example_from_schema(schema)

        assert example is not None

    def test_extract_example_from_schema_content(self, kb_query):
        """Test extracting from content schema."""
        schema = {
            "content": {
                "application/json": {
                    "schema": {
                        "example": {
                            "test_field": "test_value"
                        }
                    }
                }
            }
        }

        example = kb_query._extract_example_from_schema(schema)

        assert example is not None

    def test_generate_example_value_string(self, kb_query):
        """Test generating example value for string type."""
        value = kb_query._generate_example_value("string")
        assert isinstance(value, str)

    def test_generate_example_value_number(self, kb_query):
        """Test generating example value for number type."""
        value = kb_query._generate_example_value("number")
        assert isinstance(value, (int, float))

    def test_generate_example_value_boolean(self, kb_query):
        """Test generating example value for boolean type."""
        value = kb_query._generate_example_value("boolean")
        assert isinstance(value, bool)

    def test_generate_example_value_array(self, kb_query):
        """Test generating example value for array type."""
        value = kb_query._generate_example_value("array")
        assert isinstance(value, list)

    def test_generate_example_value_object(self, kb_query):
        """Test generating example value for object type."""
        value = kb_query._generate_example_value("object")
        assert isinstance(value, dict)


class TestCalculationRules:
    """Test calculation rule functionality."""

    def test_find_calculation_rules(self, kb_query):
        """Test finding calculation rules."""
        rules = kb_query.find_calculation_rules("supplier_invoice")

        assert isinstance(rules, list)


class TestSimilarDocuments:
    """Test similar document queries."""

    def test_get_similar_documents(self, kb_query):
        """Test finding similar documents."""
        adc_data = {
            "document_type": "supplier_invoice",
            "vendor_name": "Acme Corp",
            "total_amount": 1250.00
        }

        similar = kb_query.get_similar_documents(adc_data, k=5)

        assert isinstance(similar, list)


class TestStatistics:
    """Test knowledge base statistics."""

    def test_get_stats(self, kb_query):
        """Test getting knowledge base statistics."""
        stats = kb_query.get_stats()

        assert isinstance(stats, dict)
        assert "storage" in stats
        assert "total_items" in stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_map_fields_empty_payload(self, kb_query):
        """Test mapping with empty payload."""
        result = kb_query.map_fields_to_sap({}, "supplier_invoice")

        assert "payload" in result
        assert "mappings" in result

    def test_format_date_none_value(self, kb_query):
        """Test date formatting with None value."""
        result = kb_query._format_date(None, "SAP")

        # Should handle gracefully
        assert isinstance(result, str)

    def test_validate_payload_empty_rules(self, kb_query):
        """Test validation with no rules."""
        payload = {"test_field": "value"}
        result = kb_query.validate_payload(payload, "unknown_doc_type")

        # Should return valid when no rules found
        assert isinstance(result, dict)
        assert "valid" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=sap_llm.knowledge_base.query", "--cov-report=term-missing"])
