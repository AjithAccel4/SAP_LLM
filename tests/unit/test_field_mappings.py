"""
Unit tests for Field Mapping Manager

Tests field mapping functionality including:
- Mapping loading and caching
- Field transformations
- Data validation
- Nested structure handling
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from sap_llm.knowledge_base.field_mappings import FieldMappingManager


class TestFieldMappingManager:
    """Unit tests for FieldMappingManager class."""

    @pytest.fixture
    def mapping_manager(self):
        """Create a FieldMappingManager instance."""
        return FieldMappingManager()

    @pytest.fixture
    def sample_po_data(self):
        """Sample purchase order data for testing."""
        return {
            "po_number": "PO123456",
            "po_date": "2024-01-15",
            "vendor_id": "V001",
            "company_code": "1000",
            "currency": "usd",
            "total_amount": "1,234.56",
            "purchasing_organization": "100",
            "purchasing_group": "1",
            "items": [
                {
                    "item_number": "10",
                    "material_number": "MAT001",
                    "quantity": "100",
                    "unit_price": "12.35"
                }
            ]
        }

    @pytest.fixture
    def sample_invoice_data(self):
        """Sample supplier invoice data for testing."""
        return {
            "invoice_number": "INV-2024-001",
            "invoice_date": "2024-01-15",
            "posting_date": "2024-01-16",
            "vendor_id": "V001",
            "company_code": "1000",
            "currency": "EUR",
            "total_amount": "5000.00",
            "tax_amount": "950.00",
            "fiscal_year": "2024",
            "purchase_order": "PO123456"
        }

    def test_manager_initialization(self, mapping_manager):
        """Test that manager initializes correctly."""
        assert mapping_manager is not None
        assert isinstance(mapping_manager._mappings_cache, dict)
        assert len(mapping_manager._mappings_cache) > 0

    def test_get_mapping_purchase_order_standard(self, mapping_manager):
        """Test retrieving Purchase Order Standard mapping."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")

        assert mapping is not None
        assert mapping["document_type"] == "PurchaseOrder"
        assert mapping["subtype"] == "Standard"
        assert "mappings" in mapping
        assert "po_number" in mapping["mappings"]

    def test_get_mapping_supplier_invoice(self, mapping_manager):
        """Test retrieving Supplier Invoice mapping."""
        mapping = mapping_manager.get_mapping("SupplierInvoice", "Standard")

        assert mapping is not None
        assert mapping["document_type"] == "SupplierInvoice"
        assert "invoice_number" in mapping["mappings"]

    def test_get_mapping_not_found(self, mapping_manager):
        """Test that non-existent mappings return None."""
        mapping = mapping_manager.get_mapping("NonExistent", "Subtype")
        assert mapping is None

    def test_get_document_types(self, mapping_manager):
        """Test getting list of document types."""
        doc_types = mapping_manager.get_document_types()

        assert isinstance(doc_types, list)
        assert len(doc_types) > 0
        assert "PurchaseOrder" in doc_types
        assert "SupplierInvoice" in doc_types

    def test_get_subtypes(self, mapping_manager):
        """Test getting subtypes for a document type."""
        subtypes = mapping_manager.get_subtypes("PurchaseOrder")

        assert isinstance(subtypes, list)
        assert len(subtypes) > 0
        assert "Standard" in subtypes
        assert "Service" in subtypes

    def test_transformation_uppercase(self, mapping_manager):
        """Test uppercase transformation."""
        result = mapping_manager.apply_transformations(
            "test value",
            ["uppercase"],
            "test_field"
        )
        assert result == "TEST VALUE"

    def test_transformation_lowercase(self, mapping_manager):
        """Test lowercase transformation."""
        result = mapping_manager.apply_transformations(
            "TEST VALUE",
            ["lowercase"],
            "test_field"
        )
        assert result == "test value"

    def test_transformation_trim(self, mapping_manager):
        """Test trim transformation."""
        result = mapping_manager.apply_transformations(
            "  test value  ",
            ["trim"],
            "test_field"
        )
        assert result == "test value"

    def test_transformation_pad_left(self, mapping_manager):
        """Test pad_left transformation."""
        result = mapping_manager.apply_transformations(
            "123",
            ["pad_left:10:0"],
            "test_field"
        )
        assert result == "0000000123"

    def test_transformation_pad_right(self, mapping_manager):
        """Test pad_right transformation."""
        result = mapping_manager.apply_transformations(
            "ABC",
            ["pad_right:5:X"],
            "test_field"
        )
        assert result == "ABCXX"

    def test_transformation_parse_date(self, mapping_manager):
        """Test date parsing."""
        result = mapping_manager.apply_transformations(
            "2024-01-15",
            ["parse_date"],
            "date_field"
        )
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_transformation_format_date_yyyymmdd(self, mapping_manager):
        """Test date formatting to YYYYMMDD."""
        result = mapping_manager.apply_transformations(
            "2024-01-15",
            ["parse_date", "format_date:YYYYMMDD"],
            "date_field"
        )
        assert result == "20240115"

    def test_transformation_parse_amount(self, mapping_manager):
        """Test amount parsing."""
        result = mapping_manager.apply_transformations(
            "$1,234.56",
            ["parse_amount"],
            "amount_field"
        )
        assert result == 1234.56

    def test_transformation_format_decimal(self, mapping_manager):
        """Test decimal formatting."""
        result = mapping_manager.apply_transformations(
            "1234.5678",
            ["parse_amount", "format_decimal:2"],
            "amount_field"
        )
        assert result == 1234.57

    def test_transformation_validate_currency(self, mapping_manager):
        """Test currency validation."""
        result = mapping_manager.apply_transformations(
            "usd",
            ["uppercase", "validate_iso_currency"],
            "currency_field"
        )
        assert result == "USD"

    def test_transformation_chain(self, mapping_manager):
        """Test chaining multiple transformations."""
        result = mapping_manager.apply_transformations(
            "  test123  ",
            ["trim", "uppercase", "pad_left:10:0"],
            "test_field"
        )
        assert result == "00TEST123"

    def test_transform_data_purchase_order(self, mapping_manager, sample_po_data):
        """Test transforming purchase order data."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        result = mapping_manager.transform_data(sample_po_data, mapping)

        assert "PurchaseOrder" in result
        assert result["PurchaseOrder"] == "PO123456"
        assert result["PurchaseOrderDate"] == "20240115"
        assert result["Supplier"] == "0000000V01"  # Padded
        assert result["CompanyCode"] == "1000"
        assert result["DocumentCurrency"] == "USD"
        assert result["TotalAmount"] == 1234.56

    def test_transform_data_with_nested(self, mapping_manager, sample_po_data):
        """Test transforming data with nested items."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        result = mapping_manager.transform_data(sample_po_data, mapping)

        assert "to_PurchaseOrderItem" in result
        assert len(result["to_PurchaseOrderItem"]) == 1

        item = result["to_PurchaseOrderItem"][0]
        assert item["PurchaseOrderItem"] == "00010"  # Padded
        assert item["Material"] == "MAT001"
        assert item["OrderQuantity"] == 100.0

    def test_transform_data_supplier_invoice(self, mapping_manager, sample_invoice_data):
        """Test transforming supplier invoice data."""
        mapping = mapping_manager.get_mapping("SupplierInvoice", "Standard")
        result = mapping_manager.transform_data(sample_invoice_data, mapping)

        assert "SupplierInvoiceIDByInvcgParty" in result
        assert result["SupplierInvoiceIDByInvcgParty"] == "INV-2024-001"
        assert result["DocumentDate"] == "20240115"
        assert result["PostingDate"] == "20240116"
        assert result["InvoicingParty"] == "0000000V01"
        assert result["InvoiceGrossAmount"] == 5000.0
        assert result["TaxAmount"] == 950.0
        assert result["FiscalYear"] == "2024"

    def test_validate_mapping_success(self, mapping_manager, sample_po_data):
        """Test successful validation."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        is_valid, errors = mapping_manager.validate_mapping(
            sample_po_data, mapping, strict=False
        )

        assert is_valid
        assert len(errors) == 0

    def test_validate_mapping_missing_required(self, mapping_manager):
        """Test validation with missing required field."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        incomplete_data = {
            "vendor_id": "V001"
            # Missing required fields: po_number, po_date, etc.
        }

        is_valid, errors = mapping_manager.validate_mapping(
            incomplete_data, mapping, strict=True
        )

        assert not is_valid
        assert len(errors) > 0

    def test_validate_mapping_pattern_mismatch(self, mapping_manager):
        """Test validation with pattern mismatch."""
        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        invalid_data = {
            "po_number": "PO123456",
            "po_date": "2024-01-15",
            "vendor_id": "V001",
            "company_code": "1000",
            "purchasing_organization": "100",
            "purchasing_group": "1",
            "currency": "INVALID_CURRENCY_CODE"  # Too long, should be 3 chars
        }

        is_valid, errors = mapping_manager.validate_mapping(
            invalid_data, mapping, strict=True
        )

        assert not is_valid
        assert any("currency" in str(error).lower() for error in errors)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = FieldMappingManager._get_cache_key("PurchaseOrder", "Standard")
        key2 = FieldMappingManager._get_cache_key("purchaseorder", "standard")

        assert key1 == key2  # Should be case-insensitive
        assert key1 == "PURCHASEORDER:STANDARD"

    def test_parse_amount_variations(self, mapping_manager):
        """Test parsing various amount formats."""
        test_cases = [
            ("1234.56", 1234.56),
            ("$1,234.56", 1234.56),
            ("€1 234,56", 1234.56),
            ("1234", 1234.0),
            ("(500)", -500.0),
        ]

        for input_val, expected in test_cases:
            result = mapping_manager._parse_amount_value(input_val)
            assert result == expected, f"Failed for input: {input_val}"

    def test_parse_date_variations(self, mapping_manager):
        """Test parsing various date formats."""
        test_cases = [
            "2024-01-15",
            "15/01/2024",
            "01/15/2024",
            "20240115",
            "15.01.2024",
        ]

        for date_str in test_cases:
            result = mapping_manager._parse_date_value(date_str)
            assert isinstance(result, datetime)
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 15

    def test_transformation_error_handling(self, mapping_manager):
        """Test that transformation errors are handled gracefully."""
        # Invalid date should return original value
        result = mapping_manager.apply_transformations(
            "INVALID_DATE",
            ["parse_date"],
            "date_field"
        )
        assert result == "INVALID_DATE"

        # Invalid amount should return original value
        result = mapping_manager.apply_transformations(
            "NOT_A_NUMBER",
            ["parse_amount"],
            "amount_field"
        )
        assert result == "NOT_A_NUMBER"

    def test_deep_nesting(self, mapping_manager):
        """Test handling of deeply nested structures."""
        # Create a mapping with nested structure
        mapping = {
            "config": {"copy_unmapped": False},
            "mappings": {
                "field1": {
                    "sap_field": "Field1",
                    "transformations": ["uppercase"]
                }
            },
            "nested_mappings": {
                "level1": {
                    "sap_collection": "Level1",
                    "mappings": {
                        "field2": {
                            "sap_field": "Field2",
                            "transformations": ["uppercase"]
                        }
                    }
                }
            }
        }

        data = {
            "field1": "value1",
            "level1": [
                {"field2": "value2"}
            ]
        }

        result = mapping_manager.transform_data(data, mapping, max_nesting_level=5)

        assert result["Field1"] == "VALUE1"
        assert "Level1" in result
        assert result["Level1"][0]["Field2"] == "VALUE2"

    def test_default_values(self, mapping_manager):
        """Test that default values are applied for missing fields."""
        mapping = {
            "config": {},
            "mappings": {
                "optional_field": {
                    "sap_field": "OptionalField",
                    "required": False,
                    "default": "DEFAULT_VALUE",
                    "transformations": []
                }
            }
        }

        data = {}  # Empty data

        result = mapping_manager.transform_data(data, mapping)

        assert "OptionalField" in result
        assert result["OptionalField"] == "DEFAULT_VALUE"


class TestPerformance:
    """Performance tests for field mapping."""

    @pytest.fixture
    def mapping_manager(self):
        """Create a FieldMappingManager instance."""
        return FieldMappingManager()

    def test_transform_1000_documents_performance(self, mapping_manager):
        """Test transforming 1000 documents in under 2 seconds."""
        import time

        mapping = mapping_manager.get_mapping("PurchaseOrder", "Standard")

        # Sample data
        sample_data = {
            "po_number": "PO123456",
            "po_date": "2024-01-15",
            "vendor_id": "V001",
            "company_code": "1000",
            "currency": "USD",
            "total_amount": "1234.56",
            "purchasing_organization": "100",
            "purchasing_group": "1"
        }

        start_time = time.time()

        # Transform 1000 documents
        for _ in range(1000):
            result = mapping_manager.transform_data(sample_data, mapping)

        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0, f"Performance test failed: {elapsed_time:.2f}s > 2.0s"

    def test_mapping_cache_efficiency(self, mapping_manager):
        """Test that mapping cache is working efficiently."""
        import time

        # First call - might be slower
        start_time = time.time()
        mapping1 = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        first_call_time = time.time() - start_time

        # Second call - should be from cache
        start_time = time.time()
        mapping2 = mapping_manager.get_mapping("PurchaseOrder", "Standard")
        second_call_time = time.time() - start_time

        # Cache should make second call faster or equal
        assert second_call_time <= first_call_time * 2  # Allow some variance
        assert mapping1 == mapping2
