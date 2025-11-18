"""
Comprehensive unit tests for KnowledgeBaseQuery field mapping and transformations.

Tests the complete field transformation pipeline including:
- SAP API field mappings (Purchase Order, Invoice, Sales Order)
- Date format transformations
- Amount parsing and formatting
- Currency validation
- Field padding and trimming
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from sap_llm.knowledge_base.query import KnowledgeBaseQuery
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestFieldTransformations:
    """Tests for individual field transformation functions."""

    @pytest.fixture
    def query_engine(self):
        """Create KnowledgeBaseQuery instance with mocked storage."""
        mock_storage = MagicMock(spec=KnowledgeBaseStorage)
        return KnowledgeBaseQuery(storage=mock_storage)

    # =========================================================================
    # Date Transformation Tests
    # =========================================================================

    @pytest.mark.parametrize("date_input,expected", [
        ("2024-01-15", datetime(2024, 1, 15)),  # ISO format
        ("15/01/2024", datetime(2024, 1, 15)),  # European format
        ("01/15/2024", datetime(2024, 1, 15)),  # US format
        ("20240115", datetime(2024, 1, 15)),    # SAP format
        ("15.01.2024", datetime(2024, 1, 15)),  # German format
    ])
    def test_parse_date_value_various_formats(self, query_engine, date_input, expected):
        """Test parsing dates from various formats."""
        result = query_engine._parse_date_value(date_input)
        assert result == expected

    def test_parse_date_value_already_datetime(self, query_engine):
        """Test parsing when input is already datetime."""
        dt = datetime(2024, 1, 15)
        result = query_engine._parse_date_value(dt)
        assert result == dt

    def test_parse_date_value_invalid(self, query_engine):
        """Test parsing invalid date raises error."""
        with pytest.raises(ValueError, match="Cannot parse date"):
            query_engine._parse_date_value("invalid-date")

    @pytest.mark.parametrize("date_input,format_str,expected", [
        ("2024-01-15", "YYYYMMDD", "20240115"),
        ("2024-01-15", "YYYY-MM-DD", "2024-01-15"),
        ("2024-01-15", "DD/MM/YYYY", "15/01/2024"),
        ("2024-01-15", "MM/DD/YYYY", "01/15/2024"),
        ("2024-01-15", "%B %d, %Y", "January 15, 2024"),
    ])
    def test_format_date_value(self, query_engine, date_input, format_str, expected):
        """Test formatting dates to various formats."""
        result = query_engine._format_date_value(date_input, format_str)
        assert result == expected

    def test_format_date_value_from_datetime(self, query_engine):
        """Test formatting datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = query_engine._format_date_value(dt, "YYYYMMDD")
        assert result == "20240115"

    # =========================================================================
    # Amount Transformation Tests
    # =========================================================================

    @pytest.mark.parametrize("amount_input,expected", [
        (1250.50, 1250.50),
        ("1250.50", 1250.50),
        ("$1,250.50", 1250.50),
        ("€1.250,50", 1250.50),
        ("1,250.50", 1250.50),
        ("USD 1250.50", 1250.50),
        (1000, 1000.0),
    ])
    def test_parse_amount_value(self, query_engine, amount_input, expected):
        """Test parsing amounts from various formats."""
        result = query_engine._parse_amount_value(amount_input)
        assert result == expected

    def test_parse_amount_value_invalid(self, query_engine):
        """Test parsing invalid amount raises error."""
        with pytest.raises(ValueError, match="Cannot parse amount"):
            query_engine._parse_amount_value("not-a-number")

    def test_parse_amount_negative(self, query_engine):
        """Test parsing negative amounts."""
        result = query_engine._parse_amount_value("-1250.50")
        assert result == -1250.50

    # =========================================================================
    # Currency Validation Tests
    # =========================================================================

    @pytest.mark.parametrize("currency_input,expected", [
        ("USD", "USD"),
        ("usd", "USD"),
        ("  EUR  ", "EUR"),
        ("gbp", "GBP"),
    ])
    def test_validate_currency_value(self, query_engine, currency_input, expected):
        """Test validating and normalizing currency codes."""
        result = query_engine._validate_currency_value(currency_input)
        assert result == expected

    def test_validate_currency_invalid_length(self, query_engine):
        """Test currency validation rejects invalid length."""
        with pytest.raises(ValueError, match="Invalid currency code length"):
            query_engine._validate_currency_value("US")

        with pytest.raises(ValueError, match="Invalid currency code length"):
            query_engine._validate_currency_value("USDD")

    def test_validate_currency_uncommon_warns(self, query_engine, caplog):
        """Test uncommon currency code logs warning."""
        result = query_engine._validate_currency_value("XXX")
        assert result == "XXX"  # Still returns it
        assert "not in common list" in caplog.text

    # =========================================================================
    # General Transformation Tests
    # =========================================================================

    @pytest.mark.parametrize("value,transformations,expected", [
        ("hello", ["uppercase"], "HELLO"),
        ("HELLO", ["lowercase"], "hello"),
        ("  hello  ", ["trim"], "hello"),
        ("123", ["pad_left:5:0"], "00123"),
        ("123", ["pad_right:5:0"], "12300"),
        ("hello world", ["uppercase", "trim"], "HELLO WORLD"),
    ])
    def test_apply_transformations_basic(self, query_engine, value, transformations, expected):
        """Test basic transformations (uppercase, lowercase, trim, padding)."""
        result = query_engine._apply_transformations(value, transformations)
        assert result == expected

    def test_apply_transformations_parse_and_format_date(self, query_engine):
        """Test date parsing and formatting transformation."""
        transformations = ["parse_date", "format_date:YYYYMMDD"]
        result = query_engine._apply_transformations("2024-01-15", transformations)
        assert result == "20240115"

    def test_apply_transformations_parse_and_format_amount(self, query_engine):
        """Test amount parsing and decimal formatting transformation."""
        transformations = ["parse_amount", "format_decimal:2"]
        result = query_engine._apply_transformations("$1,250.5", transformations)
        assert result == 1250.50

    def test_apply_transformations_validate_currency(self, query_engine):
        """Test currency validation transformation."""
        transformations = ["uppercase", "validate_iso_currency"]
        result = query_engine._apply_transformations("usd", transformations)
        assert result == "USD"

    def test_apply_transformations_invalid_returns_original(self, query_engine):
        """Test that failed transformations return original value."""
        transformations = ["parse_date"]
        result = query_engine._apply_transformations("invalid-date", transformations)
        assert result == "invalid-date"

    def test_apply_transformations_unknown_transformation_warns(self, query_engine, caplog):
        """Test unknown transformation logs warning."""
        result = query_engine._apply_transformations("value", ["unknown_transform"])
        assert result == "value"
        assert "Unknown transformation" in caplog.text


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestSAPFieldMappings:
    """Tests for SAP field mapping generation."""

    @pytest.fixture
    def query_engine(self):
        """Create KnowledgeBaseQuery instance."""
        mock_storage = MagicMock(spec=KnowledgeBaseStorage)
        return KnowledgeBaseQuery(storage=mock_storage)

    # =========================================================================
    # Purchase Order Mapping Tests
    # =========================================================================

    def test_get_sap_api_mapping_purchase_order(self, query_engine):
        """Test SAP API mapping for purchase orders."""
        mapping = query_engine._get_sap_api_mapping("PURCHASE_ORDER")

        # Check base fields
        assert "document_date" in mapping
        assert mapping["document_date"]["target_field"] == "DocumentDate"
        assert "parse_date" in mapping["document_date"]["transformations"]

        # Check PO-specific fields
        assert "po_number" in mapping
        assert mapping["po_number"]["target_field"] == "PurchaseOrder"

        assert "vendor_id" in mapping
        assert mapping["vendor_id"]["target_field"] == "Supplier"
        assert "pad_left:10:0" in mapping["vendor_id"]["transformations"]

        assert "purchasing_organization" in mapping
        assert mapping["purchasing_organization"]["target_field"] == "PurchasingOrganization"

    def test_get_sap_api_mapping_purchase_order_short_code(self, query_engine):
        """Test PO mapping works with short code 'PO'."""
        mapping = query_engine._get_sap_api_mapping("PO")
        assert "po_number" in mapping
        assert mapping["po_number"]["target_field"] == "PurchaseOrder"

    # =========================================================================
    # Supplier Invoice Mapping Tests
    # =========================================================================

    def test_get_sap_api_mapping_supplier_invoice(self, query_engine):
        """Test SAP API mapping for supplier invoices."""
        mapping = query_engine._get_sap_api_mapping("SUPPLIER_INVOICE")

        # Check invoice-specific fields
        assert "invoice_number" in mapping
        assert mapping["invoice_number"]["target_field"] == "SupplierInvoiceIDByInvcgParty"

        assert "posting_date" in mapping
        assert mapping["posting_date"]["target_field"] == "PostingDate"

        assert "total_amount" in mapping
        assert mapping["total_amount"]["target_field"] == "InvoiceGrossAmount"

        assert "tax_amount" in mapping
        assert mapping["tax_amount"]["target_field"] == "TaxAmount"

        assert "fiscal_year" in mapping
        assert mapping["fiscal_year"]["target_field"] == "FiscalYear"

    def test_get_sap_api_mapping_invoice_variations(self, query_engine):
        """Test invoice mapping handles field name variations."""
        mapping = query_engine._get_sap_api_mapping("SUPPLIER_INVOICE")

        # Check all invoice number variations
        assert "invoice_number" in mapping
        assert "supplier_invoice_number" in mapping
        assert "vendor_invoice_number" in mapping

        # All should map to same target
        target = "SupplierInvoiceIDByInvcgParty"
        assert mapping["invoice_number"]["target_field"] == target
        assert mapping["supplier_invoice_number"]["target_field"] == target
        assert mapping["vendor_invoice_number"]["target_field"] == target

    # =========================================================================
    # Sales Order Mapping Tests
    # =========================================================================

    def test_get_sap_api_mapping_sales_order(self, query_engine):
        """Test SAP API mapping for sales orders."""
        mapping = query_engine._get_sap_api_mapping("SALES_ORDER")

        # Check SO-specific fields
        assert "sales_order_number" in mapping
        assert mapping["sales_order_number"]["target_field"] == "SalesOrderNumber"

        assert "customer_id" in mapping
        assert mapping["customer_id"]["target_field"] == "SoldToParty"

        assert "sales_organization" in mapping
        assert mapping["sales_organization"]["target_field"] == "SalesOrganization"

        assert "distribution_channel" in mapping
        assert mapping["distribution_channel"]["target_field"] == "DistributionChannel"

    # =========================================================================
    # Reverse Mapping Tests (SAP → Internal)
    # =========================================================================

    def test_get_from_sap_mapping(self, query_engine):
        """Test reverse mapping from SAP format to internal format."""
        mapping = query_engine._get_from_sap_mapping("INTERNAL")

        # Check reverse mappings
        assert mapping["DocumentNumber"] == "document_number"
        assert mapping["DocumentDate"] == "document_date"
        assert mapping["PurchaseOrder"] == "purchase_order"
        assert mapping["Supplier"] == "vendor_id"
        assert mapping["TotalAmount"] == "total_amount"

    # =========================================================================
    # Field Mapping Routing Tests
    # =========================================================================

    def test_get_field_mapping_to_sap(self, query_engine):
        """Test field mapping routing to SAP API."""
        mapping = query_engine._get_field_mapping("PURCHASE_ORDER", "SAP_API")
        assert len(mapping) > 0
        assert "po_number" in mapping

        mapping = query_engine._get_field_mapping("SUPPLIER_INVOICE", "SAP_ODATA")
        assert len(mapping) > 0
        assert "invoice_number" in mapping

    def test_get_field_mapping_from_sap(self, query_engine):
        """Test field mapping routing from SAP."""
        mapping = query_engine._get_field_mapping("SAP_API", "INTERNAL")
        assert len(mapping) > 0
        assert "DocumentNumber" in mapping

    def test_get_field_mapping_no_mapping(self, query_engine):
        """Test field mapping returns empty dict when no mapping exists."""
        mapping = query_engine._get_field_mapping("UNKNOWN", "UNKNOWN")
        assert mapping == {}


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestTransformFormat:
    """Tests for complete transform_format workflow."""

    @pytest.fixture
    def query_engine(self):
        """Create KnowledgeBaseQuery instance."""
        mock_storage = MagicMock(spec=KnowledgeBaseStorage)
        return KnowledgeBaseQuery(storage=mock_storage)

    # =========================================================================
    # Purchase Order Transformation Tests
    # =========================================================================

    def test_transform_purchase_order_to_sap(self, query_engine):
        """Test complete PO transformation to SAP format."""
        source_data = {
            "po_number": "po-12345",
            "vendor_id": "1001",
            "po_date": "2024-01-15",
            "total_amount": "$1,250.50",
            "currency": "usd",
            "company_code": "10",
        }

        result = query_engine.transform_format(source_data, "PURCHASE_ORDER", "SAP_API")

        # Check transformed fields
        assert result["PurchaseOrder"] == "PO-12345"  # Uppercased
        assert result["Supplier"] == "0000001001"     # Padded to 10 digits
        assert result["PurchaseOrderDate"] == "20240115"  # SAP date format
        assert result["TotalAmount"] == 1250.50      # Parsed and formatted
        assert result["DocumentCurrency"] == "USD"   # Validated currency
        assert result["CompanyCode"] == "0010"       # Padded to 4 digits

    def test_transform_purchase_order_multiple_vendor_fields(self, query_engine):
        """Test PO transformation handles vendor field variations."""
        # Test vendor_id
        result1 = query_engine.transform_format(
            {"vendor_id": "1001"},
            "PURCHASE_ORDER",
            "SAP_API"
        )
        assert result1["Supplier"] == "0000001001"

        # Test vendor_number
        result2 = query_engine.transform_format(
            {"vendor_number": "1001"},
            "PURCHASE_ORDER",
            "SAP_API"
        )
        assert result2["Supplier"] == "0000001001"

        # Test supplier
        result3 = query_engine.transform_format(
            {"supplier": "1001"},
            "PURCHASE_ORDER",
            "SAP_API"
        )
        assert result3["Supplier"] == "0000001001"

    # =========================================================================
    # Supplier Invoice Transformation Tests
    # =========================================================================

    def test_transform_supplier_invoice_to_sap(self, query_engine):
        """Test complete invoice transformation to SAP format."""
        source_data = {
            "invoice_number": "INV-2024-001",
            "vendor_id": "1001",
            "invoice_date": "2024-01-15",
            "posting_date": "2024-01-16",
            "total_amount": "€1,375.00",
            "tax_amount": "125.00",
            "currency": "EUR",
            "fiscal_year": "2024",
            "purchase_order": "po-12345",
        }

        result = query_engine.transform_format(source_data, "SUPPLIER_INVOICE", "SAP_API")

        # Check transformed fields
        assert result["SupplierInvoiceIDByInvcgParty"] == "INV-2024-001"
        assert result["InvoicingParty"] == "0000001001"
        assert result["DocumentDate"] == "20240115"
        assert result["PostingDate"] == "20240116"
        assert result["InvoiceGrossAmount"] == 1375.00
        assert result["TaxAmount"] == 125.00
        assert result["DocumentCurrency"] == "EUR"
        assert result["FiscalYear"] == "2024"
        assert result["PurchaseOrder"] == "PO-12345"

    def test_transform_invoice_with_missing_fields(self, query_engine):
        """Test invoice transformation handles missing optional fields."""
        source_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
        }

        result = query_engine.transform_format(source_data, "SUPPLIER_INVOICE", "SAP_API")

        # Required fields present
        assert "SupplierInvoiceIDByInvcgParty" in result
        assert "DocumentDate" in result

        # Optional fields not present
        assert "TaxAmount" not in result
        assert "PurchaseOrder" not in result

    # =========================================================================
    # Sales Order Transformation Tests
    # =========================================================================

    def test_transform_sales_order_to_sap(self, query_engine):
        """Test complete sales order transformation to SAP format."""
        source_data = {
            "sales_order_number": "SO-12345",
            "customer_id": "2001",
            "order_date": "2024-01-15",
            "total_amount": "1500.00",
            "currency": "USD",
            "sales_organization": "10",
            "distribution_channel": "1",
            "division": "0",
        }

        result = query_engine.transform_format(source_data, "SALES_ORDER", "SAP_API")

        # Check transformed fields
        assert result["SalesOrderNumber"] == "SO-12345"
        assert result["SoldToParty"] == "0000002001"  # Padded
        assert result["SalesOrderDate"] == "20240115"
        assert result["TotalAmount"] == 1500.00
        assert result["DocumentCurrency"] == "USD"
        assert result["SalesOrganization"] == "0010"  # Padded to 4
        assert result["DistributionChannel"] == "01"  # Padded to 2
        assert result["OrganizationDivision"] == "00"  # Padded to 2

    # =========================================================================
    # Reverse Transformation Tests (SAP → Internal)
    # =========================================================================

    def test_transform_from_sap_to_internal(self, query_engine):
        """Test transformation from SAP format to internal format."""
        sap_data = {
            "DocumentNumber": "1234567890",
            "DocumentDate": "20240115",
            "PurchaseOrder": "4500012345",
            "Supplier": "0000001001",
            "TotalAmount": "1250.50",
            "DocumentCurrency": "USD",
        }

        result = query_engine.transform_format(sap_data, "SAP_API", "INTERNAL")

        # Check reverse mappings
        assert result["document_number"] == "1234567890"
        assert result["document_date"] == "20240115"
        assert result["purchase_order"] == "4500012345"
        assert result["vendor_id"] == "0000001001"
        assert result["total_amount"] == "1250.50"
        assert result["currency"] == "USD"

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_transform_format_no_mapping_returns_original(self, query_engine, caplog):
        """Test that transform_format returns original data when no mapping exists."""
        source_data = {"field1": "value1", "field2": "value2"}

        result = query_engine.transform_format(source_data, "UNKNOWN", "UNKNOWN")

        assert result == source_data
        assert "No field mapping found" in caplog.text

    def test_transform_format_handles_transformation_errors(self, query_engine):
        """Test that transformation errors don't crash the pipeline."""
        source_data = {
            "po_date": "invalid-date",  # Will fail date parsing
            "total_amount": "not-a-number",  # Will fail amount parsing
        }

        result = query_engine.transform_format(source_data, "PURCHASE_ORDER", "SAP_API")

        # Should have attempted transformations but kept original values on error
        assert "PurchaseOrderDate" in result
        assert "TotalAmount" in result

    def test_transform_format_empty_source_data(self, query_engine):
        """Test transformation with empty source data."""
        result = query_engine.transform_format({}, "PURCHASE_ORDER", "SAP_API")
        assert result == {}

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_transform_format_preserves_unmapped_fields_when_configured(self, query_engine):
        """Test that unmapped fields can be preserved with configuration."""
        # Note: Current implementation doesn't copy unmapped fields by default
        # This test documents current behavior
        source_data = {
            "po_number": "12345",
            "custom_field": "custom_value",  # Not in mapping
        }

        result = query_engine.transform_format(source_data, "PURCHASE_ORDER", "SAP_API")

        # Mapped field is present
        assert "PurchaseOrder" in result

        # Custom field is NOT copied (current behavior)
        assert "custom_field" not in result

    def test_transform_format_chain_transformations(self, query_engine):
        """Test that multiple transformations can be chained."""
        # Currency field has both uppercase and validate_iso_currency transformations
        source_data = {"currency": "  usd  "}

        result = query_engine.transform_format(source_data, "PURCHASE_ORDER", "SAP_API")

        # Should be trimmed (via uppercase which calls str()),
        # uppercased, and validated
        assert result["DocumentCurrency"] == "USD"

    @pytest.mark.parametrize("doc_type,field_count_min", [
        ("PURCHASE_ORDER", 10),
        ("SUPPLIER_INVOICE", 15),
        ("SALES_ORDER", 10),
    ])
    def test_transform_format_comprehensive_mappings(self, query_engine, doc_type, field_count_min):
        """Test that comprehensive mappings are generated for each document type."""
        mapping = query_engine._get_sap_api_mapping(doc_type)

        # Check that we have substantial mappings
        assert len(mapping) >= field_count_min

        # Check that all mappings have proper structure
        for field, config in mapping.items():
            if isinstance(config, dict):
                assert "target_field" in config
                assert "transformations" in config
                assert isinstance(config["transformations"], list)


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestFieldMappingEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def query_engine(self):
        """Create KnowledgeBaseQuery instance."""
        mock_storage = MagicMock(spec=KnowledgeBaseStorage)
        return KnowledgeBaseQuery(storage=mock_storage)

    def test_date_parsing_whitespace(self, query_engine):
        """Test date parsing handles whitespace."""
        result = query_engine._parse_date_value("  2024-01-15  ")
        assert result == datetime(2024, 1, 15)

    def test_amount_parsing_various_symbols(self, query_engine):
        """Test amount parsing handles various currency symbols."""
        test_cases = [
            ("$1250.50", 1250.50),
            ("€1250.50", 1250.50),
            ("£1250.50", 1250.50),
            ("¥1250.50", 1250.50),
            ("1250.50 USD", 1250.50),
        ]

        for input_val, expected in test_cases:
            result = query_engine._parse_amount_value(input_val)
            assert result == expected

    def test_padding_already_correct_length(self, query_engine):
        """Test padding when value is already correct length."""
        result = query_engine._apply_transformations("12345", ["pad_left:5:0"])
        assert result == "12345"

    def test_padding_longer_than_target(self, query_engine):
        """Test padding when value is longer than target."""
        result = query_engine._apply_transformations("1234567", ["pad_left:5:0"])
        assert result == "1234567"  # Should not truncate

    def test_transformation_with_none_value(self, query_engine):
        """Test transformations handle None gracefully."""
        # Should convert None to "None" string
        result = query_engine._apply_transformations(None, ["uppercase"])
        assert result == "NONE"

    def test_decimal_formatting_rounds_correctly(self, query_engine):
        """Test decimal formatting rounds properly."""
        test_cases = [
            (1.234, 2, 1.23),
            (1.235, 2, 1.24),  # Round up
            (1.999, 2, 2.00),  # Round up
            (1.001, 2, 1.00),  # Round down
        ]

        for input_val, places, expected in test_cases:
            result = query_engine._apply_transformations(
                input_val,
                [f"format_decimal:{places}"]
            )
            assert result == expected

    def test_currency_validation_case_insensitive(self, query_engine):
        """Test currency validation is case-insensitive."""
        for code in ["usd", "USD", "Usd", "uSD"]:
            result = query_engine._validate_currency_value(code)
            assert result == "USD"

    def test_date_formats_with_timestamps(self, query_engine):
        """Test date parsing handles timestamps."""
        result1 = query_engine._parse_date_value("2024-01-15T10:30:00")
        assert result1.year == 2024
        assert result1.month == 1
        assert result1.day == 15

        result2 = query_engine._parse_date_value("2024-01-15 10:30:00")
        assert result2.year == 2024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
