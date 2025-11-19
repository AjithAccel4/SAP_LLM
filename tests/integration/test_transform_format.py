"""
Integration tests for transform_format function

Tests the complete transformation pipeline including:
- KnowledgeBaseQuery.transform_format method
- Integration with FieldMappingManager
- End-to-end document transformations
- Multiple document types
"""

import pytest
from datetime import datetime

from sap_llm.knowledge_base.query import KnowledgeBaseQuery
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage


class TestTransformFormat:
    """Integration tests for transform_format method."""

    @pytest.fixture
    def knowledge_base_query(self, tmp_path):
        """Create a KnowledgeBaseQuery instance."""
        storage = KnowledgeBaseStorage(str(tmp_path / "test_kb.db"))
        return KnowledgeBaseQuery(storage)

    @pytest.fixture
    def purchase_order_data(self):
        """Sample purchase order data."""
        return {
            "po_number": "4500012345",
            "po_date": "2024-01-15",
            "vendor_id": "VENDOR001",
            "vendor_name": "Acme Corporation",
            "company_code": "1000",
            "purchasing_organization": "1000",
            "purchasing_group": "001",
            "currency": "USD",
            "total_amount": "$10,500.00",
            "payment_terms": "NET30",
            "items": [
                {
                    "item_number": "10",
                    "material_number": "MAT-12345",
                    "description": "Office Supplies",
                    "quantity": "100",
                    "unit_of_measure": "EA",
                    "unit_price": "105.00"
                },
                {
                    "item_number": "20",
                    "material_number": "MAT-67890",
                    "description": "Paper Products",
                    "quantity": "50",
                    "unit_price": "50.00"
                }
            ]
        }

    @pytest.fixture
    def supplier_invoice_data(self):
        """Sample supplier invoice data."""
        return {
            "invoice_number": "INV-2024-0001",
            "invoice_date": "2024-02-15",
            "posting_date": "2024-02-16",
            "vendor_id": "VENDOR001",
            "company_code": "1000",
            "currency": "EUR",
            "total_amount": "5,250.00",
            "tax_amount": "997.50",
            "purchase_order": "4500012345",
            "fiscal_year": "2024",
            "payment_terms": "NET30",
            "items": [
                {
                    "item_number": "1",
                    "purchase_order": "4500012345",
                    "po_item": "10",
                    "quantity": "100",
                    "unit_price": "52.50"
                }
            ]
        }

    @pytest.fixture
    def goods_receipt_data(self):
        """Sample goods receipt data."""
        return {
            "posting_date": "2024-01-20",
            "purchase_order": "4500012345",
            "delivery_note": "DN-2024-001",
            "items": [
                {
                    "item_number": "1",
                    "material_number": "MAT-12345",
                    "plant": "1000",
                    "storage_location": "0001",
                    "quantity": "100",
                    "unit_of_measure": "EA",
                    "purchase_order": "4500012345",
                    "po_item": "10"
                }
            ]
        }

    def test_transform_purchase_order_standard(self, knowledge_base_query, purchase_order_data):
        """Test transforming a standard purchase order."""
        result = knowledge_base_query.transform_format(
            purchase_order_data,
            "PURCHASE_ORDER",
            "SAP_API"
        )

        # Verify header fields
        assert "PurchaseOrder" in result
        assert result["PurchaseOrder"] == "4500012345"

        assert "PurchaseOrderDate" in result
        assert result["PurchaseOrderDate"] == "20240115"

        assert "Supplier" in result
        assert result["Supplier"] == "0VENDOR001"  # Padded to 10 chars

        assert "CompanyCode" in result
        assert result["CompanyCode"] == "1000"

        assert "DocumentCurrency" in result
        assert result["DocumentCurrency"] == "USD"

        assert "TotalAmount" in result
        assert result["TotalAmount"] == 10500.0

        # Verify items
        assert "to_PurchaseOrderItem" in result
        assert len(result["to_PurchaseOrderItem"]) == 2

        item1 = result["to_PurchaseOrderItem"][0]
        assert item1["PurchaseOrderItem"] == "00010"
        assert item1["Material"] == "MAT-12345"
        assert item1["OrderQuantity"] == 100.0

    def test_transform_purchase_order_service(self, knowledge_base_query):
        """Test transforming a service purchase order."""
        service_po_data = {
            "po_number": "4500067890",
            "po_date": "2024-03-01",
            "vendor_id": "SERV001",
            "company_code": "1000",
            "purchasing_organization": "1000",
            "purchasing_group": "002",
            "currency": "USD",
            "po_type": "FO",
            "services": [
                {
                    "item_number": "10",
                    "description": "Consulting Services",
                    "quantity": "40",
                    "unit_of_measure": "HR",
                    "unit_price": "150.00"
                }
            ]
        }

        result = knowledge_base_query.transform_format(
            service_po_data,
            "PURCHASE_ORDER_SERVICE",
            "SAP_API"
        )

        assert result["PurchaseOrder"] == "4500067890"
        assert result["PurchaseOrderType"] == "FO"

        # Service items should be in the result
        assert "to_PurchaseOrderItem" in result
        assert len(result["to_PurchaseOrderItem"]) == 1

    def test_transform_supplier_invoice(self, knowledge_base_query, supplier_invoice_data):
        """Test transforming a supplier invoice."""
        result = knowledge_base_query.transform_format(
            supplier_invoice_data,
            "SUPPLIER_INVOICE",
            "SAP_API"
        )

        # Verify invoice fields
        assert "SupplierInvoiceIDByInvcgParty" in result
        assert result["SupplierInvoiceIDByInvcgParty"] == "INV-2024-0001"

        assert "DocumentDate" in result
        assert result["DocumentDate"] == "20240215"

        assert "PostingDate" in result
        assert result["PostingDate"] == "20240216"

        assert "InvoicingParty" in result
        assert result["InvoicingParty"] == "0VENDOR001"

        assert "InvoiceGrossAmount" in result
        assert result["InvoiceGrossAmount"] == 5250.0

        assert "TaxAmount" in result
        assert result["TaxAmount"] == 997.5

        assert "FiscalYear" in result
        assert result["FiscalYear"] == "2024"

        # Verify items
        assert "to_SuplrInvcItemPurOrdRef" in result
        assert len(result["to_SuplrInvcItemPurOrdRef"]) == 1

    def test_transform_credit_memo(self, knowledge_base_query):
        """Test transforming a credit memo."""
        credit_memo_data = {
            "credit_memo_number": "CM-2024-001",
            "document_date": "2024-02-20",
            "posting_date": "2024-02-20",
            "vendor_id": "VENDOR001",
            "company_code": "1000",
            "currency": "USD",
            "credit_amount": "500.00",
            "fiscal_year": "2024",
            "reference_invoice": "INV-2024-0001"
        }

        result = knowledge_base_query.transform_format(
            credit_memo_data,
            "SUPPLIER_INVOICE_CREDIT_MEMO",
            "SAP_API"
        )

        assert "SupplierInvoiceIDByInvcgParty" in result
        assert result["SupplierInvoiceIDByInvcgParty"] == "CM-2024-001"

        # Credit amount should be negative
        assert "InvoiceGrossAmount" in result
        assert result["InvoiceGrossAmount"] < 0

    def test_transform_goods_receipt(self, knowledge_base_query, goods_receipt_data):
        """Test transforming a goods receipt."""
        result = knowledge_base_query.transform_format(
            goods_receipt_data,
            "GOODS_RECEIPT_FOR_PO",
            "SAP_API"
        )

        # Verify header fields
        assert "PostingDate" in result
        assert result["PostingDate"] == "20240120"

        # Verify items
        assert "to_MaterialDocumentItem" in result
        assert len(result["to_MaterialDocumentItem"]) == 1

        item = result["to_MaterialDocumentItem"][0]
        assert item["Material"] == "MAT-12345"
        assert item["Plant"] == "1000"
        assert item["QuantityInEntryUnit"] == 100.0

    def test_transform_service_entry_sheet(self, knowledge_base_query):
        """Test transforming a service entry sheet."""
        ses_data = {
            "purchase_order": "4500067890",
            "entry_date": "2024-03-05",
            "performer": "EMP001",
            "services": [
                {
                    "item_number": "1",
                    "purchase_order": "4500067890",
                    "po_item": "10",
                    "description": "Consulting Services",
                    "quantity": "40",
                    "unit_of_measure": "HR"
                }
            ]
        }

        result = knowledge_base_query.transform_format(
            ses_data,
            "SERVICE_ENTRY_SHEET_FOR_PO",
            "SAP_API"
        )

        assert "PurchaseOrder" in result
        assert result["PurchaseOrder"] == "4500067890"

        assert "ServiceEntrySheetDate" in result
        assert result["ServiceEntrySheetDate"] == "20240305"

        # Verify services
        assert "to_ServiceEntrySheetItem" in result
        assert len(result["to_ServiceEntrySheetItem"]) == 1

    def test_transform_payment_terms(self, knowledge_base_query):
        """Test transforming payment terms."""
        payment_terms_data = {
            "payment_terms": "Z030",
            "description": "Net 30 Days",
            "cash_discount_1_days": "10",
            "cash_discount_1_percent": "2.0",
            "cash_discount_2_days": "20",
            "cash_discount_2_percent": "1.0",
            "net_payment_days": "30"
        }

        result = knowledge_base_query.transform_format(
            payment_terms_data,
            "PAYMENT_TERMS",
            "SAP_API"
        )

        assert "PaymentTerms" in result
        assert result["PaymentTerms"] == "Z030"

        assert "CashDiscount1Days" in result
        assert result["CashDiscount1Days"] == 10

        assert "CashDiscount1Percent" in result
        assert result["CashDiscount1Percent"] == 2.0

        assert "NetPaymentDays" in result
        assert result["NetPaymentDays"] == 30

    def test_transform_incoterms(self, knowledge_base_query):
        """Test transforming incoterms."""
        incoterms_data = {
            "incoterms": "fob",
            "incoterms_location": "New York Port",
            "incoterms_version": "2020"
        }

        result = knowledge_base_query.transform_format(
            incoterms_data,
            "INCOTERMS",
            "SAP_API"
        )

        assert "IncotermsClassification" in result
        assert result["IncotermsClassification"] == "FOB"

        assert "IncotermsLocation1" in result
        assert result["IncotermsLocation1"] == "New York Port"

        assert "IncotermsVersion" in result
        assert result["IncotermsVersion"] == "2020"

    def test_parse_format_identifier_explicit(self, knowledge_base_query):
        """Test parsing explicit format identifiers."""
        doc_type, subtype = knowledge_base_query._parse_format_identifier(
            "PurchaseOrder:Service"
        )

        assert doc_type == "PurchaseOrder"
        assert subtype == "Service"

    def test_parse_format_identifier_uppercase(self, knowledge_base_query):
        """Test parsing uppercase format identifiers."""
        doc_type, subtype = knowledge_base_query._parse_format_identifier(
            "PURCHASE_ORDER"
        )

        assert doc_type == "PurchaseOrder"
        assert subtype == "Standard"

    def test_parse_format_identifier_with_subtype(self, knowledge_base_query):
        """Test parsing format identifiers with subtype."""
        doc_type, subtype = knowledge_base_query._parse_format_identifier(
            "SUPPLIER_INVOICE_CREDIT_MEMO"
        )

        assert doc_type == "SupplierInvoice"
        assert subtype == "CreditMemo"

    def test_multiple_document_types_in_sequence(self, knowledge_base_query):
        """Test transforming multiple document types in sequence."""
        # Transform PO
        po_data = {
            "po_number": "PO001",
            "po_date": "2024-01-01",
            "vendor_id": "V001",
            "company_code": "1000",
            "purchasing_organization": "1000",
            "purchasing_group": "001",
            "currency": "USD"
        }

        po_result = knowledge_base_query.transform_format(
            po_data, "PURCHASE_ORDER", "SAP_API"
        )

        assert "PurchaseOrder" in po_result

        # Transform Invoice
        inv_data = {
            "invoice_number": "INV001",
            "invoice_date": "2024-01-15",
            "posting_date": "2024-01-15",
            "vendor_id": "V001",
            "company_code": "1000",
            "currency": "USD",
            "total_amount": "1000.00",
            "fiscal_year": "2024"
        }

        inv_result = knowledge_base_query.transform_format(
            inv_data, "SUPPLIER_INVOICE", "SAP_API"
        )

        assert "SupplierInvoiceIDByInvcgParty" in inv_result

        # Transform GR
        gr_data = {
            "posting_date": "2024-01-20",
            "purchase_order": "PO001",
            "items": []
        }

        gr_result = knowledge_base_query.transform_format(
            gr_data, "GOODS_RECEIPT", "SAP_API"
        )

        assert "PostingDate" in gr_result

    def test_edge_case_empty_data(self, knowledge_base_query):
        """Test transformation with empty data."""
        result = knowledge_base_query.transform_format(
            {},
            "PURCHASE_ORDER",
            "SAP_API"
        )

        # Should return transformed data with defaults
        assert isinstance(result, dict)

    def test_edge_case_partial_data(self, knowledge_base_query):
        """Test transformation with partial data."""
        partial_data = {
            "po_number": "PO123",
            "currency": "usd"
        }

        result = knowledge_base_query.transform_format(
            partial_data,
            "PURCHASE_ORDER",
            "SAP_API"
        )

        assert "PurchaseOrder" in result
        assert result["PurchaseOrder"] == "PO123"

        assert "DocumentCurrency" in result
        assert result["DocumentCurrency"] == "USD"

    def test_special_characters_in_data(self, knowledge_base_query):
        """Test handling of special characters."""
        data_with_special_chars = {
            "po_number": "PO-123/456",
            "po_date": "2024-01-15",
            "vendor_id": "V&001",
            "company_code": "1000",
            "purchasing_organization": "1000",
            "purchasing_group": "001",
            "currency": "USD"
        }

        result = knowledge_base_query.transform_format(
            data_with_special_chars,
            "PURCHASE_ORDER",
            "SAP_API"
        )

        # Should handle special characters without errors
        assert "PurchaseOrder" in result

    def test_fallback_to_legacy_mapping(self, knowledge_base_query):
        """Test fallback to legacy hardcoded mappings."""
        # Use a format that might not have a JSON mapping
        result = knowledge_base_query.transform_format(
            {"test": "value"},
            "UNKNOWN_FORMAT",
            "SAP_API"
        )

        # Should not crash and return some result
        assert isinstance(result, dict)
