"""
Sample test documents for SAP_LLM testing.
"""

import json
from typing import Dict, Any, List
from PIL import Image
import numpy as np


def create_sample_purchase_order() -> Dict[str, Any]:
    """Create sample purchase order ADC."""
    return {
        "document_type": "PURCHASE_ORDER",
        "document_subtype": "STANDARD",
        "po_number": "4500123456",
        "po_date": "2024-01-15",
        "vendor_id": "100001",
        "vendor_name": "ACME Corp",
        "company_code": "1000",
        "purchasing_org": "1000",
        "purchasing_group": "001",
        "currency": "USD",
        "payment_terms": "NET30",
        "incoterms": "FOB",
        "delivery_date": "2024-02-15",
        "subtotal": 2000.00,
        "tax_amount": 200.00,
        "total_amount": 2200.00,
        "line_items": [
            {
                "line_number": 1,
                "material": "Widget A",
                "material_number": "MAT001",
                "quantity": 100,
                "unit": "EA",
                "unit_price": 10.00,
                "total": 1000.00,
                "delivery_date": "2024-02-15",
            },
            {
                "line_number": 2,
                "material": "Widget B",
                "material_number": "MAT002",
                "quantity": 50,
                "unit": "EA",
                "unit_price": 20.00,
                "total": 1000.00,
                "delivery_date": "2024-02-15",
            },
        ],
    }


def create_sample_supplier_invoice() -> Dict[str, Any]:
    """Create sample supplier invoice ADC."""
    return {
        "document_type": "SUPPLIER_INVOICE",
        "document_subtype": "STANDARD",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-20",
        "vendor_id": "100001",
        "vendor_name": "ACME Corp",
        "po_reference": "4500123456",
        "company_code": "1000",
        "currency": "USD",
        "payment_terms": "NET30",
        "due_date": "2024-02-20",
        "subtotal": 2200.00,
        "tax_amount": 220.00,
        "total_amount": 2420.00,
        "bank_details": {
            "bank_name": "Test Bank",
            "account_number": "12345678",
            "routing_number": "987654321",
        },
    }


def create_sample_sales_order() -> Dict[str, Any]:
    """Create sample sales order ADC."""
    return {
        "document_type": "SALES_ORDER",
        "document_subtype": "STANDARD",
        "order_number": "SO-2024-001",
        "order_date": "2024-01-10",
        "customer_id": "200001",
        "customer_name": "Customer Corp",
        "sales_org": "1000",
        "distribution_channel": "10",
        "division": "00",
        "currency": "USD",
        "payment_terms": "NET30",
        "incoterms": "CIF",
        "requested_delivery_date": "2024-02-10",
        "subtotal": 5000.00,
        "tax_amount": 500.00,
        "total_amount": 5500.00,
        "line_items": [
            {
                "line_number": 1,
                "material": "Product X",
                "material_number": "PROD001",
                "quantity": 100,
                "unit": "EA",
                "unit_price": 50.00,
                "total": 5000.00,
            },
        ],
    }


def create_sample_document_image(doc_type: str = "PURCHASE_ORDER") -> Image.Image:
    """Create sample document image with realistic patterns."""
    # Create white background
    img_array = np.ones((1200, 800, 3), dtype=np.uint8) * 255

    # Add header text area (simulate dark text)
    img_array[50:100, 50:750] = 0

    # Add document type indicator
    if doc_type == "PURCHASE_ORDER":
        img_array[120:150, 50:300] = 0
    elif doc_type == "SUPPLIER_INVOICE":
        img_array[120:150, 50:350] = 0

    # Add table-like structure (simulate line items)
    for i in range(5):
        y = 200 + i * 80
        # Row separator
        img_array[y:y+2, 50:750] = 200
        # Column separators
        img_array[y:y+70, 200:202] = 200
        img_array[y:y+70, 400:402] = 200
        img_array[y:y+70, 600:602] = 200

    return Image.fromarray(img_array)


def create_sample_ocr_output(doc_type: str = "PURCHASE_ORDER") -> Dict[str, Any]:
    """Create sample OCR output."""
    if doc_type == "PURCHASE_ORDER":
        text = """
        PURCHASE ORDER

        PO Number: 4500123456
        Date: 2024-01-15
        Vendor: ACME Corp
        Vendor ID: 100001

        Line Items:
        1. Widget A - Qty: 100 - Price: $10.00 - Total: $1,000.00
        2. Widget B - Qty: 50 - Price: $20.00 - Total: $1,000.00

        Subtotal: $2,000.00
        Tax (10%): $200.00
        Total: $2,200.00
        """
        words = [
            "PURCHASE", "ORDER", "PO", "Number:", "4500123456",
            "Date:", "2024-01-15", "Vendor:", "ACME", "Corp",
        ]
        boxes = [
            [50, 50, 150, 80],
            [160, 50, 220, 80],
            [50, 120, 80, 140],
            [85, 120, 150, 140],
            [155, 120, 280, 140],
        ] + [[0, 0, 0, 0]] * (len(words) - 5)  # Dummy boxes for remaining words

    elif doc_type == "SUPPLIER_INVOICE":
        text = """
        INVOICE

        Invoice Number: INV-2024-001
        Invoice Date: 2024-01-20
        Vendor: ACME Corp
        PO Reference: 4500123456

        Amount Due: $2,420.00
        Due Date: 2024-02-20
        """
        words = [
            "INVOICE", "Invoice", "Number:", "INV-2024-001",
            "Invoice", "Date:", "2024-01-20", "Vendor:", "ACME", "Corp",
        ]
        boxes = [[50, 50, 150, 80]] * len(words)

    else:
        text = f"SAMPLE {doc_type}"
        words = ["SAMPLE", doc_type]
        boxes = [[50, 50, 150, 80], [160, 50, 250, 80]]

    return {
        "text": text,
        "words": words,
        "boxes": boxes,
    }


def create_sample_exception_cluster() -> Dict[str, Any]:
    """Create sample exception cluster for SHWL testing."""
    return {
        "id": "cluster_001",
        "label": 0,
        "size": 25,
        "category": "VALIDATION_ERROR",
        "severity": "HIGH",
        "field": "total_amount",
        "pattern": "Total amount does not match subtotal + tax",
        "exceptions": [
            {
                "id": f"exc_{i:03d}",
                "category": "VALIDATION_ERROR",
                "severity": "HIGH",
                "field": "total_amount",
                "message": "Total amount does not match subtotal + tax",
                "expected": "2200.00",
                "actual": "2000.00",
                "timestamp": f"2024-01-15T{10+i//6:02d}:{(i*10)%60:02d}:00Z",
                "document_id": f"doc_{i:04d}",
            }
            for i in range(25)
        ],
    }


def create_sample_api_schemas() -> List[Dict[str, Any]]:
    """Create sample SAP API schemas."""
    return [
        {
            "name": "API_PURCHASEORDER_PROCESS_SRV",
            "version": "0001",
            "description": "Purchase Order Processing",
            "entities": ["PurchaseOrder", "PurchaseOrderItem"],
            "operations": ["POST", "GET", "PATCH"],
            "fields": {
                "PurchaseOrder": [
                    "PurchaseOrder",
                    "CompanyCode",
                    "PurchasingOrganization",
                    "PurchasingGroup",
                    "Supplier",
                    "PurchaseOrderDate",
                ],
                "PurchaseOrderItem": [
                    "PurchaseOrderItem",
                    "Material",
                    "OrderQuantity",
                    "NetPriceAmount",
                ],
            },
        },
        {
            "name": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "version": "0001",
            "description": "Supplier Invoice Processing",
            "entities": ["SupplierInvoice"],
            "operations": ["POST", "GET"],
            "fields": {
                "SupplierInvoice": [
                    "SupplierInvoice",
                    "InvoiceDate",
                    "CompanyCode",
                    "Supplier",
                    "DocumentCurrency",
                    "InvoiceGrossAmount",
                ],
            },
        },
        {
            "name": "API_SALES_ORDER_SRV",
            "version": "0001",
            "description": "Sales Order Processing",
            "entities": ["SalesOrder", "SalesOrderItem"],
            "operations": ["POST", "GET", "PATCH", "DELETE"],
            "fields": {
                "SalesOrder": [
                    "SalesOrder",
                    "SalesOrganization",
                    "DistributionChannel",
                    "SoldToParty",
                    "TotalNetAmount",
                ],
            },
        },
    ]


def create_sample_business_rules() -> List[Dict[str, Any]]:
    """Create sample business rules."""
    return [
        {
            "id": "rule_001",
            "name": "Total Amount Validation",
            "description": "Total amount must equal subtotal + tax",
            "category": "VALIDATION",
            "document_types": ["PURCHASE_ORDER", "SUPPLIER_INVOICE"],
            "condition": "total_amount == subtotal + tax_amount",
            "tolerance": 0.02,  # 2%
            "severity": "HIGH",
            "action": "ESCALATE",
        },
        {
            "id": "rule_002",
            "name": "Vendor ID Required",
            "description": "Vendor ID must be present",
            "category": "REQUIRED_FIELD",
            "document_types": ["PURCHASE_ORDER", "SUPPLIER_INVOICE"],
            "condition": "vendor_id is not null and vendor_id != ''",
            "severity": "HIGH",
            "action": "REJECT",
        },
        {
            "id": "rule_003",
            "name": "Line Item Total Validation",
            "description": "Sum of line item totals must match subtotal",
            "category": "VALIDATION",
            "document_types": ["PURCHASE_ORDER", "SALES_ORDER"],
            "condition": "sum(line_items.total) == subtotal",
            "tolerance": 0.01,  # 1%
            "severity": "MEDIUM",
            "action": "ESCALATE",
        },
    ]
