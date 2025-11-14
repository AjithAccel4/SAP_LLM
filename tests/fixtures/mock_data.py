"""
Mock data generators for SAP_LLM testing.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List


def generate_random_po_number() -> str:
    """Generate random PO number."""
    return f"45{random.randint(10000000, 99999999)}"


def generate_random_invoice_number() -> str:
    """Generate random invoice number."""
    prefix = random.choice(["INV", "SI", "BILL"])
    year = datetime.now().year
    number = random.randint(1000, 9999)
    return f"{prefix}-{year}-{number:04d}"


def generate_random_vendor_id() -> str:
    """Generate random vendor ID."""
    return f"{random.randint(100000, 999999)}"


def generate_random_amount(min_val: float = 100, max_val: float = 10000) -> float:
    """Generate random amount."""
    return round(random.uniform(min_val, max_val), 2)


def generate_random_date(days_back: int = 365) -> str:
    """Generate random date."""
    start_date = datetime.now() - timedelta(days=days_back)
    random_days = random.randint(0, days_back)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")


def generate_mock_adc(doc_type: str = "PURCHASE_ORDER") -> Dict[str, Any]:
    """Generate mock ADC document."""
    base_adc = {
        "document_type": doc_type,
        "document_subtype": random.choice(["STANDARD", "BLANKET", "CONTRACT"]),
        "company_code": random.choice(["1000", "2000", "3000"]),
        "currency": random.choice(["USD", "EUR", "GBP"]),
    }

    if doc_type == "PURCHASE_ORDER":
        subtotal = generate_random_amount(1000, 50000)
        tax_rate = 0.10
        tax_amount = round(subtotal * tax_rate, 2)

        base_adc.update({
            "po_number": generate_random_po_number(),
            "po_date": generate_random_date(90),
            "vendor_id": generate_random_vendor_id(),
            "vendor_name": f"Vendor {random.randint(1, 100)}",
            "subtotal": subtotal,
            "tax_amount": tax_amount,
            "total_amount": subtotal + tax_amount,
            "line_items": [
                {
                    "line_number": i + 1,
                    "material": f"Material {i+1}",
                    "quantity": random.randint(1, 100),
                    "unit_price": generate_random_amount(10, 500),
                }
                for i in range(random.randint(1, 5))
            ],
        })

    elif doc_type == "SUPPLIER_INVOICE":
        subtotal = generate_random_amount(1000, 50000)
        tax_amount = round(subtotal * 0.10, 2)

        base_adc.update({
            "invoice_number": generate_random_invoice_number(),
            "invoice_date": generate_random_date(90),
            "vendor_id": generate_random_vendor_id(),
            "vendor_name": f"Vendor {random.randint(1, 100)}",
            "po_reference": generate_random_po_number(),
            "subtotal": subtotal,
            "tax_amount": tax_amount,
            "total_amount": subtotal + tax_amount,
            "due_date": generate_random_date(-30),  # Future date
        })

    return base_adc


def generate_mock_exception(category: str = "VALIDATION_ERROR") -> Dict[str, Any]:
    """Generate mock exception."""
    return {
        "id": f"exc_{random.randint(1000, 9999)}",
        "category": category,
        "severity": random.choice(["HIGH", "MEDIUM", "LOW"]),
        "field": random.choice(["total_amount", "vendor_id", "po_number", "tax_amount"]),
        "message": f"Test exception for {category}",
        "timestamp": datetime.now().isoformat(),
        "document_id": f"doc_{random.randint(1000, 9999)}",
    }


def generate_mock_pmg_transaction() -> Dict[str, Any]:
    """Generate mock PMG transaction."""
    return {
        "id": f"tx_{random.randint(100000, 999999)}",
        "timestamp": datetime.now().isoformat(),
        "document_type": random.choice(["PURCHASE_ORDER", "SUPPLIER_INVOICE", "SALES_ORDER"]),
        "vendor_id": generate_random_vendor_id(),
        "total_amount": generate_random_amount(),
        "routing": {
            "endpoint": random.choice([
                "API_PURCHASEORDER_PROCESS_SRV",
                "API_SUPPLIERINVOICE_PROCESS_SRV",
                "API_SALES_ORDER_SRV",
            ]),
            "success": random.choice([True, True, True, False]),  # 75% success rate
            "latency_ms": random.randint(50, 500),
        },
        "quality_score": round(random.uniform(0.85, 0.99), 2),
    }


def generate_batch_documents(count: int = 100, doc_type: str = None) -> List[Dict[str, Any]]:
    """Generate batch of mock documents."""
    doc_types = ["PURCHASE_ORDER", "SUPPLIER_INVOICE", "SALES_ORDER"]
    return [
        generate_mock_adc(doc_type or random.choice(doc_types))
        for _ in range(count)
    ]
