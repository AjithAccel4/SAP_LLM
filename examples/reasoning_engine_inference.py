"""
Inference Examples for Reasoning Engine.

Demonstrates:
1. SAP endpoint selection
2. Payload generation
3. Confidence scoring
4. Explanation generation
5. Exception handling
"""

import json
from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def example_1_purchase_order_routing():
    """Example 1: Purchase Order Routing."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Purchase Order Routing")
    print("="*60)

    # Load model
    model = ReasoningEngine.load(
        model_path="./models/reasoning_engine_rlhf/final",
        device="cuda",
        precision="int8",
    )

    # Sample ADC (extracted document data)
    adc_json = {
        "doc_id": "PO_20240115_001",
        "doc_type": "PURCHASE_ORDER",
        "supplier_name": "ACME Corporation",
        "supplier_id": "SUP1234",
        "company_code": "1000",
        "currency": "USD",
        "total_amount": 15750.00,
        "document_date": "2024-01-15",
        "po_number": "PO456789",
        "items": [
            {
                "material": "MAT5678",
                "description": "Laptop Computer",
                "quantity": 10,
                "unit_price": 1500.00,
                "total": 15000.00,
            },
            {
                "material": "MAT5679",
                "description": "Mouse",
                "quantity": 15,
                "unit_price": 50.00,
                "total": 750.00,
            },
        ],
    }

    # API schemas
    api_schemas = [
        {
            "name": "API_PURCHASEORDER_PROCESS_SRV",
            "entity": "A_PurchaseOrder",
            "method": "POST",
        },
        {
            "name": "API_SALES_ORDER_SRV",
            "entity": "A_SalesOrder",
            "method": "POST",
        },
    ]

    # Similar cases from PMG
    similar_cases = [
        {
            "doc_id": "PO_20240110_003",
            "doc_type": "PURCHASE_ORDER",
            "supplier_id": "SUP1234",
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "success": True,
            "confidence": 0.98,
        },
        {
            "doc_id": "PO_20240108_012",
            "doc_type": "PURCHASE_ORDER",
            "supplier_id": "SUP1234",
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "success": True,
            "confidence": 0.96,
        },
    ]

    # Make routing decision
    decision = model.decide_routing(
        adc_json=adc_json,
        doc_type="PURCHASE_ORDER",
        api_schemas=api_schemas,
        similar_cases=similar_cases,
    )

    # Print results
    print("\n📄 Input Document:")
    print(f"  Type: {adc_json['doc_type']}")
    print(f"  Supplier: {adc_json['supplier_name']} ({adc_json['supplier_id']})")
    print(f"  Company Code: {adc_json['company_code']}")
    print(f"  Total Amount: ${adc_json['total_amount']:,.2f} {adc_json['currency']}")

    print("\n🤖 Routing Decision:")
    print(f"  Endpoint: {decision.get('endpoint')}")
    print(f"  Method: {decision.get('method')}")
    print(f"  Confidence: {decision.get('confidence', 0.0):.2%}")

    print("\n💭 Reasoning:")
    print(f"  {decision.get('reasoning', 'No reasoning provided')}")

    print("\n✅ Decision: Route to {0}".format(decision.get('endpoint')))


def example_2_exception_handling():
    """Example 2: Exception Handling."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Exception Handling")
    print("="*60)

    # Load model
    model = ReasoningEngine.load(
        model_path="./models/reasoning_engine_rlhf/final",
        device="cuda",
        precision="int8",
    )

    # Exception case
    exception = {
        "category": "FIELD_VALIDATION",
        "severity": "HIGH",
        "field": "total_amount",
        "expected": "15000.00",
        "value": "15750.00",
        "message": "Total amount mismatch: sum of line items (15000.00) != header total (15750.00)",
    }

    # Similar exceptions
    similar_exceptions = [
        {
            "category": "FIELD_VALIDATION",
            "field": "total_amount",
            "action": "AUTO_CORRECT",
            "resolution": "Recalculate from line items",
            "success": True,
        },
        {
            "category": "FIELD_VALIDATION",
            "field": "total_amount",
            "action": "AUTO_CORRECT",
            "resolution": "Use line item sum",
            "success": True,
        },
    ]

    # Handle exception
    decision = model.handle_exception(
        exception=exception,
        similar_exceptions=similar_exceptions,
    )

    # Print results
    print("\n⚠️  Exception:")
    print(f"  Category: {exception['category']}")
    print(f"  Severity: {exception['severity']}")
    print(f"  Field: {exception['field']}")
    print(f"  Expected: {exception['expected']}")
    print(f"  Actual: {exception['value']}")

    print("\n🤖 Recommended Action:")
    print(f"  Action: {decision.get('action')}")
    print(f"  Confidence: {decision.get('confidence', 0.0):.2%}")

    print("\n💭 Reasoning:")
    print(f"  {decision.get('reasoning', 'No reasoning provided')}")

    if decision.get('correction'):
        print("\n🔧 Suggested Correction:")
        print(f"  {json.dumps(decision.get('correction'), indent=2)}")


def example_3_multi_document_batch():
    """Example 3: Batch Processing Multiple Documents."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)

    # Load model
    model = ReasoningEngine.load(
        model_path="./models/reasoning_engine_rlhf/final",
        device="cuda",
        precision="int8",
    )

    # Batch of documents
    documents = [
        {
            "doc_type": "PURCHASE_ORDER",
            "adc_json": {
                "supplier_id": "SUP1234",
                "company_code": "1000",
                "total_amount": 5000.00,
            },
        },
        {
            "doc_type": "SUPPLIER_INVOICE",
            "adc_json": {
                "supplier_id": "SUP5678",
                "company_code": "2000",
                "total_amount": 12500.00,
            },
        },
        {
            "doc_type": "SALES_ORDER",
            "adc_json": {
                "customer_id": "CUST9012",
                "sales_org": "1000",
                "total_amount": 25000.00,
            },
        },
    ]

    # Process batch
    results = []

    for i, doc in enumerate(documents):
        decision = model.decide_routing(
            adc_json=doc["adc_json"],
            doc_type=doc["doc_type"],
            api_schemas=[],  # Would be loaded from config
            similar_cases=[],
        )

        results.append({
            "doc_index": i,
            "doc_type": doc["doc_type"],
            "endpoint": decision.get("endpoint"),
            "confidence": decision.get("confidence", 0.0),
        })

    # Print results
    print("\n📊 Batch Processing Results:")
    print("\n| # | Document Type | Endpoint | Confidence |")
    print("|---|---------------|----------|------------|")

    for result in results:
        print(f"| {result['doc_index'] + 1} | {result['doc_type']:<13} | {result['endpoint']:<35} | {result['confidence']:.2%} |")

    print(f"\n✅ Processed {len(documents)} documents successfully")


def example_4_confidence_calibration():
    """Example 4: Confidence Scoring and Calibration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Confidence Calibration")
    print("="*60)

    # Load model
    model = ReasoningEngine.load(
        model_path="./models/reasoning_engine_rlhf/final",
        device="cuda",
        precision="int8",
    )

    # Test cases with varying complexity
    test_cases = [
        {
            "name": "Clear Case (High Confidence Expected)",
            "doc_type": "PURCHASE_ORDER",
            "adc_json": {"supplier_id": "SUP1234", "company_code": "1000"},
            "similar_cases": [
                {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "success": True},
                {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "success": True},
                {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "success": True},
            ],
        },
        {
            "name": "Ambiguous Case (Medium Confidence Expected)",
            "doc_type": "PURCHASE_ORDER",
            "adc_json": {"supplier_id": "UNKNOWN", "company_code": "9999"},
            "similar_cases": [
                {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "success": True},
                {"endpoint": "API_SALES_ORDER_SRV", "success": False},
            ],
        },
        {
            "name": "Novel Case (Low Confidence Expected)",
            "doc_type": "CUSTOM_DOCUMENT",
            "adc_json": {"custom_field": "value"},
            "similar_cases": [],
        },
    ]

    print("\n📊 Confidence Calibration Test:\n")

    for test_case in test_cases:
        decision = model.decide_routing(
            adc_json=test_case["adc_json"],
            doc_type=test_case["doc_type"],
            api_schemas=[],
            similar_cases=test_case["similar_cases"],
        )

        confidence = decision.get("confidence", 0.0)

        print(f"{test_case['name']}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Endpoint: {decision.get('endpoint')}")
        print()


def main():
    """Run all examples."""
    print("\n" + "🤖 " + "="*58)
    print("  REASONING ENGINE INFERENCE EXAMPLES")
    print("="*60 + "\n")

    try:
        example_1_purchase_order_routing()
        example_2_exception_handling()
        example_3_multi_document_batch()
        example_4_confidence_calibration()

        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print("\n" + "="*60)
        print("❌ EXAMPLES FAILED")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
