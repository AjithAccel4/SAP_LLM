"""
Example Usage of Language Decoder with LoRA and Constrained Decoding.

Demonstrates:
1. Loading trained model
2. Extracting fields from documents
3. Using constrained decoding for JSON compliance
4. Evaluating extraction quality
"""

import json
from pathlib import Path

import torch

from sap_llm.models.language_decoder_with_lora import LanguageDecoderWithLoRA
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def example_invoice_extraction():
    """Example: Extract fields from invoice."""
    logger.info("=" * 80)
    logger.info("Example 1: Invoice Field Extraction")
    logger.info("=" * 80)

    # Load model
    logger.info("\nLoading model...")
    model = LanguageDecoderWithLoRA(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="fp16",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        enable_fsm=True,
    )

    # Define schema
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {
                "type": "string",
                "description": "Invoice number or ID",
            },
            "invoice_date": {
                "type": "string",
                "description": "Invoice date (YYYY-MM-DD)",
            },
            "vendor_name": {
                "type": "string",
                "description": "Vendor or supplier name",
            },
            "vendor_address": {
                "type": "string",
                "description": "Vendor address",
            },
            "total_amount": {
                "type": "number",
                "description": "Total amount",
            },
            "currency": {
                "type": "string",
                "description": "Currency code (e.g., USD, EUR)",
            },
            "payment_terms": {
                "type": "string",
                "description": "Payment terms",
            },
            "line_items": {
                "type": "array",
                "description": "Line items",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "amount": {"type": "number"},
                    },
                },
            },
        },
        "required": ["invoice_number", "invoice_date", "vendor_name", "total_amount"],
    }

    # Sample OCR text
    ocr_text = """
    INVOICE

    Acme Corporation
    123 Business Ave
    New York, NY 10001

    INVOICE NO: INV-2024-001
    DATE: January 15, 2024

    BILL TO:
    Customer Corp
    456 Client Street
    Los Angeles, CA 90001

    ITEMS:
    Description                 Qty     Unit Price      Amount
    Widget A                    10      $25.00          $250.00
    Service B                   5       $100.00         $500.00
    Product C                   2       $150.00         $300.00

    SUBTOTAL:                                           $1,050.00
    TAX (10%):                                          $105.00
    TOTAL:                                              $1,155.00

    Payment Terms: Net 30
    """

    # Extract fields
    logger.info("\nExtracting fields...")
    extracted_data = model.extract_fields(
        ocr_text=ocr_text,
        doc_type="invoice",
        schema=invoice_schema,
        use_self_correction=True,
    )

    # Display results
    logger.info("\n--- Extracted Data ---")
    logger.info(json.dumps(extracted_data, indent=2))

    # Validate compliance
    logger.info("\n--- Validation ---")
    try:
        from jsonschema import validate

        validate(instance=extracted_data, schema=invoice_schema)
        logger.info("✓ Schema validation passed")
    except Exception as e:
        logger.error(f"✗ Schema validation failed: {e}")

    # Check required fields
    required_fields = invoice_schema["required"]
    missing_fields = [f for f in required_fields if f not in extracted_data]

    if not missing_fields:
        logger.info("✓ All required fields present")
    else:
        logger.warning(f"✗ Missing required fields: {missing_fields}")

    logger.info("\n" + "=" * 80)


def example_purchase_order_extraction():
    """Example: Extract fields from purchase order."""
    logger.info("=" * 80)
    logger.info("Example 2: Purchase Order Field Extraction")
    logger.info("=" * 80)

    # Load model (reuse from previous example in production)
    model = LanguageDecoderWithLoRA(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="fp16",
        use_lora=True,
        enable_fsm=True,
    )

    # Define schema
    po_schema = {
        "type": "object",
        "properties": {
            "po_number": {
                "type": "string",
                "description": "Purchase order number",
            },
            "po_date": {
                "type": "string",
                "description": "Purchase order date",
            },
            "vendor_name": {
                "type": "string",
                "description": "Vendor name",
            },
            "delivery_date": {
                "type": "string",
                "description": "Expected delivery date",
            },
            "shipping_address": {
                "type": "string",
                "description": "Shipping address",
            },
            "total_amount": {
                "type": "number",
                "description": "Total amount",
            },
        },
        "required": ["po_number", "po_date", "vendor_name"],
    }

    # Sample OCR text
    ocr_text = """
    PURCHASE ORDER

    PO Number: PO-2024-5678
    Date: 2024-01-20

    Vendor:
    Tech Supplies Inc.
    789 Vendor Road
    Chicago, IL 60601

    Ship To:
    Our Company Warehouse
    321 Storage Blvd
    Dallas, TX 75201

    Expected Delivery: 2024-02-01

    Items:
    - Laptop Computers (10 units) @ $1,200.00 = $12,000.00
    - Office Chairs (20 units) @ $250.00 = $5,000.00

    Total: $17,000.00

    Authorized by: John Smith
    """

    # Extract fields
    logger.info("\nExtracting fields...")
    extracted_data = model.extract_fields(
        ocr_text=ocr_text,
        doc_type="purchase_order",
        schema=po_schema,
        use_self_correction=True,
    )

    # Display results
    logger.info("\n--- Extracted Data ---")
    logger.info(json.dumps(extracted_data, indent=2))

    logger.info("\n" + "=" * 80)


def example_with_vision_features():
    """Example: Extract fields with vision features."""
    logger.info("=" * 80)
    logger.info("Example 3: Extraction with Vision Features")
    logger.info("=" * 80)

    # Load model
    model = LanguageDecoderWithLoRA(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="fp16",
        use_lora=True,
        enable_fsm=True,
        vision_hidden_size=768,  # LayoutLMv3 hidden size
    )

    # Simulate vision features (in production, from LayoutLMv3)
    # Shape: [batch_size=1, num_patches=196, hidden_size=768]
    vision_features = torch.randn(1, 196, 768)

    if torch.cuda.is_available():
        vision_features = vision_features.cuda()

    # Define schema
    receipt_schema = {
        "type": "object",
        "properties": {
            "merchant_name": {"type": "string"},
            "transaction_date": {"type": "string"},
            "total": {"type": "number"},
            "tax": {"type": "number"},
        },
        "required": ["merchant_name", "total"],
    }

    # Sample OCR text
    ocr_text = """
    Coffee Shop
    123 Main Street

    Date: 2024-01-15 10:30 AM

    Latte          $4.50
    Croissant      $3.25

    Subtotal:      $7.75
    Tax:           $0.62
    Total:         $8.37

    Thank you!
    """

    # Extract with vision features
    logger.info("\nExtracting fields with vision features...")

    # Create prompt
    prompt = model._create_extraction_prompt(ocr_text, "receipt", receipt_schema)

    # Generate with vision features
    generated_json = model.generate_with_constraints(
        prompt=prompt,
        schema=receipt_schema,
        vision_features=vision_features,
        max_new_tokens=512,
    )

    # Parse result
    try:
        extracted_data = json.loads(generated_json)
        logger.info("\n--- Extracted Data ---")
        logger.info(json.dumps(extracted_data, indent=2))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Generated: {generated_json}")

    logger.info("\n" + "=" * 80)


def example_constrained_generation():
    """Example: Demonstrate FSM-based constrained generation."""
    logger.info("=" * 80)
    logger.info("Example 4: FSM-Based Constrained Generation")
    logger.info("=" * 80)

    # Load model with FSM enabled
    model = LanguageDecoderWithLoRA(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="fp16",
        enable_fsm=True,  # Enable FSM
    )

    # Simple schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"},
        },
        "required": ["name", "age"],
    }

    # Simple prompt
    prompt = """
    Extract person information:

    Name: Alice Johnson
    Age: 30
    Email: alice@example.com

    Output JSON:
    """

    # Generate with FSM constraints
    logger.info("\nGenerating with FSM constraints...")
    generated_json = model.generate_with_constraints(
        prompt=prompt,
        schema=schema,
        max_new_tokens=256,
        temperature=0.0,  # Deterministic
    )

    logger.info("\n--- Generated JSON ---")
    logger.info(generated_json)

    # Verify it's valid JSON
    try:
        data = json.loads(generated_json)
        logger.info("\n✓ Valid JSON generated")
        logger.info(f"Parsed: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"\n✗ Invalid JSON: {e}")

    logger.info("\n" + "=" * 80)


def main():
    """Run all examples."""
    logger.info("\n\n")
    logger.info("*" * 80)
    logger.info("Language Decoder with LoRA - Example Usage")
    logger.info("*" * 80)

    # Run examples
    try:
        example_invoice_extraction()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    try:
        example_purchase_order_extraction()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    try:
        example_with_vision_features()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")

    try:
        example_constrained_generation()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")

    logger.info("\n\n" + "*" * 80)
    logger.info("All examples complete!")
    logger.info("*" * 80)


if __name__ == "__main__":
    main()
