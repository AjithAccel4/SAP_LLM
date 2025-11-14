# Hierarchical Type Identifier Implementation

## Overview

This document describes the implementation of the hierarchical classifier for SAP document subtype identification in `/home/user/SAP_LLM/sap_llm/stages/type_identifier.py`.

## Implementation Summary

### TODO Resolution
**Line 79: ✅ COMPLETED** - Implemented actual hierarchical classifier

### What Was Implemented

1. **Comprehensive Subtype Hierarchy (63 subtypes across 15 document types)**
   - PURCHASE_ORDER (10 subtypes)
   - SUPPLIER_INVOICE (8 subtypes)
   - SALES_ORDER (6 subtypes)
   - CUSTOMER_INVOICE (6 subtypes)
   - GOODS_RECEIPT (4 subtypes)
   - DELIVERY_NOTE (4 subtypes)
   - STATEMENT_OF_ACCOUNT (4 subtypes)
   - ADVANCED_SHIPPING_NOTICE (3 subtypes)
   - CREDIT_NOTE (3 subtypes)
   - DEBIT_NOTE (3 subtypes)
   - PAYMENT_ADVICE (3 subtypes)
   - QUOTE (3 subtypes)
   - CONTRACT (3 subtypes)
   - REMITTANCE_ADVICE (2 subtypes)
   - OTHER (1 subtype)

2. **Vision Encoder Integration**
   - Uses LayoutLMv3-based VisionEncoder for feature extraction
   - Processes document images with OCR tokens and bounding boxes
   - Extracts 768-dimensional visual-text features
   - Mean pooling over all tokens for document-level representation

3. **Hierarchical Classification Architecture**
   - Level 1: Document type (from Stage 3 Classification)
   - Level 2: Document subtype (this stage)
   - Separate classifier head for each document type
   - Each classifier: 768-dim → 384-dim (ReLU + Dropout) → num_subtypes

4. **Multi-Label Classification Support**
   - Configurable via `enable_multi_label` parameter
   - Threshold-based multi-label prediction
   - Documents can have multiple subtypes (e.g., CONTRACT + SERVICE)
   - Returns both primary subtype and all predicted subtypes

5. **Confidence Scoring**
   - Softmax probabilities for all subtypes
   - Configurable confidence threshold (default: 0.75)
   - Returns confidence score for primary prediction
   - Full probability distribution in `subtype_scores`

6. **Model Persistence**
   - `save_classifiers()`: Save trained models to disk
   - `load_classifiers()`: Load trained models from disk
   - Saves model weights, subtypes, and architecture info

## Architecture Details

### Class Structure

```python
class TypeIdentifierStage(BaseStage):
    """
    Document subtype identification using hierarchical classification.

    Attributes:
        vision_encoder: LayoutLMv3-based encoder for feature extraction
        classifiers: Dict of classifier heads (one per document type)
        device: cuda/cpu
        confidence_threshold: Threshold for multi-label classification
        enable_multi_label: Enable multi-label predictions
    """
```

### Key Methods

1. **`_load_models()`**
   - Lazy loads vision encoder in feature extraction mode
   - Initializes hierarchical classifiers for all document types

2. **`_initialize_classifiers()`**
   - Creates separate classifier head for each document type
   - Architecture: Linear(768, 384) → ReLU → Dropout(0.1) → Linear(384, num_subtypes)

3. **`process(input_data)`**
   - Main processing method
   - Extracts features using vision encoder
   - Performs hierarchical classification
   - Returns results (single or multi-label)

4. **`_extract_features(image, words, boxes)`**
   - Uses vision encoder to extract visual-text features
   - Returns 768-dim feature vector via mean pooling

5. **`_classify_subtype(doc_type, features)`**
   - Runs appropriate classifier head based on doc_type
   - Returns softmax probabilities for all subtypes

6. **`_single_label_results(doc_type, subtype_probs)`**
   - Returns top prediction with confidence score

7. **`_multi_label_results(doc_type, subtype_probs, threshold)`**
   - Returns all predictions above threshold
   - Handles case where no predictions meet threshold

8. **`get_total_subtypes()`**
   - Returns total number of subtypes (63)

9. **`get_hierarchy_info()`**
   - Returns comprehensive hierarchy statistics

10. **`save_classifiers(output_dir)` / `load_classifiers(input_dir)`**
    - Model persistence for trained classifiers

## Input/Output Format

### Input
```python
{
    "doc_type": str,              # From Stage 3 (e.g., "PURCHASE_ORDER")
    "enhanced_images": List[Image],  # Preprocessed images
    "ocr_results": List[Dict],    # OCR tokens and bounding boxes
}
```

### Output
```python
{
    "subtype": str,                    # Primary subtype (e.g., "CONTRACT")
    "subtypes": List[str],             # All predicted subtypes (multi-label)
    "confidence": float,               # Confidence for primary (0.0-1.0)
    "subtype_scores": Dict[str, float],  # All subtype probabilities
}
```

## Example Usage

```python
from sap_llm.stages.type_identifier import TypeIdentifierStage

# Initialize stage
stage = TypeIdentifierStage(config)

# Get hierarchy info
info = stage.get_hierarchy_info()
print(f"Total subtypes: {info['total_subtypes']}")  # 63

# Process document
result = stage.process({
    "doc_type": "PURCHASE_ORDER",
    "enhanced_images": [image],
    "ocr_results": [{"words": [...], "boxes": [...]}],
})

print(f"Primary subtype: {result['subtype']}")  # e.g., "CONTRACT"
print(f"All subtypes: {result['subtypes']}")    # e.g., ["CONTRACT", "SERVICE"]
print(f"Confidence: {result['confidence']:.4f}")  # e.g., 0.9234
```

## Document Type → Subtype Hierarchy

### Invoice Types
- **SUPPLIER_INVOICE** → STANDARD, CREDIT_MEMO, DEBIT_MEMO, PREPAYMENT, DOWN_PAYMENT, RECURRING, PROFORMA, COMMERCIAL
- **CUSTOMER_INVOICE** → STANDARD, CREDIT_NOTE, DEBIT_NOTE, PROFORMA, RECURRING, MILESTONE

### Purchase Order Types
- **PURCHASE_ORDER** → STANDARD, BLANKET, CONTRACT, SERVICE, SUBCONTRACT, CONSIGNMENT, STOCK_TRANSFER, LIMIT, DROP_SHIP, CAPEX

### Sales Order Types
- **SALES_ORDER** → STANDARD, RUSH, SCHEDULED, CONSIGNMENT, RETURNS, CREDIT_ONLY

### Other Document Types
- **GOODS_RECEIPT** → STANDARD, RETURN_TO_VENDOR, TRANSFER_POSTING, OTHER_RECEIPT
- **DELIVERY_NOTE** → STANDARD, PARTIAL, COMPLETE, RETURNS
- **ADVANCED_SHIPPING_NOTICE** → STANDARD, PARTIAL, COMPLETE
- **CREDIT_NOTE** → SUPPLIER_CREDIT, CUSTOMER_CREDIT, GENERAL
- **DEBIT_NOTE** → SUPPLIER_DEBIT, CUSTOMER_DEBIT, GENERAL
- **PAYMENT_ADVICE** → STANDARD, PARTIAL, ADVANCE
- **REMITTANCE_ADVICE** → STANDARD, CONSOLIDATED
- **STATEMENT_OF_ACCOUNT** → MONTHLY, QUARTERLY, ANNUAL, ON_DEMAND
- **QUOTE** → REQUEST_FOR_QUOTE, SUPPLIER_QUOTE, SALES_QUOTE
- **CONTRACT** → PURCHASING, SALES, MASTER_SERVICE_AGREEMENT

## Technical Details

### Dependencies
- PyTorch for neural network operations
- LayoutLMv3 (via VisionEncoder) for visual-text feature extraction
- Transformers library for model loading

### Performance Characteristics
- **Model Size**: ~300M parameters (LayoutLMv3) + classifier heads (~1M each)
- **Precision**: FP16 (GPU) or FP32 (CPU)
- **Device**: CUDA if available, otherwise CPU
- **Latency**: Expected <200ms per document (GPU)

### Training Notes
The current implementation initializes classifiers with random weights. In production:
1. Train each classifier head using labeled data
2. Use `save_classifiers()` to persist trained models
3. Load trained models with `load_classifiers()` at initialization

## Verification

Run the verification script to confirm implementation:
```bash
python examples/verify_type_identifier.py
```

Expected output:
- ✅ 63 subtypes implemented (exceeds 35+ requirement)
- ✅ Vision encoder integration
- ✅ All required methods implemented
- ✅ PyTorch neural network support

## Future Enhancements

1. **Active Learning**: Implement confidence-based active learning for model improvement
2. **Ensemble Methods**: Combine multiple classifiers for better accuracy
3. **Attention Visualization**: Show which document regions influenced classification
4. **Cross-lingual Support**: Extend to non-English documents
5. **Incremental Learning**: Add new subtypes without full retraining

## Files Modified

- `/home/user/SAP_LLM/sap_llm/stages/type_identifier.py` - Main implementation

## Files Created

- `/home/user/SAP_LLM/examples/verify_type_identifier.py` - Verification script
- `/home/user/SAP_LLM/docs/TYPE_IDENTIFIER_IMPLEMENTATION.md` - This documentation
