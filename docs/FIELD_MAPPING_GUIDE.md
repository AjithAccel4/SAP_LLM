# SAP Field Mapping Guide

## Overview

The SAP Field Mapping system provides a database-driven, extensible approach to transforming document data between different formats (OCR extraction, internal formats, SAP API formats). This guide covers how to use, modify, and extend the field mapping system.

## Architecture

### Components

1. **FieldMappingManager** (`sap_llm/knowledge_base/field_mappings.py`)
   - Loads and caches field mappings from JSON configuration files
   - Applies transformations and validations
   - Handles nested structures and arrays
   - Provides performance-optimized data transformation

2. **Mapping Configuration Files** (`data/field_mappings/*.json`)
   - JSON files defining field mappings for each document type
   - Support for 13 standard SAP document types
   - Extensible for custom document types

3. **KnowledgeBaseQuery** (`sap_llm/knowledge_base/query.py`)
   - High-level interface using `transform_format()` method
   - Automatically selects appropriate mapping
   - Falls back to legacy mappings if needed

## Supported Document Types

### Purchase Orders (4 subtypes)
1. **Standard** (`purchase_order_standard.json`)
   - Regular material purchase orders
   - Standard procurement processes

2. **Service** (`purchase_order_service.json`)
   - Service-based purchase orders
   - Consulting, maintenance, professional services

3. **Subcontracting** (`purchase_order_subcontracting.json`)
   - Subcontracting arrangements
   - Component tracking

4. **Consignment** (`purchase_order_consignment.json`)
   - Consignment stock arrangements

### Supplier Invoices (3 subtypes)
1. **Standard** (`supplier_invoice_standard.json`)
   - Regular vendor invoices

2. **Credit Memo** (`supplier_invoice_credit_memo.json`)
   - Credit notes and reversals

3. **Down Payment** (`supplier_invoice_down_payment.json`)
   - Advance payment invoices

### Goods Receipts (2 subtypes)
1. **Purchase Order** (`goods_receipt_for_po.json`)
   - Material receipts against POs

2. **Return** (`goods_receipt_return.json`)
   - Return deliveries to vendors

### Service Entry Sheets (2 subtypes)
1. **Purchase Order** (`service_entry_sheet_for_po.json`)
   - Service confirmations against POs

2. **Blanket PO** (`service_entry_sheet_blanket.json`)
   - Service confirmations against blanket agreements

### Master Data (2 types)
1. **Payment Terms** (`payment_terms.json`)
   - Payment conditions and discounts

2. **Incoterms** (`incoterms.json`)
   - International commercial terms

## Mapping File Structure

### Basic Structure

```json
{
  "document_type": "PurchaseOrder",
  "subtype": "Standard",
  "api_version": "A_PurchaseOrder",
  "description": "Standard Purchase Order mapping for SAP S/4HANA OData API",
  "config": {
    "copy_unmapped": false,
    "strict_validation": true,
    "allow_partial": false
  },
  "mappings": {
    "source_field": {
      "sap_field": "SAPFieldName",
      "data_type": "string",
      "required": true,
      "max_length": 10,
      "transformations": ["uppercase", "trim"],
      "validation": "^[A-Z0-9]{1,10}$",
      "default": null,
      "description": "Field description"
    }
  },
  "nested_mappings": {
    "items": {
      "sap_collection": "to_PurchaseOrderItem",
      "mappings": {
        // Item-level field mappings
      }
    }
  }
}
```

### Field Configuration Options

#### Basic Properties
- **sap_field**: Target SAP field name
- **data_type**: Data type (string, integer, decimal, date)
- **required**: Whether field is mandatory
- **description**: Human-readable description

#### Validation
- **validation**: Regex pattern for validation
- **max_length**: Maximum length for string fields
- **default**: Default value if source field is missing

#### Transformations
List of transformations to apply in order. See [Transformations](#transformations) section.

## Transformations

### String Transformations
- `uppercase` - Convert to uppercase
- `lowercase` - Convert to lowercase
- `trim` - Remove leading/trailing whitespace

### Padding Transformations
- `pad_left:LENGTH:CHAR` - Pad left (e.g., `pad_left:10:0` → "0000012345")
- `pad_right:LENGTH:CHAR` - Pad right

### Date Transformations
- `parse_date` - Parse date from various formats
- `format_date:FORMAT` - Format date
  - `YYYYMMDD` → "20240115"
  - `YYYY-MM-DD` → "2024-01-15"
  - `DD/MM/YYYY` → "15/01/2024"
  - Custom strftime formats

### Number Transformations
- `parse_amount` - Parse monetary amounts (handles currency symbols, commas)
- `parse_integer` - Parse integers
- `format_decimal:PLACES` - Round to decimal places
- `negate` - Negate value (for credit memos)

### Special Transformations
- `validate_iso_currency` - Validate ISO 4217 currency codes

### Example Transformation Chains

```json
{
  "vendor_id": {
    "sap_field": "Supplier",
    "transformations": ["uppercase", "trim", "pad_left:10:0"]
  },
  "invoice_date": {
    "sap_field": "DocumentDate",
    "transformations": ["parse_date", "format_date:YYYYMMDD"]
  },
  "total_amount": {
    "sap_field": "InvoiceGrossAmount",
    "transformations": ["parse_amount", "format_decimal:2"]
  }
}
```

## Usage Examples

### Basic Transformation

```python
from sap_llm.knowledge_base.query import KnowledgeBaseQuery
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage

# Initialize
storage = KnowledgeBaseStorage("kb.db")
kb_query = KnowledgeBaseQuery(storage)

# Transform purchase order data
po_data = {
    "po_number": "4500012345",
    "po_date": "2024-01-15",
    "vendor_id": "VENDOR001",
    "company_code": "1000",
    "currency": "USD",
    "total_amount": "10,500.00"
}

# Transform to SAP API format
sap_data = kb_query.transform_format(
    po_data,
    source_format="PURCHASE_ORDER",
    target_format="SAP_API"
)

print(sap_data)
# {
#     "PurchaseOrder": "4500012345",
#     "PurchaseOrderDate": "20240115",
#     "Supplier": "0VENDOR001",
#     "CompanyCode": "1000",
#     "DocumentCurrency": "USD",
#     "TotalAmount": 10500.0
# }
```

### With Nested Items

```python
po_data = {
    "po_number": "4500012345",
    "po_date": "2024-01-15",
    "vendor_id": "VENDOR001",
    "company_code": "1000",
    "currency": "USD",
    "items": [
        {
            "item_number": "10",
            "material_number": "MAT-001",
            "quantity": "100",
            "unit_price": "105.00"
        }
    ]
}

sap_data = kb_query.transform_format(
    po_data,
    "PURCHASE_ORDER",
    "SAP_API"
)

# Items transformed to to_PurchaseOrderItem
print(sap_data["to_PurchaseOrderItem"])
# [
#     {
#         "PurchaseOrderItem": "00010",
#         "Material": "MAT-001",
#         "OrderQuantity": 100.0,
#         "NetPriceAmount": 105.0
#     }
# ]
```

### Explicit Document Subtype

```python
# Transform service PO
service_po_data = {...}

sap_data = kb_query.transform_format(
    service_po_data,
    "PurchaseOrder:Service",  # Explicit subtype
    "SAP_API"
)
```

### Direct FieldMappingManager Usage

```python
from sap_llm.knowledge_base.field_mappings import FieldMappingManager

# Initialize manager
manager = FieldMappingManager()

# Get mapping
mapping = manager.get_mapping("PurchaseOrder", "Standard")

# Validate data
is_valid, errors = manager.validate_mapping(po_data, mapping)

# Transform data
sap_data = manager.transform_data(po_data, mapping)
```

## Adding New Document Types

### Step 1: Create Mapping File

Create a new JSON file in `data/field_mappings/`:

```bash
touch data/field_mappings/my_document_type.json
```

### Step 2: Define Mapping Structure

```json
{
  "document_type": "MyDocument",
  "subtype": "Standard",
  "api_version": "A_MyDocument",
  "description": "My custom document mapping",
  "config": {
    "copy_unmapped": false,
    "strict_validation": true
  },
  "mappings": {
    "my_field": {
      "sap_field": "MySAPField",
      "data_type": "string",
      "required": true,
      "transformations": ["uppercase", "trim"]
    }
  }
}
```

### Step 3: Add Format Identifier (Optional)

Update `_parse_format_identifier()` in `query.py`:

```python
type_mapping = {
    # ... existing mappings ...
    "MY_DOCUMENT": ("MyDocument", "Standard"),
}
```

### Step 4: Test

```python
result = kb_query.transform_format(
    my_data,
    "MY_DOCUMENT",
    "SAP_API"
)
```

## Modifying Existing Mappings

### Adding a New Field

Edit the appropriate JSON file:

```json
{
  "mappings": {
    // Existing fields...
    "new_field": {
      "sap_field": "NewSAPField",
      "data_type": "string",
      "required": false,
      "transformations": ["trim"],
      "default": null
    }
  }
}
```

The FieldMappingManager automatically reloads mappings on initialization.

### Changing Transformations

```json
{
  "vendor_id": {
    "sap_field": "Supplier",
    "transformations": [
      "uppercase",
      "trim",
      "pad_left:10:0"  // Add padding
    ]
  }
}
```

### Adding Validation Rules

```json
{
  "company_code": {
    "sap_field": "CompanyCode",
    "validation": "^[A-Z0-9]{4}$",  // Must be 4 alphanumeric chars
    "max_length": 4
  }
}
```

## Performance Considerations

### Caching
- Mappings are loaded once and cached in memory
- `get_mapping()` uses `@lru_cache` for fast repeated access
- Transformations are optimized for bulk processing

### Benchmarks
- **1000 documents < 2 seconds** (tested)
- Nested structures (5 levels deep) supported
- Arrays of arbitrary size supported

### Best Practices
1. Minimize transformation complexity
2. Use appropriate data types
3. Cache FieldMappingManager instance
4. Reuse KnowledgeBaseQuery instance

## Validation

### Strict vs. Non-Strict Mode

```python
# Strict validation (raises errors for missing required fields)
is_valid, errors = manager.validate_mapping(
    data, mapping, strict=True
)

# Non-strict validation (only warns)
is_valid, errors = manager.validate_mapping(
    data, mapping, strict=False
)
```

### Validation Rules
1. **Required fields** - Checked if `required: true`
2. **Regex patterns** - Validated against `validation` pattern
3. **Max length** - String length checked against `max_length`
4. **Data types** - Type conversion attempted during transformation

## Troubleshooting

### Issue: Mapping Not Found

**Symptom**: Warning "No mapping found for {document_type}:{subtype}"

**Solutions**:
1. Check file exists in `data/field_mappings/`
2. Verify `document_type` and `subtype` in JSON match request
3. Check file is valid JSON
4. Restart application to reload mappings

### Issue: Transformation Fails

**Symptom**: Original value returned unchanged

**Solutions**:
1. Check logs for transformation error details
2. Verify transformation syntax (e.g., `pad_left:10:0` not `pad_left:10`)
3. Ensure data type is compatible with transformation
4. Test transformation in isolation

### Issue: Validation Errors

**Symptom**: Validation warnings in logs

**Solutions**:
1. Check required fields are present
2. Verify field values match validation regex
3. Ensure field lengths don't exceed max_length
4. Use non-strict mode if errors are non-critical

### Issue: Nested Items Not Transformed

**Symptom**: Nested items missing or not transformed

**Solutions**:
1. Verify `nested_mappings` is defined
2. Check `sap_collection` matches source field structure
3. Ensure source data contains nested field
4. Check nesting level < `max_nesting_level` (default 5)

## API Reference

### FieldMappingManager

#### Methods

##### `get_mapping(document_type, subtype="Standard", target_format="SAP_API")`
Get field mapping for a document type.

**Returns**: Mapping dictionary or None

##### `validate_mapping(data, mapping, strict=None)`
Validate data against mapping.

**Returns**: Tuple of (is_valid, list_of_errors)

##### `apply_transformations(value, transformations, field_name="")`
Apply transformation functions to a value.

**Returns**: Transformed value

##### `transform_data(source_data, mapping, max_nesting_level=5)`
Transform data using mapping configuration.

**Returns**: Transformed data dictionary

##### `get_document_types()`
Get list of available document types.

**Returns**: List of document type names

##### `get_subtypes(document_type)`
Get list of subtypes for a document type.

**Returns**: List of subtype names

### KnowledgeBaseQuery

#### Methods

##### `transform_format(source_data, source_format, target_format)`
Transform document data from source format to target format.

**Returns**: Transformed document data

## FAQ

**Q: How do I add support for a custom SAP API?**

A: Create a new mapping JSON file with the appropriate SAP field names and transformations. No code changes required.

**Q: Can I use the same field mapping for multiple document subtypes?**

A: No, each subtype requires its own mapping file. However, you can copy and modify existing files as a starting point.

**Q: What happens if a required field is missing?**

A: In strict mode, validation fails and errors are returned. In non-strict mode, a warning is logged but transformation continues.

**Q: How do I handle custom date formats?**

A: Use the `format_date` transformation with a custom strftime format string.

**Q: Can I transform data without using the knowledge base?**

A: Yes, you can use FieldMappingManager directly without requiring KnowledgeBaseStorage.

**Q: How do I debug transformation issues?**

A: Enable debug logging to see detailed transformation steps:

```python
import logging
logging.getLogger('sap_llm.knowledge_base.field_mappings').setLevel(logging.DEBUG)
```

## Support

For issues or questions:
1. Check this guide and API reference
2. Review test files for usage examples
3. Check logs for detailed error messages
4. Consult SAP API documentation for field requirements
