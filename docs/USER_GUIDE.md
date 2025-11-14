# SAP_LLM User Guide

Complete guide for end users to process documents with SAP_LLM.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Processing Documents](#processing-documents)
4. [Understanding Results](#understanding-results)
5. [Working with the API](#working-with-the-api)
6. [Web Interface](#web-interface)
7. [Batch Processing](#batch-processing)
8. [Monitoring Your Documents](#monitoring-your-documents)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

SAP_LLM is an autonomous document processing system designed to extract structured data from business documents (invoices, purchase orders, receipts, etc.) and route them to SAP systems. This guide will help you understand how to use the system effectively.

### Who Should Use This Guide?

- Business users who need to process documents
- Operations teams managing document workflows
- System administrators configuring the system
- Anyone new to SAP_LLM

### What You'll Learn

- How to submit documents for processing
- How to interpret extraction results
- How to troubleshoot common issues
- Best practices for optimal results

---

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- Access credentials (API key or JWT token)
- Documents to process (PDF, PNG, JPG, or TIFF format)
- Internet connection to access the API
- (Optional) A REST client like Postman or cURL

### First Steps

1. **Obtain Your API Key**
   - Contact your system administrator
   - Your API key will be in the format: `sap-llm_xxxxxxxxxxxxx`
   - Store it securely (never share or commit to version control)

2. **Set Up Your Environment**
   ```bash
   # Set your API key as an environment variable
   export SAP_LLM_API_KEY="your-api-key-here"

   # Set the API endpoint
   export SAP_LLM_ENDPOINT="https://api.yourcompany.com/v1"
   ```

3. **Verify Your Access**
   ```bash
   curl -X GET "$SAP_LLM_ENDPOINT/health" \
     -H "Authorization: Bearer $SAP_LLM_API_KEY"
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "version": "1.0.0",
     "timestamp": "2025-11-14T10:00:00Z"
   }
   ```

---

## Processing Documents

### Supported Document Types

SAP_LLM can process the following document types:

| Document Type | Subtypes | Common Fields |
|--------------|----------|---------------|
| **Invoice** | Standard, Credit Note, Proforma | Vendor, Amount, Date, Items |
| **Purchase Order** | Standard, Blanket, Contract | PO Number, Items, Delivery Date |
| **Receipt** | Sales, Return | Merchant, Total, Items |
| **Delivery Note** | Inbound, Outbound | Tracking Number, Items |
| **Credit Memo** | Standard, Adjustment | Original Invoice, Credit Amount |
| **Bill of Lading** | Ocean, Air, Ground | Shipment Details, Carrier |
| **Packing Slip** | Standard | Contents, Quantities |
| **Payment Advice** | Standard | Payment Details, Bank Info |

### File Format Requirements

- **Supported Formats**: PDF, PNG, JPG, JPEG, TIFF
- **Maximum File Size**: 50 MB
- **Resolution**: Minimum 150 DPI for images (300 DPI recommended)
- **Quality**: Clear, readable text; avoid blurry or low-contrast images
- **Orientation**: Any (system auto-rotates)
- **Color**: Color or grayscale (color recommended for logos/stamps)

### Processing a Single Document

#### Using cURL

```bash
curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/extract" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/invoice.pdf" \
  -F "document_type=invoice" \
  -F "priority=normal"
```

#### Using Python

```python
import requests

# Configuration
API_KEY = "your-api-key"
ENDPOINT = "https://api.yourcompany.com/v1"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Upload document
with open("invoice.pdf", "rb") as file:
    files = {"file": file}
    data = {
        "document_type": "invoice",
        "priority": "normal"
    }

    response = requests.post(
        f"{ENDPOINT}/v1/documents/extract",
        headers=headers,
        files=files,
        data=data
    )

# Get job ID
result = response.json()
job_id = result["job_id"]
print(f"Job submitted: {job_id}")
```

#### Using JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('invoice.pdf'));
form.append('document_type', 'invoice');
form.append('priority', 'normal');

axios.post('https://api.yourcompany.com/v1/documents/extract', form, {
  headers: {
    'Authorization': 'Bearer your-api-key',
    ...form.getHeaders()
  }
})
.then(response => {
  console.log('Job ID:', response.data.job_id);
})
.catch(error => {
  console.error('Error:', error.response.data);
});
```

### Processing Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Document file to process |
| `document_type` | String | auto | Document type (auto-detect if not specified) |
| `priority` | String | normal | Processing priority (low, normal, high, urgent) |
| `webhook_url` | String | Optional | URL to receive results via webhook |
| `extract_tables` | Boolean | true | Extract table data from document |
| `extract_line_items` | Boolean | true | Extract line item details |
| `language` | String | auto | Document language (auto-detect if not specified) |
| `return_images` | Boolean | false | Include processed images in response |
| `confidence_threshold` | Float | 0.85 | Minimum confidence for auto-approval |

### Example with All Options

```bash
curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/extract" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY" \
  -F "file=@invoice.pdf" \
  -F "document_type=invoice" \
  -F "priority=high" \
  -F "webhook_url=https://your-app.com/webhook" \
  -F "extract_tables=true" \
  -F "extract_line_items=true" \
  -F "language=en" \
  -F "confidence_threshold=0.90"
```

---

## Understanding Results

### Checking Job Status

After submitting a document, you'll receive a `job_id`. Use it to check the processing status:

```bash
curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/status" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY"
```

Response:
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "progress": 100,
  "created_at": "2025-11-14T10:00:00Z",
  "updated_at": "2025-11-14T10:00:15Z",
  "processing_time_ms": 1250
}
```

### Job Statuses

| Status | Description | Next Action |
|--------|-------------|-------------|
| `queued` | Job is waiting to be processed | Wait |
| `processing` | Job is currently being processed | Wait |
| `completed` | Processing completed successfully | Retrieve results |
| `failed` | Processing failed | Check error details |
| `cancelled` | Job was cancelled | Resubmit if needed |
| `requires_review` | Low confidence, needs human review | Review and approve |

### Retrieving Results

Once status is `completed`, retrieve the extraction results:

```bash
curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/results" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY"
```

### Understanding the Response

```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "document_type": "invoice",
  "document_subtype": "standard_invoice",
  "confidence": 0.96,
  "language": "en",
  "processing_stages": {
    "inbox": {"status": "completed", "duration_ms": 50},
    "preprocessing": {"status": "completed", "duration_ms": 200},
    "classification": {"status": "completed", "duration_ms": 150},
    "type_identification": {"status": "completed", "duration_ms": 100},
    "extraction": {"status": "completed", "duration_ms": 500},
    "quality_check": {"status": "completed", "duration_ms": 150},
    "validation": {"status": "completed", "duration_ms": 50},
    "routing": {"status": "completed", "duration_ms": 50}
  },
  "extracted_data": {
    "header": {
      "invoice_number": {
        "value": "INV-2025-001234",
        "confidence": 0.99,
        "bounding_box": [100, 150, 250, 170]
      },
      "invoice_date": {
        "value": "2025-11-10",
        "confidence": 0.98,
        "normalized_format": "YYYY-MM-DD"
      },
      "due_date": {
        "value": "2025-12-10",
        "confidence": 0.97
      },
      "vendor_name": {
        "value": "ACME Corporation",
        "confidence": 0.99
      },
      "vendor_address": {
        "value": "123 Main St, Springfield, IL 62701",
        "confidence": 0.96
      },
      "total_amount": {
        "value": 15234.56,
        "currency": "USD",
        "confidence": 0.99
      },
      "tax_amount": {
        "value": 1234.56,
        "currency": "USD",
        "confidence": 0.98
      },
      "subtotal": {
        "value": 14000.00,
        "currency": "USD",
        "confidence": 0.99
      }
    },
    "line_items": [
      {
        "line_number": 1,
        "description": "Widget A",
        "quantity": 100,
        "unit_price": 50.00,
        "total": 5000.00,
        "confidence": 0.97
      },
      {
        "line_number": 2,
        "description": "Widget B",
        "quantity": 200,
        "unit_price": 45.00,
        "total": 9000.00,
        "confidence": 0.98
      }
    ],
    "payment_terms": {
      "value": "Net 30",
      "confidence": 0.95
    }
  },
  "validation_results": {
    "overall_status": "passed",
    "checks": [
      {
        "rule": "total_matches_sum",
        "status": "passed",
        "message": "Total amount matches sum of line items plus tax"
      },
      {
        "rule": "required_fields_present",
        "status": "passed",
        "message": "All required fields extracted"
      },
      {
        "rule": "date_format_valid",
        "status": "passed",
        "message": "All dates are valid"
      }
    ]
  },
  "routing_decision": {
    "destination": "sap_s4hana",
    "endpoint": "/sap/opu/odata/sap/API_INVOICE",
    "method": "POST",
    "payload_ready": true
  }
}
```

### Key Fields Explained

#### Confidence Scores
- **0.95-1.00**: Excellent - High confidence, auto-approve
- **0.85-0.94**: Good - Generally accurate, spot-check recommended
- **0.75-0.84**: Fair - Review recommended
- **Below 0.75**: Low - Manual review required

#### Bounding Boxes
- Format: `[x1, y1, x2, y2]` in pixels
- Indicates where on the document the field was found
- Useful for visual verification

#### Normalized Values
- Dates are normalized to ISO 8601 format (YYYY-MM-DD)
- Amounts are normalized to decimal numbers
- Currencies are in ISO 4217 codes (USD, EUR, GBP, etc.)

### Handling Low Confidence Results

If `confidence` is below your threshold or `status` is `requires_review`:

1. **Review the extracted data**
   ```bash
   curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/review" \
     -H "Authorization: Bearer $SAP_LLM_API_KEY"
   ```

2. **Approve or correct the data**
   ```bash
   curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/review" \
     -H "Authorization: Bearer $SAP_LLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "action": "approve",
       "corrections": {
         "invoice_number": "INV-2025-001234"
       },
       "reviewer": "john.doe@company.com"
     }'
   ```

---

## Working with the API

### Authentication

All API requests require authentication using one of these methods:

#### API Key (Recommended for Applications)
```bash
curl -H "Authorization: Bearer your-api-key" \
  "$SAP_LLM_ENDPOINT/v1/documents"
```

#### JWT Token (For User Sessions)
```bash
# Login to get token
curl -X POST "$SAP_LLM_ENDPOINT/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@company.com", "password": "password"}'

# Response includes access_token
# Use it in subsequent requests
curl -H "Authorization: Bearer eyJhbGc..." \
  "$SAP_LLM_ENDPOINT/v1/documents"
```

### Rate Limits

| Plan | Requests/min | Requests/hour | Requests/day |
|------|--------------|---------------|--------------|
| Free | 10 | 100 | 1,000 |
| Basic | 60 | 1,000 | 10,000 |
| Professional | 300 | 5,000 | 50,000 |
| Enterprise | Custom | Custom | Custom |

When you exceed the rate limit:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 30,
  "limit": 60,
  "remaining": 0,
  "reset": "2025-11-14T10:05:00Z"
}
```

### Webhooks

Instead of polling for results, configure a webhook to receive notifications:

```bash
curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/extract" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY" \
  -F "file=@invoice.pdf" \
  -F "webhook_url=https://your-app.com/webhook/sap-llm"
```

Your webhook will receive:
```json
{
  "event": "document.completed",
  "job_id": "job_abc123",
  "status": "completed",
  "document_type": "invoice",
  "confidence": 0.96,
  "results_url": "https://api.yourcompany.com/v1/documents/job_abc123/results",
  "timestamp": "2025-11-14T10:00:15Z"
}
```

---

## Web Interface

SAP_LLM provides a web interface for easier document processing.

### Accessing the Web Interface

1. Navigate to `https://app.yourcompany.com/sap-llm`
2. Log in with your credentials
3. You'll see the dashboard

### Dashboard Overview

The dashboard shows:
- **Recent Documents**: Your recently processed documents
- **Processing Queue**: Documents currently being processed
- **Statistics**: Daily/weekly/monthly processing stats
- **Quick Upload**: Drag-and-drop area for new documents

### Uploading Documents via Web Interface

1. Click **Upload Document** or drag files to the upload area
2. Select document type (or leave as "Auto-detect")
3. Set processing priority if needed
4. Click **Process**
5. Monitor progress in real-time

### Viewing Results

1. Click on a document in the list
2. View extracted data in structured format
3. See confidence scores for each field
4. Download results as JSON or Excel
5. Approve or request corrections

### Bulk Operations

1. Select multiple documents using checkboxes
2. Click **Bulk Actions**
3. Choose action:
   - Approve all
   - Export all
   - Reprocess all
   - Delete all

---

## Batch Processing

For processing large volumes of documents:

### Upload Batch

```bash
curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/batch" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY" \
  -F "files[]=@invoice1.pdf" \
  -F "files[]=@invoice2.pdf" \
  -F "files[]=@invoice3.pdf" \
  -F "priority=normal"
```

Response:
```json
{
  "batch_id": "batch_xyz789",
  "total_documents": 3,
  "status": "queued",
  "estimated_completion": "2025-11-14T10:05:00Z",
  "job_ids": ["job_1", "job_2", "job_3"]
}
```

### Check Batch Status

```bash
curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/batch/{batch_id}/status" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY"
```

### Download Batch Results

```bash
curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/batch/{batch_id}/results" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY" \
  -o results.zip
```

Results are provided as:
- **JSON**: One file per document
- **Excel**: All results in one spreadsheet
- **CSV**: Flat file format

---

## Monitoring Your Documents

### Real-Time Monitoring via WebSocket

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://api.yourcompany.com/v1/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-api-key'
  }));

  // Subscribe to job updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    job_id: 'job_abc123'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status update:', data);
  // { type: 'status', job_id: 'job_abc123', status: 'processing', progress: 50 }
};
```

### Viewing Metrics

Access your processing metrics:

```bash
curl -X GET "$SAP_LLM_ENDPOINT/v1/metrics" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY"
```

Response:
```json
{
  "period": "last_30_days",
  "documents_processed": 15234,
  "average_processing_time_ms": 1250,
  "accuracy_rate": 0.97,
  "touchless_rate": 0.89,
  "cost_per_document": 0.00006,
  "by_document_type": {
    "invoice": 10000,
    "purchase_order": 3000,
    "receipt": 2234
  }
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Authentication failed"
**Solution:**
- Verify your API key is correct
- Check if the key has expired
- Ensure the `Authorization` header is properly formatted: `Bearer your-api-key`

#### Issue: "File too large"
**Solution:**
- Maximum file size is 50 MB
- Compress the PDF or reduce image resolution
- Split multi-page PDFs into smaller files

#### Issue: "Unsupported file format"
**Solution:**
- Convert your file to PDF, PNG, JPG, or TIFF
- Ensure file extension matches content type

#### Issue: "Low confidence scores"
**Causes and Solutions:**
- **Poor image quality**: Scan at higher DPI (300+ recommended)
- **Blurry or skewed**: Re-scan or take a clearer photo
- **Low contrast**: Adjust brightness/contrast before uploading
- **Unusual layout**: May require manual review
- **Handwritten text**: System works best with printed text

#### Issue: "Processing timeout"
**Solution:**
- Large or complex documents may take longer
- Check if the document has excessive pages (>50 pages may be split)
- Retry with higher priority if urgent

#### Issue: "Incorrect extraction"
**Solution:**
- Verify document quality and readability
- Ensure correct document type is specified
- Use the review interface to correct and retrain the system
- Report persistent issues to support

### Getting Help

1. **Check the logs**
   ```bash
   curl -X GET "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/logs" \
     -H "Authorization: Bearer $SAP_LLM_API_KEY"
   ```

2. **Contact Support**
   - Email: support@qorsync.com
   - Include: Job ID, error message, and sample document (if possible)

3. **Check System Status**
   - Status page: https://status.yourcompany.com

---

## FAQ

### General Questions

**Q: How long does processing take?**
A: Typically 1-3 seconds for standard documents. Complex documents with many line items may take up to 10 seconds.

**Q: What languages are supported?**
A: 50+ languages including English, Spanish, German, French, Chinese, Japanese, and more. The system auto-detects the language.

**Q: Can I process handwritten documents?**
A: Limited support. The system is optimized for printed/typed text. Handwritten documents may have lower accuracy.

**Q: Is my data secure?**
A: Yes. All data is encrypted in transit (TLS 1.3) and at rest (AES-256). Documents are automatically deleted after 30 days unless configured otherwise.

**Q: Can I reprocess a document?**
A: Yes. Use the reprocess endpoint:
```bash
curl -X POST "$SAP_LLM_ENDPOINT/v1/documents/{job_id}/reprocess" \
  -H "Authorization: Bearer $SAP_LLM_API_KEY"
```

### Billing Questions

**Q: How am I charged?**
A: Billing is based on:
- Number of documents processed
- Processing time (for complex documents)
- Storage (if documents are retained beyond 30 days)
- API calls (for high-volume users)

**Q: What happens if I exceed my quota?**
A: Processing will be queued until your quota resets (typically monthly) or you can upgrade your plan.

### Technical Questions

**Q: Can I use this in my application?**
A: Yes. SAP_LLM provides a RESTful API designed for integration. See the [API Documentation](API_DOCUMENTATION.md).

**Q: Do you have SDKs?**
A: Yes. Official SDKs are available for:
- Python: `pip install sap-llm-client`
- JavaScript/Node.js: `npm install @qorsync/sap-llm-client`
- Java: Maven/Gradle packages available

**Q: Can I self-host SAP_LLM?**
A: Yes, for Enterprise customers. Contact sales@qorsync.com for licensing.

**Q: How do I improve extraction accuracy?**
A:
- Provide high-quality scans (300 DPI)
- Ensure documents are not skewed or rotated
- Specify the correct document type
- Use the review interface to correct errors (system learns from feedback)

**Q: Can I customize the extraction fields?**
A: Yes, Enterprise customers can define custom schemas. Contact support for details.

---

## Best Practices

### For Best Results

1. **Document Quality**
   - Use 300 DPI scans
   - Ensure good contrast
   - Avoid skewed or rotated images
   - Use color scans when possible

2. **File Preparation**
   - Remove unnecessary pages
   - Ensure correct orientation
   - Use searchable PDFs when available
   - Keep file sizes reasonable (<10 MB recommended)

3. **API Usage**
   - Implement retry logic with exponential backoff
   - Use webhooks instead of polling
   - Batch similar documents together
   - Cache results when appropriate

4. **Error Handling**
   - Always check job status before retrieving results
   - Implement proper error handling for rate limits
   - Log all API interactions for debugging
   - Monitor confidence scores and review low-confidence extractions

5. **Security**
   - Rotate API keys regularly (every 90 days)
   - Use environment variables for credentials
   - Implement proper access controls
   - Enable audit logging

---

## Next Steps

Now that you understand how to use SAP_LLM:

1. **Try Processing Your First Document** - Use the examples above to process a test document
2. **Integrate with Your Application** - Use our SDKs or REST API
3. **Explore Advanced Features** - Check out [Developer Guide](DEVELOPER_GUIDE.md) for customization options
4. **Monitor Performance** - Set up dashboards to track your metrics
5. **Optimize Your Workflow** - Use batch processing and webhooks for efficiency

For more information:
- [Developer Guide](DEVELOPER_GUIDE.md) - Technical documentation
- [API Documentation](API_DOCUMENTATION.md) - Complete API reference
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Detailed troubleshooting
- [Architecture](ARCHITECTURE.md) - System architecture

---

**Need Help?** Contact support@qorsync.com or visit our [documentation portal](https://docs.qorsync.com/sap-llm).
