# SAP_LLM API Documentation

Comprehensive guide to the SAP_LLM Document Processing API.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [WebSocket API](#websocket-api)
- [Request & Response Examples](#request--response-examples)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Code Examples](#code-examples)
- [Production Deployment](#production-deployment)

## Overview

The SAP_LLM API is a production-ready REST API for autonomous document processing with zero 3rd party LLM dependencies. It provides intelligent extraction of fields from business documents and seamless integration with SAP systems.

### Key Features

- **Intelligent Document Processing**: Automatically extract fields from invoices, purchase orders, receipts, and more
- **Real-time Processing**: Track document processing status in real-time via WebSocket
- **High Accuracy**: Vision-language models trained specifically for business documents
- **SAP Integration**: Seamless routing to SAP S/4HANA, SAP Ariba, and other SAP systems
- **Quality Assurance**: Built-in quality checks and validation
- **Production Ready**: Rate limiting, monitoring, and observability built-in

### Base URL

```
Production: https://api.sap-llm.example.com
Development: http://localhost:8000
```

## Architecture

### Processing Pipeline

The API processes documents through an 8-stage pipeline:

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ Inbox   │───▶│Preprocessing │───▶│Classification │───▶│Type Identifier│
└─────────┘    └──────────────┘    └───────────────┘    └──────────────┘
     ▼                                                           ▼
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ Routing │◀───│ Validation   │◀───│Quality Check  │◀───│  Extraction  │
└─────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

**Stage Descriptions:**

1. **Inbox**: Document ingestion and initial validation
2. **Preprocessing**: Image enhancement and normalization
3. **Classification**: Document category identification
4. **Type Identification**: Specific document type detection
5. **Extraction**: Field extraction using unified model
6. **Quality Check**: Confidence scoring and validation
7. **Validation**: Business rules and data validation
8. **Routing**: SAP system routing and posting

### System Components

- **FastAPI Server**: High-performance async web server
- **Unified Model**: Vision-language model for document understanding
- **Process Memory Graph (PMG)**: CosmosDB-based historical data store
- **APOP Framework**: Autonomous processing orchestration
- **Monitoring**: Prometheus metrics and SLO tracking

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- SAP system credentials (for production)

### Quick Start

#### 1. Start the API Server

```bash
# Using Python
cd SAP_LLM
python -m sap_llm.api.server

# Using Docker
docker-compose up api
```

#### 2. Verify Server is Running

```bash
curl http://localhost:8000/health
```

#### 3. Process Your First Document

```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "expected_type=invoice"
```

#### 4. Check Processing Status

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

### Interactive Documentation

Access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

### Current Implementation

The API currently uses **IP-based rate limiting** for protection. No authentication required for development.

### Production Authentication (Planned)

For production deployments, configure API key authentication:

```bash
# Environment variables
export SAP_LLM_API_KEY_ENABLED=true
export SAP_LLM_API_KEYS="key1:secret1,key2:secret2"
```

Then include the API key in requests:

```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -H "X-API-Key: your-api-key" \
  -F "file=@invoice.pdf"
```

### OAuth 2.0 Integration (Future)

OAuth 2.0 integration with SAP Identity Service is planned for enterprise deployments.

## API Endpoints

### General Endpoints

#### GET /

Root endpoint providing API information.

**Response:**
```json
{
  "service": "SAP_LLM API",
  "version": "0.1.0",
  "status": "running",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

### Health & Monitoring

#### GET /health

Health check endpoint for load balancers.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-11-14T10:30:00Z",
  "components": {
    "api": "healthy",
    "models": "loaded",
    "pmg": "connected"
  }
}
```

**Use Cases:**
- Load balancer health checks
- Monitoring system integration
- Alerting triggers

#### GET /ready

Readiness check for Kubernetes probes.

**Response:**
```json
{
  "ready": true,
  "details": {
    "config": true,
    "unified_model": true,
    "pmg": true,
    "stages": true
  }
}
```

**Use Cases:**
- Kubernetes readiness probes
- Startup validation
- Rolling deployment verification

### Document Processing

#### POST /v1/extract

Upload document for asynchronous processing.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Rate Limit: 100 requests/minute

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | file | Yes | Document file (PDF, PNG, JPG, TIFF) |
| expected_type | string | No | Document type hint (invoice, purchase_order, etc.) |

**Response (HTTP 202):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Document queued for processing",
  "timestamp": "2025-11-14T10:30:00Z"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "expected_type=invoice"
```

#### POST /v1/extract/sync

Upload document for synchronous processing (wait for results).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Rate Limit: 20 requests/minute

**Parameters:** Same as `/v1/extract`

**Response (HTTP 200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "document_type": "invoice",
  "document_subtype": "standard_invoice",
  "extracted_data": {
    "invoice_number": "INV-2025-001",
    "invoice_date": "2025-11-14",
    "total_amount": 1250.00,
    "currency": "USD",
    "vendor_name": "Acme Corporation",
    "line_items": [
      {
        "description": "Professional Services",
        "quantity": 10,
        "unit_price": 125.00,
        "total": 1250.00
      }
    ]
  },
  "quality_score": 0.95,
  "confidence": 0.92,
  "routing_decision": {
    "action": "post",
    "target_system": "SAP_S4HANA",
    "priority": "normal"
  },
  "processing_time_ms": 1234.56,
  "timestamp": "2025-11-14T10:30:15Z"
}
```

**When to Use:**
- **Async** (`/v1/extract`): Batch processing, background systems, minimal API latency
- **Sync** (`/v1/extract/sync`): Interactive UIs, real-time validation, immediate results

### Job Management

#### GET /v1/jobs/{job_id}

Get job status and results.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.75,
  "current_stage": "validation",
  "timestamp": "2025-11-14T10:30:10Z"
}
```

**Status Values:**
- `queued`: Job waiting to be processed
- `processing`: Currently being processed
- `completed`: Processing finished successfully
- `failed`: Processing encountered an error

**Pipeline Stages:**
- `inbox` - 12.5% complete
- `preprocessing` - 25%
- `classification` - 37.5%
- `type_identifier` - 50%
- `extraction` - 62.5%
- `quality_check` - 75%
- `validation` - 87.5%
- `routing` - 90%

**Polling Strategy:**
1. Wait 1 second after job creation
2. Poll every 2-3 seconds while processing
3. Use 5-10 second intervals if queued
4. Consider WebSocket for real-time updates

#### DELETE /v1/jobs/{job_id}

Delete a job from server memory.

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 deleted"
}
```

**Notes:**
- Does not cancel active processing
- PMG historical data preserved
- Idempotent operation

### Monitoring Endpoints

#### GET /metrics

Prometheus metrics in exposition format.

**Metrics Include:**
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `document_processing_total`: Documents processed
- `processing_errors_total`: Processing errors

**Example:**
```bash
curl http://localhost:8000/metrics
```

#### GET /v1/slo

SLO status and error budgets.

**Response:**
```json
{
  "availability": {
    "target": 0.999,
    "current": 0.9995,
    "error_budget_remaining": 0.5,
    "status": "healthy"
  },
  "latency_p95": {
    "target_ms": 5000,
    "current_ms": 3450,
    "status": "healthy"
  }
}
```

#### GET /v1/stats

Real-time system statistics.

**Response:**
```json
{
  "jobs": {
    "total": 150,
    "completed": 142,
    "failed": 3,
    "processing": 2,
    "queued": 3
  },
  "system": {
    "models_loaded": true,
    "pmg_connected": true,
    "active_websockets": 5
  },
  "timestamp": "2025-11-14T10:30:00Z"
}
```

## WebSocket API

### Connection

Connect to WebSocket for real-time job updates:

```
ws://localhost:8000/v1/ws/{job_id}
```

### Message Format

All messages are JSON objects:

```json
{
  "status": "processing",
  "stage": "extraction",
  "progress": 0.625,
  "message": "Extracting fields..."
}
```

### Keep-Alive

Send ping messages:
```
Client → "ping"
Server → "pong"
```

### Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws/550e8400-e29b-41d4-a716-446655440000');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Status: ${update.status}, Progress: ${update.progress * 100}%`);

  if (update.status === 'completed') {
    console.log('Results:', update.result);
    ws.close();
  }
};

// Keep-alive
setInterval(() => ws.send('ping'), 30000);
```

## Request & Response Examples

### Invoice Processing

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/extract/sync" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "expected_type=invoice"
```

**Response:**
```json
{
  "job_id": "abc-123",
  "status": "completed",
  "document_type": "invoice",
  "extracted_data": {
    "invoice_number": "INV-2025-001",
    "invoice_date": "2025-11-14",
    "due_date": "2025-12-14",
    "vendor_name": "Acme Corporation",
    "vendor_address": "123 Main St, City, State 12345",
    "customer_name": "Your Company",
    "total_amount": 1250.00,
    "tax_amount": 125.00,
    "currency": "USD",
    "line_items": [
      {
        "line_number": 1,
        "description": "Professional Services",
        "quantity": 10,
        "unit_price": 125.00,
        "total": 1250.00
      }
    ]
  },
  "quality_score": 0.95,
  "confidence": 0.92
}
```

### Purchase Order Processing

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -F "file=@purchase_order.pdf" \
  -F "expected_type=purchase_order"
```

**Response:**
```json
{
  "job_id": "def-456",
  "status": "queued",
  "message": "Document queued for processing",
  "timestamp": "2025-11-14T10:30:00Z"
}
```

### Batch Processing

**Python Example:**
```python
import requests
from pathlib import Path

def process_batch(file_paths):
    url = "http://localhost:8000/v1/extract"
    job_ids = []

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            job_ids.append(response.json()['job_id'])

    return job_ids

# Process all PDFs in directory
files = list(Path('./invoices').glob('*.pdf'))
job_ids = process_batch(files)
print(f"Queued {len(job_ids)} documents")
```

## Error Handling

### Error Response Format

All errors follow consistent format:

```json
{
  "error": "Error message",
  "status_code": 400,
  "timestamp": "2025-11-14T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 202 | Accepted | Job queued successfully |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File exceeds 50MB limit |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service not ready |

### Common Errors

#### File Too Large

```json
{
  "error": "File size exceeds 50MB limit",
  "status_code": 413,
  "timestamp": "2025-11-14T10:30:00Z"
}
```

**Solution:** Compress file or split into smaller documents

#### Rate Limit Exceeded

```json
{
  "error": "Rate limit exceeded",
  "status_code": 429,
  "timestamp": "2025-11-14T10:30:00Z"
}
```

**Solution:** Wait and retry, or implement exponential backoff

#### Service Not Ready

```json
{
  "error": "Service not ready",
  "status_code": 503,
  "timestamp": "2025-11-14T10:30:00Z"
}
```

**Solution:** Wait for service initialization (check `/ready` endpoint)

### Error Handling Best Practices

**1. Implement Retry Logic**

```python
import time
import requests

def upload_with_retry(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    "http://localhost:8000/v1/extract",
                    files={'file': f}
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

**2. Handle Timeouts**

```python
response = requests.post(
    url,
    files=files,
    timeout=(5, 30)  # (connect timeout, read timeout)
)
```

**3. Validate Responses**

```python
if response.status_code == 202:
    job_id = response.json()['job_id']
    # Poll for results
elif response.status_code == 500:
    # Log error and retry
    pass
```

## Rate Limiting

### Rate Limits

| Endpoint | Limit | Per |
|----------|-------|-----|
| POST /v1/extract | 100 requests | minute per IP |
| POST /v1/extract/sync | 20 requests | minute per IP |
| GET /v1/jobs/{job_id} | Unlimited | - |
| GET /v1/stats | Unlimited | - |

### Rate Limit Headers

Response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699999999
```

### Handling Rate Limits

**Check Remaining Quota:**
```python
response = requests.post(url, files=files)
remaining = int(response.headers.get('X-RateLimit-Remaining', 0))

if remaining < 10:
    print(f"Warning: Only {remaining} requests remaining")
```

**Implement Backoff:**
```python
if response.status_code == 429:
    reset_time = int(response.headers.get('X-RateLimit-Reset'))
    wait_seconds = reset_time - time.time()
    time.sleep(max(0, wait_seconds))
```

## Code Examples

### Python

#### Basic Usage

```python
import requests

# Upload document
with open('invoice.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/extract',
        files={'file': f},
        data={'expected_type': 'invoice'}
    )

job_id = response.json()['job_id']
print(f"Job ID: {job_id}")

# Check status
import time
while True:
    status_response = requests.get(
        f'http://localhost:8000/v1/jobs/{job_id}'
    )
    data = status_response.json()

    if data['status'] == 'completed':
        print("Processing complete!")
        print(data['result']['extracted_data'])
        break
    elif data['status'] == 'failed':
        print(f"Processing failed: {data['error']}")
        break

    print(f"Progress: {data['progress'] * 100:.1f}%")
    time.sleep(2)
```

#### Async Processing

```python
import asyncio
import aiohttp

async def process_document(session, file_path):
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f)

        async with session.post(
            'http://localhost:8000/v1/extract',
            data=data
        ) as response:
            return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_document(session, f'invoice_{i}.pdf')
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        return results

results = asyncio.run(main())
```

### JavaScript

#### Fetch API

```javascript
// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('expected_type', 'invoice');

const response = await fetch('http://localhost:8000/v1/extract', {
  method: 'POST',
  body: formData
});

const { job_id } = await response.json();
console.log(`Job ID: ${job_id}`);

// Poll for results
async function pollJobStatus(jobId) {
  while (true) {
    const response = await fetch(`http://localhost:8000/v1/jobs/${jobId}`);
    const data = await response.json();

    if (data.status === 'completed') {
      console.log('Processing complete!');
      return data.result;
    } else if (data.status === 'failed') {
      throw new Error(data.error);
    }

    console.log(`Progress: ${data.progress * 100}%`);
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

const result = await pollJobStatus(job_id);
```

#### React Hook

```javascript
import { useState, useEffect } from 'react';

function useDocumentProcessing() {
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const processDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/v1/extract', {
        method: 'POST',
        body: formData
      });

      const { job_id } = await response.json();
      pollStatus(job_id);
    } catch (err) {
      setError(err.message);
    }
  };

  const pollStatus = async (jobId) => {
    const interval = setInterval(async () => {
      const response = await fetch(`http://localhost:8000/v1/jobs/${jobId}`);
      const data = await response.json();

      setStatus(data);

      if (data.status === 'completed') {
        setResult(data.result);
        clearInterval(interval);
      } else if (data.status === 'failed') {
        setError(data.error);
        clearInterval(interval);
      }
    }, 2000);
  };

  return { processDocument, status, result, error };
}
```

### cURL

#### Upload Document

```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "expected_type=invoice"
```

#### Check Status

```bash
curl http://localhost:8000/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```

#### Synchronous Processing

```bash
curl -X POST "http://localhost:8000/v1/extract/sync" \
  -F "file=@invoice.pdf" | jq .
```

#### Health Check

```bash
curl http://localhost:8000/health | jq .
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "time"
)

type JobResponse struct {
    JobID     string `json:"job_id"`
    Status    string `json:"status"`
    Message   string `json:"message"`
    Timestamp string `json:"timestamp"`
}

type StatusResponse struct {
    JobID        string                 `json:"job_id"`
    Status       string                 `json:"status"`
    Progress     float64                `json:"progress"`
    CurrentStage string                 `json:"current_stage"`
    Result       map[string]interface{} `json:"result"`
}

func uploadDocument(filePath string) (*JobResponse, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)

    part, err := writer.CreateFormFile("file", filePath)
    if err != nil {
        return nil, err
    }

    io.Copy(part, file)
    writer.Close()

    req, err := http.NewRequest("POST", "http://localhost:8000/v1/extract", body)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", writer.FormDataContentType())

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result JobResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}

func pollJobStatus(jobID string) (*StatusResponse, error) {
    for {
        resp, err := http.Get(fmt.Sprintf("http://localhost:8000/v1/jobs/%s", jobID))
        if err != nil {
            return nil, err
        }

        var status StatusResponse
        json.NewDecoder(resp.Body).Decode(&status)
        resp.Body.Close()

        if status.Status == "completed" {
            return &status, nil
        } else if status.Status == "failed" {
            return nil, fmt.Errorf("processing failed")
        }

        fmt.Printf("Progress: %.1f%%\n", status.Progress*100)
        time.Sleep(2 * time.Second)
    }
}

func main() {
    // Upload document
    job, err := uploadDocument("invoice.pdf")
    if err != nil {
        panic(err)
    }
    fmt.Printf("Job ID: %s\n", job.JobID)

    // Poll for results
    result, err := pollJobStatus(job.JobID)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Processing complete! Result: %+v\n", result.Result)
}
```

## Production Deployment

### Environment Variables

```bash
# Server Configuration
SAP_LLM_HOST=0.0.0.0
SAP_LLM_PORT=8000
SAP_LLM_WORKERS=4

# CORS Configuration
SAP_LLM_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Rate Limiting
SAP_LLM_RATE_LIMIT_ENABLED=true

# Model Configuration
SAP_LLM_MODEL_PATH=/models/unified_model

# Database
SAP_LLM_COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
SAP_LLM_COSMOS_KEY=your-key
SAP_LLM_COSMOS_DATABASE=sap_llm
SAP_LLM_COSMOS_CONTAINER=documents

# SAP Configuration
SAP_LLM_SAP_ENDPOINT=https://your-sap-system.com
SAP_LLM_SAP_CLIENT=100
SAP_LLM_SAP_USERNAME=api_user
SAP_LLM_SAP_PASSWORD=secure_password

# Monitoring
SAP_LLM_METRICS_ENABLED=true
SAP_LLM_PROMETHEUS_PORT=9090
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: sap-llm-api:latest
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - SAP_LLM_WORKERS=4
      - SAP_LLM_COSMOS_ENDPOINT=${COSMOS_ENDPOINT}
      - SAP_LLM_COSMOS_KEY=${COSMOS_KEY}
    volumes:
      - ./models:/models
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sap-llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sap-llm-api
  template:
    metadata:
      labels:
        app: sap-llm-api
    spec:
      containers:
      - name: api
        image: sap-llm-api:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: SAP_LLM_WORKERS
          value: "4"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sap-llm-api
spec:
  selector:
    app: sap-llm-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sap_llm_api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

#### Grafana Dashboard

Import dashboard JSON from `/monitoring/grafana/dashboard.json`

Key metrics to monitor:
- Request rate and latency
- Processing success/failure rate
- Queue depth
- Model inference time
- Error rates by type

### Security Considerations

1. **HTTPS Only**: Always use TLS in production
2. **API Keys**: Implement API key authentication
3. **Rate Limiting**: Configure appropriate limits
4. **Input Validation**: Validate file types and sizes
5. **Network Security**: Use VPC/firewall rules
6. **Secret Management**: Use environment variables or secret managers
7. **Audit Logging**: Enable request logging

### Performance Tuning

#### Recommended Settings

```python
# Uvicorn workers
workers = cpu_count() * 2 + 1

# Connection limits
max_connections = 1000
keepalive = 65

# Timeouts
timeout = 30
```

#### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -T 'multipart/form-data' \
   -p invoice.pdf \
   http://localhost:8000/v1/extract

# Using k6
k6 run load_test.js
```

### Troubleshooting

#### Common Issues

**1. High Memory Usage**
- Reduce number of workers
- Implement request queuing
- Scale horizontally

**2. Slow Processing**
- Check GPU availability
- Optimize model batch size
- Review pipeline stage timings

**3. Database Connection Issues**
- Verify Cosmos DB credentials
- Check network connectivity
- Review connection pool settings

**4. Rate Limit Errors**
- Increase rate limits
- Implement request queuing
- Add more API instances

## Support

### Documentation

- **API Reference**: http://localhost:8000/docs
- **Architecture**: `/docs/ARCHITECTURE.md`
- **Operations**: `/docs/OPERATIONS.md`
- **Troubleshooting**: `/docs/TROUBLESHOOTING.md`

### Community

- **GitHub**: https://github.com/your-org/SAP_LLM
- **Issues**: https://github.com/your-org/SAP_LLM/issues
- **Discussions**: https://github.com/your-org/SAP_LLM/discussions

### Contact

- **Email**: support@example.com
- **Slack**: #sap-llm-support

---

**Version**: 0.1.0
**Last Updated**: 2025-11-14
**License**: MIT
