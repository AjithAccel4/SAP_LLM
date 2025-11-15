# SAP_LLM API Examples

Comprehensive examples demonstrating how to use the SAP_LLM Document Processing API.

## Overview

This directory contains practical, production-ready examples for:

- Uploading documents for processing
- Checking job status
- Real-time tracking with WebSocket
- Batch processing multiple documents
- Error handling and retry logic
- Performance optimization

## Prerequisites

### Required

```bash
pip install requests
```

### Optional (for WebSocket and async examples)

```bash
pip install websockets aiohttp
```

## Quick Start

### 1. Start the API Server

```bash
# From project root
python -m sap_llm.api.server
```

Verify server is running:

```bash
curl http://localhost:8000/health
```

### 2. Upload a Document

```bash
python example_upload_document.py sample_invoice.pdf invoice
```

### 3. Check Status

```bash
python example_check_status.py <job_id>
```

### 4. Real-time Tracking (WebSocket)

```bash
python example_websocket_client.py <job_id>
```

### 5. Batch Processing

```bash
python example_batch_processing.py ./documents/ invoice
```

## Examples

### example_upload_document.py

**Upload documents for processing (async or sync)**

**Features:**
- Asynchronous upload (returns immediately with job_id)
- Synchronous upload (waits for results)
- Document type hints
- Error handling
- Retry logic examples

**Usage:**

```bash
# Interactive examples
python example_upload_document.py

# Command-line usage
python example_upload_document.py invoice.pdf invoice http://localhost:8000
```

**Python Code:**

```python
from example_upload_document import upload_document

# Async upload
result = upload_document("invoice.pdf", expected_type="invoice")
print(f"Job ID: {result['job_id']}")

# Sync upload (wait for results)
result = upload_document_sync("invoice.pdf", expected_type="invoice")
print(f"Extracted data: {result['extracted_data']}")
```

### example_check_status.py

**Check job status and poll for results**

**Features:**
- Single status check
- Polling until completion
- Adaptive polling intervals
- Progress tracking
- Pipeline stage monitoring
- Result display

**Usage:**

```bash
# Interactive examples
python example_check_status.py

# Command-line usage (polls until complete)
python example_check_status.py 550e8400-e29b-41d4-a716-446655440000
```

**Python Code:**

```python
from example_check_status import check_status, poll_until_complete

# Single check
status = check_status(job_id)
print(f"Status: {status['status']}, Progress: {status['progress']}")

# Poll until complete
result = poll_until_complete(job_id, poll_interval=2, timeout=300)
print(f"Extraction complete: {result['result']}")
```

### example_websocket_client.py

**Real-time job tracking via WebSocket**

**Features:**
- Real-time progress updates
- Keep-alive ping/pong
- Automatic reconnection
- Multiple concurrent jobs
- Connection error handling

**Usage:**

```bash
# Interactive examples
python example_websocket_client.py

# Command-line usage
python example_websocket_client.py 550e8400-e29b-41d4-a716-446655440000
```

**Python Code:**

```python
import asyncio
from example_websocket_client import track_job_websocket

# Track single job
result = asyncio.run(track_job_websocket(job_id))
print(f"Processing complete: {result}")

# Track multiple jobs
from example_websocket_client import track_multiple_jobs
job_ids = ["job-1", "job-2", "job-3"]
results = asyncio.run(track_multiple_jobs(job_ids))
```

### example_batch_processing.py

**Process multiple documents efficiently**

**Features:**
- Sequential processing
- Concurrent processing (ThreadPoolExecutor)
- Async processing (aiohttp)
- Progress tracking
- Performance comparison
- Results summary
- Directory processing

**Usage:**

```bash
# Interactive examples
python example_batch_processing.py

# Process directory
python example_batch_processing.py ./invoices/ invoice http://localhost:8000
```

**Python Code:**

```python
from example_batch_processing import BatchProcessor
from pathlib import Path

# Initialize processor
processor = BatchProcessor(max_workers=5)

# Find files
file_paths = list(Path('./invoices').glob('*.pdf'))

# Process concurrently
uploads = processor.process_batch_concurrent(file_paths, expected_type='invoice')

# Wait for completion
job_ids = [u['job_id'] for u in uploads if 'job_id' in u]
results = processor.wait_for_completion(job_ids)

# Generate summary
from example_batch_processing import generate_summary
summary = generate_summary(results)
print(f"Success rate: {summary['success_rate']:.1f}%")
```

## Common Workflows

### Workflow 1: Simple Document Processing

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

# Wait for results
import time
while True:
    status_response = requests.get(f'http://localhost:8000/v1/jobs/{job_id}')
    status = status_response.json()

    if status['status'] == 'completed':
        print("Processing complete!")
        print(status['result']['extracted_data'])
        break
    elif status['status'] == 'failed':
        print(f"Processing failed: {status['error']}")
        break

    time.sleep(2)
```

### Workflow 2: Real-time UI Integration

```python
import asyncio
import websockets
import json

async def process_with_updates(file_path, callback):
    """Process document with real-time updates."""
    # Upload document
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/v1/extract',
            files={'file': f}
        )

    job_id = response.json()['job_id']

    # Connect to WebSocket
    uri = f"ws://localhost:8000/v1/ws/{job_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            update = json.loads(message)

            # Call callback with update
            callback(update)

            if update['status'] in ['completed', 'failed']:
                break

# Usage
def update_ui(update):
    print(f"Progress: {update.get('progress', 0) * 100:.1f}%")

asyncio.run(process_with_updates('invoice.pdf', update_ui))
```

### Workflow 3: Batch Processing with Retry

```python
from example_batch_processing import BatchProcessor
from pathlib import Path
import time

processor = BatchProcessor(max_workers=5)
file_paths = list(Path('./documents').glob('*.pdf'))

# Upload with retry
max_retries = 3
uploads = []

for attempt in range(max_retries):
    try:
        uploads = processor.process_batch_concurrent(file_paths)
        break
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:  # Rate limit
            wait_time = 2 ** attempt
            print(f"Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise

# Wait for completion
job_ids = [u['job_id'] for u in uploads if 'job_id' in u]
results = processor.wait_for_completion(job_ids, timeout=600)

print(f"Processed {len(results)} documents")
```

## Error Handling

### Rate Limiting

```python
import time

def upload_with_rate_limit_handling(file_path):
    """Upload with automatic rate limit retry."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:8000/v1/extract',
                files={'file': open(file_path, 'rb')}
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Get retry-after header if available
                retry_after = int(e.response.headers.get('Retry-After', 2 ** attempt))
                print(f"Rate limit exceeded, retrying in {retry_after}s...")
                time.sleep(retry_after)
            else:
                raise

    raise Exception("Max retries exceeded")
```

### Connection Errors

```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retry():
    """Create requests session with automatic retry."""
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

# Usage
session = create_session_with_retry()
response = session.post('http://localhost:8000/v1/extract', ...)
```

### Timeout Handling

```python
try:
    response = requests.post(
        'http://localhost:8000/v1/extract/sync',
        files={'file': open('large_document.pdf', 'rb')},
        timeout=(5, 60)  # (connect timeout, read timeout)
    )
except requests.exceptions.Timeout:
    print("Request timed out, try async endpoint instead")
    # Fall back to async processing
    response = requests.post(
        'http://localhost:8000/v1/extract',
        files={'file': open('large_document.pdf', 'rb')}
    )
```

## Performance Tips

### 1. Use Concurrent Processing for Batches

```python
# Sequential: ~0.5 files/sec
processor.process_batch_sequential(files)

# Concurrent: ~2.5 files/sec (5x faster)
processor.process_batch_concurrent(files, max_workers=5)

# Async: ~2.9 files/sec (6x faster)
asyncio.run(process_batch_async(files, max_concurrent=10))
```

### 2. Choose the Right Endpoint

```python
# Async endpoint: Best for batches, returns immediately
response = requests.post('http://localhost:8000/v1/extract', ...)

# Sync endpoint: Best for single documents, immediate results
response = requests.post('http://localhost:8000/v1/extract/sync', ...)
```

### 3. Use WebSocket for Real-time Updates

```python
# Polling: Checks every 2 seconds, delays in updates
while True:
    status = check_status(job_id)
    time.sleep(2)

# WebSocket: Instant updates, more efficient
async with websockets.connect(f'ws://localhost:8000/v1/ws/{job_id}') as ws:
    async for message in ws:
        update = json.loads(message)
```

### 4. Optimize Worker Count

```python
import os

# CPU-bound: Use CPU count
cpu_workers = os.cpu_count()

# I/O-bound: Use higher count
io_workers = os.cpu_count() * 2

# For API uploads (I/O-bound)
processor = BatchProcessor(max_workers=io_workers)
```

## Testing

### Mock API Server

For testing without running the full server:

```python
from unittest.mock import Mock, patch
import requests

def test_upload():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 202
        mock_post.return_value.json.return_value = {
            'job_id': 'test-job-id',
            'status': 'queued'
        }

        result = upload_document('test.pdf')
        assert result['job_id'] == 'test-job-id'
```

### Integration Tests

```python
import pytest
from pathlib import Path

@pytest.mark.integration
def test_full_workflow():
    """Test complete document processing workflow."""
    # Upload
    result = upload_document('test_invoice.pdf', 'invoice')
    job_id = result['job_id']

    # Wait for completion
    status = poll_until_complete(job_id, timeout=60)

    # Verify results
    assert status['status'] == 'completed'
    assert status['result']['document_type'] == 'invoice'
    assert 'extracted_data' in status['result']
```

## Troubleshooting

### Common Issues

**1. Connection Refused**

```
Error: Connection refused
```

**Solution:** Ensure API server is running:

```bash
python -m sap_llm.api.server
curl http://localhost:8000/health
```

**2. Rate Limit Exceeded**

```
Error: 429 Too Many Requests
```

**Solution:** Reduce request rate or implement backoff:

```python
if response.status_code == 429:
    time.sleep(60)  # Wait 1 minute
```

**3. Job Not Found**

```
Error: 404 Job not found
```

**Solution:** Check job_id and timing:

```python
# Wait a moment after upload
time.sleep(1)
status = check_status(job_id)
```

**4. Processing Timeout**

```
Error: Processing timeout after 300s
```

**Solution:** Increase timeout or use async endpoint:

```python
# Increase timeout
result = poll_until_complete(job_id, timeout=600)

# Or use async
upload_document()  # Returns immediately
```

## Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Architecture Guide**: `../../docs/ARCHITECTURE.md`
- **API Reference**: `../../docs/API_DOCUMENTATION.md`
- **Operations Manual**: `../../docs/OPERATIONS.md`

## Support

For issues or questions:

- **GitHub Issues**: https://github.com/your-org/SAP_LLM/issues
- **Documentation**: http://localhost:8000/docs
- **Email**: support@example.com

## License

MIT License - See LICENSE file for details
