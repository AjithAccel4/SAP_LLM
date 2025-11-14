"""
Example: Upload Document for Processing

Demonstrates how to upload a document to the SAP_LLM API
for asynchronous processing.
"""

import requests
import sys
from pathlib import Path


def upload_document(
    file_path: str,
    expected_type: str = None,
    api_url: str = "http://localhost:8000"
) -> dict:
    """
    Upload a document for processing.

    Args:
        file_path: Path to document file
        expected_type: Optional document type hint (invoice, purchase_order, etc.)
        api_url: API base URL

    Returns:
        Response with job_id and status
    """
    # Validate file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Prepare request
    url = f"{api_url}/v1/extract"

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {}
        if expected_type:
            data['expected_type'] = expected_type

        # Upload document
        response = requests.post(url, files=files, data=data)

        # Check response
        if response.status_code == 202:
            result = response.json()
            print(f"✓ Document uploaded successfully!")
            print(f"  Job ID: {result['job_id']}")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
            return result
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            response.raise_for_status()


def upload_document_sync(
    file_path: str,
    expected_type: str = None,
    api_url: str = "http://localhost:8000"
) -> dict:
    """
    Upload a document and wait for processing to complete.

    Args:
        file_path: Path to document file
        expected_type: Optional document type hint
        api_url: API base URL

    Returns:
        Complete extraction results
    """
    # Validate file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Prepare request
    url = f"{api_url}/v1/extract/sync"

    print(f"Uploading {file_path.name} for synchronous processing...")

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {}
        if expected_type:
            data['expected_type'] = expected_type

        # Upload and wait
        response = requests.post(url, files=files, data=data)

        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Processing completed!")
            print(f"  Job ID: {result['job_id']}")
            print(f"  Document Type: {result['document_type']}")
            print(f"  Subtype: {result['document_subtype']}")
            print(f"  Quality Score: {result['quality_score']:.2%}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")

            # Display extracted data
            if result.get('extracted_data'):
                print(f"\n  Extracted Data:")
                for key, value in result['extracted_data'].items():
                    if isinstance(value, (str, int, float)):
                        print(f"    {key}: {value}")

            return result
        else:
            print(f"✗ Processing failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            response.raise_for_status()


def main():
    """Main example demonstrating document upload."""
    # Example 1: Async upload
    print("=" * 60)
    print("Example 1: Asynchronous Document Upload")
    print("=" * 60)

    # Upload invoice
    try:
        result = upload_document(
            file_path="sample_invoice.pdf",
            expected_type="invoice"
        )
        print(f"\nNext steps:")
        print(f"1. Use job_id '{result['job_id']}' to check status")
        print(f"2. Run: python example_check_status.py {result['job_id']}")
        print(f"3. Or use WebSocket: python example_websocket_client.py {result['job_id']}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"  Please provide a valid document file path")
        print(f"  Usage: python {sys.argv[0]} <file_path> [expected_type]")

    # Example 2: Sync upload
    print("\n" + "=" * 60)
    print("Example 2: Synchronous Document Upload (Wait for Results)")
    print("=" * 60)

    try:
        result = upload_document_sync(
            file_path="sample_invoice.pdf",
            expected_type="invoice"
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")

    # Example 3: Different document types
    print("\n" + "=" * 60)
    print("Example 3: Different Document Types")
    print("=" * 60)

    document_types = [
        ("invoice.pdf", "invoice"),
        ("purchase_order.pdf", "purchase_order"),
        ("receipt.jpg", "receipt"),
        ("delivery_note.pdf", "delivery_note")
    ]

    print("\nSupported document types:")
    for file_name, doc_type in document_types:
        print(f"  - {doc_type}: {file_name}")

    # Example 4: Error handling
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    print("\nCommon errors and solutions:")
    print("  1. File too large (>50MB)")
    print("     Solution: Compress or split the document")
    print("  2. Rate limit exceeded (429)")
    print("     Solution: Wait and retry with exponential backoff")
    print("  3. Service not ready (503)")
    print("     Solution: Check /ready endpoint and wait for initialization")

    # Example 5: With retry logic
    print("\n" + "=" * 60)
    print("Example 5: Upload with Retry Logic")
    print("=" * 60)

    def upload_with_retry(file_path, max_retries=3):
        """Upload with exponential backoff retry."""
        import time

        for attempt in range(max_retries):
            try:
                return upload_document(file_path)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif e.response.status_code == 503:  # Service unavailable
                    print(f"  Service not ready, waiting 5s before retry...")
                    time.sleep(5)
                else:
                    raise

        raise Exception("Max retries exceeded")

    print("\nRetry logic example (not executed):")
    print("  try:")
    print("      result = upload_with_retry('document.pdf', max_retries=3)")
    print("  except Exception as e:")
    print("      print(f'Upload failed after retries: {e}')")


if __name__ == "__main__":
    # Allow command-line usage
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        expected_type = sys.argv[2] if len(sys.argv) > 2 else None
        api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"

        print(f"Uploading document: {file_path}")
        if expected_type:
            print(f"Expected type: {expected_type}")

        try:
            result = upload_document(file_path, expected_type, api_url)
            print(f"\nJob ID: {result['job_id']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Run examples
        main()
