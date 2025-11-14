"""
Example: Check Job Status

Demonstrates how to check the status of a document processing job
and retrieve results when complete.
"""

import requests
import sys
import time
from typing import Optional


def check_status(
    job_id: str,
    api_url: str = "http://localhost:8000"
) -> dict:
    """
    Check the status of a processing job.

    Args:
        job_id: Job identifier
        api_url: API base URL

    Returns:
        Job status information
    """
    url = f"{api_url}/v1/jobs/{job_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise ValueError(f"Job not found: {job_id}")
    else:
        response.raise_for_status()


def poll_until_complete(
    job_id: str,
    api_url: str = "http://localhost:8000",
    poll_interval: int = 2,
    timeout: int = 300
) -> dict:
    """
    Poll job status until completion or timeout.

    Args:
        job_id: Job identifier
        api_url: API base URL
        poll_interval: Seconds between polls
        timeout: Maximum wait time in seconds

    Returns:
        Final job status with results

    Raises:
        TimeoutError: If processing exceeds timeout
        ValueError: If processing fails
    """
    start_time = time.time()
    last_stage = None

    print(f"Polling job {job_id}...")
    print(f"Poll interval: {poll_interval}s, Timeout: {timeout}s\n")

    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Processing timeout after {elapsed:.1f}s")

        # Get status
        status = check_status(job_id, api_url)

        # Display progress
        if status['status'] == 'processing':
            progress = status.get('progress', 0)
            current_stage = status.get('current_stage', 'unknown')

            # Show stage transition
            if current_stage != last_stage:
                print(f"[{elapsed:.1f}s] Stage: {current_stage}")
                last_stage = current_stage

            # Progress bar
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  Progress: [{bar}] {progress*100:.1f}%", end='', flush=True)

        elif status['status'] == 'queued':
            print(f"\r[{elapsed:.1f}s] Status: Queued, waiting to start...", end='', flush=True)

        elif status['status'] == 'completed':
            print(f"\n\n✓ Processing completed in {elapsed:.1f}s!")
            return status

        elif status['status'] == 'failed':
            error = status.get('error', 'Unknown error')
            raise ValueError(f"Processing failed: {error}")

        # Wait before next poll
        time.sleep(poll_interval)


def display_results(status: dict):
    """Display extraction results in a readable format."""
    if status['status'] != 'completed':
        print(f"Status: {status['status']}")
        return

    result = status.get('result')
    if not result:
        print("No results available")
        return

    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    # Basic information
    print(f"\nJob ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    print(f"Timestamp: {result['timestamp']}")

    # Document information
    print(f"\nDocument Information:")
    print(f"  Type: {result.get('document_type', 'N/A')}")
    print(f"  Subtype: {result.get('document_subtype', 'N/A')}")

    # Quality metrics
    print(f"\nQuality Metrics:")
    if result.get('quality_score') is not None:
        print(f"  Quality Score: {result['quality_score']:.2%}")
    if result.get('confidence') is not None:
        print(f"  Confidence: {result['confidence']:.2%}")
    if result.get('processing_time_ms') is not None:
        print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")

    # Extracted data
    if result.get('extracted_data'):
        print(f"\nExtracted Data:")
        display_extracted_data(result['extracted_data'])

    # Routing decision
    if result.get('routing_decision'):
        print(f"\nRouting Decision:")
        for key, value in result['routing_decision'].items():
            print(f"  {key}: {value}")

    # SAP response
    if result.get('sap_response'):
        print(f"\nSAP Response:")
        for key, value in result['sap_response'].items():
            print(f"  {key}: {value}")

    # Exceptions
    if result.get('exceptions'):
        print(f"\nExceptions ({len(result['exceptions'])}):")
        for i, exc in enumerate(result['exceptions'], 1):
            print(f"  {i}. {exc.get('type', 'Unknown')}: {exc.get('message', 'N/A')}")


def display_extracted_data(data: dict, indent: int = 2):
    """Recursively display extracted data."""
    indent_str = " " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            display_extracted_data(value, indent + 2)
        elif isinstance(value, list):
            print(f"{indent_str}{key}:")
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    print(f"{indent_str}  [{i}]:")
                    display_extracted_data(item, indent + 4)
                else:
                    print(f"{indent_str}  [{i}] {item}")
        else:
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            print(f"{indent_str}{key}: {formatted_value}")


def main():
    """Main example demonstrating status checking."""
    # Example 1: Single status check
    print("=" * 60)
    print("Example 1: Single Status Check")
    print("=" * 60)

    example_job_id = "550e8400-e29b-41d4-a716-446655440000"
    print(f"\nChecking status for job: {example_job_id}")
    print("(This will fail if job doesn't exist)\n")

    try:
        status = check_status(example_job_id)
        print(f"Status: {status['status']}")
        if status.get('progress') is not None:
            print(f"Progress: {status['progress']*100:.1f}%")
        if status.get('current_stage'):
            print(f"Current Stage: {status['current_stage']}")
    except ValueError as e:
        print(f"✗ {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Example 2: Poll until complete
    print("\n" + "=" * 60)
    print("Example 2: Poll Until Complete")
    print("=" * 60)

    print("\nThis example polls for status until processing completes.")
    print("Run this after uploading a document:\n")
    print("  python example_upload_document.py sample.pdf")
    print("  python example_check_status.py <job_id>\n")

    # Example 3: Adaptive polling
    print("=" * 60)
    print("Example 3: Adaptive Polling Strategy")
    print("=" * 60)

    def adaptive_poll(job_id: str, api_url: str = "http://localhost:8000"):
        """
        Poll with adaptive intervals based on job status.

        - Queued: Poll every 5 seconds
        - Processing: Poll every 2 seconds
        - Long-running: Poll every 5 seconds
        """
        start_time = time.time()
        consecutive_processing = 0

        while True:
            status = check_status(job_id, api_url)
            elapsed = time.time() - start_time

            if status['status'] == 'completed':
                return status
            elif status['status'] == 'failed':
                raise ValueError(status.get('error'))

            # Determine poll interval
            if status['status'] == 'queued':
                interval = 5  # Longer interval when queued
            elif status['status'] == 'processing':
                consecutive_processing += 1
                # Increase interval if processing for long time
                interval = 5 if consecutive_processing > 30 else 2
            else:
                interval = 2

            print(f"[{elapsed:.1f}s] Status: {status['status']}, "
                  f"Progress: {status.get('progress', 0)*100:.1f}%, "
                  f"Next check in {interval}s")

            time.sleep(interval)

    print("\nAdaptive polling example (not executed):")
    print("  - Queued: Check every 5 seconds")
    print("  - Processing (< 1 min): Check every 2 seconds")
    print("  - Processing (> 1 min): Check every 5 seconds")

    # Example 4: Pipeline stage progress
    print("\n" + "=" * 60)
    print("Example 4: Pipeline Stage Progress")
    print("=" * 60)

    stages = [
        ("inbox", 0.125, "Document ingestion"),
        ("preprocessing", 0.250, "Image enhancement"),
        ("classification", 0.375, "Category detection"),
        ("type_identifier", 0.500, "Type identification"),
        ("extraction", 0.625, "Field extraction"),
        ("quality_check", 0.750, "Quality validation"),
        ("validation", 0.875, "Business rules"),
        ("routing", 0.900, "SAP routing"),
    ]

    print("\nProcessing Pipeline Stages:")
    print(f"{'Stage':<20} {'Progress':<10} {'Description':<30}")
    print("-" * 60)
    for stage, progress, description in stages:
        print(f"{stage:<20} {progress*100:>6.1f}%   {description:<30}")

    # Example 5: Comprehensive monitoring
    print("\n" + "=" * 60)
    print("Example 5: Comprehensive Job Monitoring")
    print("=" * 60)

    def monitor_job(job_id: str, api_url: str = "http://localhost:8000"):
        """
        Comprehensive job monitoring with statistics.
        """
        start_time = time.time()
        stage_times = {}
        last_stage = None

        while True:
            status = check_status(job_id, api_url)
            elapsed = time.time() - start_time

            if status['status'] == 'processing':
                current_stage = status.get('current_stage')
                if current_stage != last_stage and last_stage:
                    stage_times[last_stage] = elapsed
                last_stage = current_stage

            if status['status'] in ['completed', 'failed']:
                # Display statistics
                print("\n" + "=" * 60)
                print("Processing Statistics:")
                print(f"  Total Time: {elapsed:.2f}s")
                print(f"  Final Status: {status['status']}")

                if stage_times:
                    print("\n  Stage Durations:")
                    for stage, duration in stage_times.items():
                        print(f"    {stage}: {duration:.2f}s")

                return status

            time.sleep(2)

    print("\nComprehensive monitoring tracks:")
    print("  - Total processing time")
    print("  - Time spent in each stage")
    print("  - Real-time progress updates")
    print("  - Final statistics summary")


if __name__ == "__main__":
    # Allow command-line usage
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
        api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

        try:
            # Poll until complete
            status = poll_until_complete(job_id, api_url)

            # Display results
            display_results(status)

        except TimeoutError as e:
            print(f"\n✗ {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"\n✗ {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
    else:
        # Run examples
        main()
