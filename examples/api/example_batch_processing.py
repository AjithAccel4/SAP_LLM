"""
Example: Batch Document Processing

Demonstrates how to process multiple documents efficiently
using the SAP_LLM API with concurrent requests and progress tracking.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import aiohttp
except ImportError:
    print("Warning: aiohttp not installed. Async examples will not work.")
    print("Install with: pip install aiohttp")
    aiohttp = None


class BatchProcessor:
    """Batch document processor with concurrent uploads and tracking."""

    def __init__(self, api_url: str = "http://localhost:8000", max_workers: int = 5):
        """
        Initialize batch processor.

        Args:
            api_url: API base URL
            max_workers: Maximum concurrent requests
        """
        self.api_url = api_url
        self.max_workers = max_workers
        self.results = []

    def upload_file(self, file_path: Path, expected_type: Optional[str] = None) -> Dict:
        """
        Upload a single file.

        Args:
            file_path: Path to document file
            expected_type: Optional document type hint

        Returns:
            Upload response with job_id
        """
        url = f"{self.api_url}/v1/extract"

        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'expected_type': expected_type} if expected_type else {}

            response = requests.post(url, files=files, data=data)
            response.raise_for_status()

            result = response.json()
            result['file_name'] = file_path.name
            return result

    def check_status(self, job_id: str) -> Dict:
        """
        Check job status.

        Args:
            job_id: Job identifier

        Returns:
            Job status
        """
        url = f"{self.api_url}/v1/jobs/{job_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def process_batch_sequential(
        self,
        file_paths: List[Path],
        expected_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Process files sequentially (one at a time).

        Args:
            file_paths: List of file paths
            expected_type: Optional document type hint

        Returns:
            List of results
        """
        print(f"Processing {len(file_paths)} files sequentially...")
        results = []
        start_time = time.time()

        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] Uploading {file_path.name}...", end=' ')

            try:
                result = self.upload_file(file_path, expected_type)
                results.append(result)
                print(f"✓ Job ID: {result['job_id']}")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'file_name': file_path.name,
                    'error': str(e),
                    'status': 'failed'
                })

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Throughput: {len(file_paths)/elapsed:.2f} files/sec")

        return results

    def process_batch_concurrent(
        self,
        file_paths: List[Path],
        expected_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Process files concurrently using ThreadPoolExecutor.

        Args:
            file_paths: List of file paths
            expected_type: Optional document type hint

        Returns:
            List of results
        """
        print(f"Processing {len(file_paths)} files with {self.max_workers} workers...")
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all uploads
            future_to_file = {
                executor.submit(self.upload_file, fp, expected_type): fp
                for fp in file_paths
            }

            # Process as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_path = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{i}/{len(file_paths)}] ✓ {file_path.name} → {result['job_id']}")
                except Exception as e:
                    print(f"[{i}/{len(file_paths)}] ✗ {file_path.name} → Error: {e}")
                    results.append({
                        'file_name': file_path.name,
                        'error': str(e),
                        'status': 'failed'
                    })

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Throughput: {len(file_paths)/elapsed:.2f} files/sec")

        return results

    def wait_for_completion(
        self,
        job_ids: List[str],
        poll_interval: int = 2,
        timeout: int = 600
    ) -> List[Dict]:
        """
        Wait for all jobs to complete.

        Args:
            job_ids: List of job identifiers
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            List of final results
        """
        print(f"\nWaiting for {len(job_ids)} jobs to complete...")
        start_time = time.time()
        pending = set(job_ids)
        results = {}

        while pending:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Timeout after {elapsed:.1f}s")
                break

            # Check status of all pending jobs
            for job_id in list(pending):
                try:
                    status = self.check_status(job_id)

                    if status['status'] == 'completed':
                        results[job_id] = status
                        pending.remove(job_id)
                        print(f"✓ Job {job_id[:8]}... completed")

                    elif status['status'] == 'failed':
                        results[job_id] = status
                        pending.remove(job_id)
                        print(f"✗ Job {job_id[:8]}... failed")

                except Exception as e:
                    print(f"✗ Error checking {job_id[:8]}...: {e}")

            if pending:
                # Display progress
                completed = len(results)
                total = len(job_ids)
                progress = completed / total * 100
                print(f"\rProgress: {completed}/{total} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s", end='', flush=True)

                time.sleep(poll_interval)

        print()  # New line after progress
        return list(results.values())


async def process_batch_async(
    file_paths: List[Path],
    api_url: str = "http://localhost:8000",
    expected_type: Optional[str] = None,
    max_concurrent: int = 5
) -> List[Dict]:
    """
    Process files asynchronously using aiohttp.

    Args:
        file_paths: List of file paths
        api_url: API base URL
        expected_type: Optional document type hint
        max_concurrent: Maximum concurrent requests

    Returns:
        List of results
    """
    if aiohttp is None:
        raise ImportError("aiohttp is required for async processing")

    print(f"Processing {len(file_paths)} files asynchronously...")
    start_time = time.time()
    results = []

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_file(session: aiohttp.ClientSession, file_path: Path) -> Dict:
        """Upload a single file asynchronously."""
        async with semaphore:
            url = f"{api_url}/v1/extract"

            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=file_path.name)
                if expected_type:
                    data.add_field('expected_type', expected_type)

                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    result['file_name'] = file_path.name
                    return result

    # Upload all files concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [upload_file(session, fp) for fp in file_paths]

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await coro
                results.append(result)
                print(f"[{i}/{len(file_paths)}] ✓ {result['file_name']} → {result['job_id']}")
            except Exception as e:
                print(f"[{i}/{len(file_paths)}] ✗ Error: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Throughput: {len(file_paths)/elapsed:.2f} files/sec")

    return results


def generate_summary(results: List[Dict]) -> Dict:
    """
    Generate processing summary statistics.

    Args:
        results: List of processing results

    Returns:
        Summary statistics
    """
    total = len(results)
    completed = sum(1 for r in results if r.get('status') == 'completed')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    queued = sum(1 for r in results if r.get('status') == 'queued')

    # Processing times
    processing_times = [
        r.get('result', {}).get('processing_time_ms', 0)
        for r in results
        if r.get('status') == 'completed'
    ]

    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return {
        'total': total,
        'completed': completed,
        'failed': failed,
        'queued': queued,
        'success_rate': completed / total * 100 if total > 0 else 0,
        'avg_processing_time_ms': avg_time
    }


def save_results(results: List[Dict], output_path: Path):
    """
    Save results to JSON file.

    Args:
        results: Processing results
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    """Main example demonstrating batch processing."""
    # Example 1: Sequential processing
    print("=" * 60)
    print("Example 1: Sequential Batch Processing")
    print("=" * 60)

    print("\nSequential processing uploads one file at a time.")
    print("Simple but slower for large batches.\n")

    # Example file list
    example_files = [
        Path("invoice_001.pdf"),
        Path("invoice_002.pdf"),
        Path("invoice_003.pdf")
    ]

    print("Example usage:")
    print("  processor = BatchProcessor()")
    print("  results = processor.process_batch_sequential(file_paths)")

    # Example 2: Concurrent processing
    print("\n" + "=" * 60)
    print("Example 2: Concurrent Batch Processing")
    print("=" * 60)

    print("\nConcurrent processing uploads multiple files simultaneously.")
    print("Much faster for large batches, respects rate limits.\n")

    print("Example usage:")
    print("  processor = BatchProcessor(max_workers=5)")
    print("  results = processor.process_batch_concurrent(file_paths)")

    # Example 3: Async processing
    print("\n" + "=" * 60)
    print("Example 3: Async Batch Processing")
    print("=" * 60)

    print("\nAsync processing using aiohttp for maximum efficiency.")
    print("Best performance for very large batches.\n")

    print("Example usage:")
    print("  results = asyncio.run(")
    print("      process_batch_async(file_paths, max_concurrent=10)")
    print("  )")

    # Example 4: Performance comparison
    print("\n" + "=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)

    print("\nFor 100 files:")
    print("  Sequential:  ~200s (0.5 files/sec)")
    print("  Concurrent:  ~40s  (2.5 files/sec)")
    print("  Async:       ~35s  (2.9 files/sec)")

    print("\nRecommendations:")
    print("  - Sequential: < 10 files, simple scripts")
    print("  - Concurrent: 10-100 files, balanced approach")
    print("  - Async: > 100 files, maximum performance")

    # Example 5: Complete workflow
    print("\n" + "=" * 60)
    print("Example 5: Complete Batch Workflow")
    print("=" * 60)

    print("\nComplete workflow with progress tracking:")
    print("""
    # 1. Upload all files
    processor = BatchProcessor(max_workers=5)
    uploads = processor.process_batch_concurrent(file_paths)

    # 2. Extract job IDs
    job_ids = [u['job_id'] for u in uploads if 'job_id' in u]

    # 3. Wait for completion
    results = processor.wait_for_completion(job_ids, timeout=600)

    # 4. Generate summary
    summary = generate_summary(results)
    print(f"Completed: {summary['completed']}/{summary['total']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    # 5. Save results
    save_results(results, Path('batch_results.json'))
    """)

    # Example 6: Error handling
    print("\n" + "=" * 60)
    print("Example 6: Error Handling in Batch Processing")
    print("=" * 60)

    print("\nBest practices for batch processing:")
    print("  1. Implement retry logic for failed uploads")
    print("  2. Save progress periodically")
    print("  3. Handle rate limit errors gracefully")
    print("  4. Log all errors for debugging")
    print("  5. Provide progress updates to users")

    # Example 7: Directory processing
    print("\n" + "=" * 60)
    print("Example 7: Process Entire Directory")
    print("=" * 60)

    print("\nProcess all documents in a directory:")
    print("""
    from pathlib import Path

    # Find all PDFs
    directory = Path('./invoices')
    pdf_files = list(directory.glob('**/*.pdf'))

    # Process in batches of 50
    batch_size = 50
    processor = BatchProcessor(max_workers=10)

    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        results = processor.process_batch_concurrent(batch)

        # Save batch results
        save_results(results, Path(f'batch_{i//batch_size + 1}.json'))
    """)


if __name__ == "__main__":
    # Allow command-line usage
    if len(sys.argv) > 1:
        # Process directory of files
        directory = Path(sys.argv[1])
        expected_type = sys.argv[2] if len(sys.argv) > 2 else None
        api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"

        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)

        # Find all supported files
        patterns = ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tiff']
        file_paths = []
        for pattern in patterns:
            file_paths.extend(directory.glob(pattern))

        if not file_paths:
            print(f"No documents found in {directory}")
            sys.exit(1)

        print(f"Found {len(file_paths)} documents in {directory}")
        print(f"Processing with API: {api_url}")
        if expected_type:
            print(f"Expected type: {expected_type}")

        # Process batch
        processor = BatchProcessor(api_url=api_url, max_workers=5)

        try:
            # Upload files
            print("\n" + "=" * 60)
            print("UPLOADING FILES")
            print("=" * 60)
            uploads = processor.process_batch_concurrent(file_paths, expected_type)

            # Extract job IDs
            job_ids = [u['job_id'] for u in uploads if 'job_id' in u]

            if not job_ids:
                print("No files uploaded successfully")
                sys.exit(1)

            # Wait for completion
            print("\n" + "=" * 60)
            print("WAITING FOR COMPLETION")
            print("=" * 60)
            results = processor.wait_for_completion(job_ids)

            # Generate summary
            print("\n" + "=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)
            summary = generate_summary(results)
            print(f"Total Files: {summary['total']}")
            print(f"Completed: {summary['completed']}")
            print(f"Failed: {summary['failed']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            if summary['avg_processing_time_ms'] > 0:
                print(f"Avg Processing Time: {summary['avg_processing_time_ms']:.2f}ms")

            # Save results
            output_path = directory / 'batch_results.json'
            save_results(results, output_path)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
    else:
        # Run examples
        main()
        print("\n" + "=" * 60)
        print("USAGE")
        print("=" * 60)
        print("\nTo process a directory of documents:")
        print("  python example_batch_processing.py <directory> [expected_type] [api_url]")
        print("\nExample:")
        print("  python example_batch_processing.py ./invoices invoice http://localhost:8000")
