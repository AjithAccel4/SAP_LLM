"""
Example: WebSocket Client for Real-time Updates

Demonstrates how to use WebSocket for real-time job status updates.
More efficient than polling for long-running jobs.
"""

import asyncio
import json
import sys
import time
from typing import Optional

try:
    import websockets
except ImportError:
    print("Error: websockets library not installed")
    print("Install with: pip install websockets")
    sys.exit(1)


async def track_job_websocket(
    job_id: str,
    api_url: str = "ws://localhost:8000"
) -> dict:
    """
    Track job progress via WebSocket.

    Args:
        job_id: Job identifier
        api_url: WebSocket API base URL

    Returns:
        Final processing result
    """
    # Construct WebSocket URL
    ws_url = f"{api_url}/v1/ws/{job_id}"

    print(f"Connecting to WebSocket: {ws_url}")
    start_time = time.time()

    try:
        async with websockets.connect(ws_url) as websocket:
            print("✓ Connected\n")

            # Keep-alive task
            async def send_keepalive():
                while True:
                    try:
                        await asyncio.sleep(30)
                        await websocket.send("ping")
                    except:
                        break

            keepalive_task = asyncio.create_task(send_keepalive())

            # Receive updates
            last_stage = None

            try:
                while True:
                    message = await websocket.recv()
                    elapsed = time.time() - start_time

                    # Handle pong response
                    if message == "pong":
                        continue

                    # Parse JSON update
                    try:
                        update = json.loads(message)
                    except json.JSONDecodeError:
                        print(f"Received non-JSON message: {message}")
                        continue

                    # Display update
                    status = update.get('status')
                    stage = update.get('stage')
                    progress = update.get('progress', 0)

                    if stage and stage != last_stage:
                        print(f"[{elapsed:6.1f}s] Stage: {stage}")
                        last_stage = stage

                    # Progress bar
                    if progress is not None:
                        bar_length = 40
                        filled = int(bar_length * progress)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r  Progress: [{bar}] {progress*100:.1f}%", end='', flush=True)

                    # Check for completion
                    if status == 'completed':
                        print(f"\n\n✓ Processing completed in {elapsed:.1f}s!")
                        keepalive_task.cancel()
                        return update.get('result')

                    elif status == 'failed':
                        error = update.get('error', 'Unknown error')
                        print(f"\n\n✗ Processing failed: {error}")
                        keepalive_task.cancel()
                        raise ValueError(error)

            except websockets.exceptions.ConnectionClosed:
                print("\n\nWebSocket connection closed")
                keepalive_task.cancel()

    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
        raise


async def track_multiple_jobs(
    job_ids: list,
    api_url: str = "ws://localhost:8000"
):
    """
    Track multiple jobs concurrently via WebSocket.

    Args:
        job_ids: List of job identifiers
        api_url: WebSocket API base URL
    """
    print(f"Tracking {len(job_ids)} jobs concurrently...\n")

    async def track_single(job_id: str, index: int):
        """Track a single job."""
        ws_url = f"{api_url}/v1/ws/{job_id}"
        start_time = time.time()

        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    message = await websocket.recv()

                    if message == "pong":
                        continue

                    try:
                        update = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    status = update.get('status')
                    progress = update.get('progress', 0)
                    elapsed = time.time() - start_time

                    print(f"[Job {index+1}] Status: {status}, "
                          f"Progress: {progress*100:.1f}%, "
                          f"Time: {elapsed:.1f}s")

                    if status in ['completed', 'failed']:
                        return {
                            'job_id': job_id,
                            'status': status,
                            'result': update.get('result'),
                            'time': elapsed
                        }

        except Exception as e:
            return {
                'job_id': job_id,
                'status': 'error',
                'error': str(e)
            }

    # Track all jobs concurrently
    tasks = [track_single(job_id, i) for i, job_id in enumerate(job_ids)]
    results = await asyncio.gather(*tasks)

    # Display summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\nJob {i} ({result['job_id']}):")
        print(f"  Status: {result['status']}")
        if result.get('time'):
            print(f"  Time: {result['time']:.2f}s")

    return results


async def track_with_reconnect(
    job_id: str,
    api_url: str = "ws://localhost:8000",
    max_retries: int = 3
) -> dict:
    """
    Track job with automatic reconnection on failure.

    Args:
        job_id: Job identifier
        api_url: WebSocket API base URL
        max_retries: Maximum reconnection attempts

    Returns:
        Final processing result
    """
    for attempt in range(max_retries):
        try:
            return await track_job_websocket(job_id, api_url)
        except websockets.exceptions.WebSocketException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nConnection failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"\nMax retries exceeded")
                raise


def main():
    """Main example demonstrating WebSocket usage."""
    # Example 1: Single job tracking
    print("=" * 60)
    print("Example 1: Real-time Job Tracking via WebSocket")
    print("=" * 60)

    print("\nWebSocket provides real-time updates without polling.")
    print("More efficient for long-running jobs.\n")

    example_job_id = "550e8400-e29b-41d4-a716-446655440000"
    print(f"Example job ID: {example_job_id}")
    print("(This will fail if job doesn't exist)\n")

    print("To use:")
    print("  1. Upload a document:")
    print("     python example_upload_document.py sample.pdf")
    print("  2. Track via WebSocket:")
    print("     python example_websocket_client.py <job_id>\n")

    # Example 2: Multiple concurrent jobs
    print("=" * 60)
    print("Example 2: Track Multiple Jobs Concurrently")
    print("=" * 60)

    print("\nWebSocket allows tracking multiple jobs efficiently.")
    print("Each job gets its own connection with real-time updates.\n")

    example_jobs = [
        "job-1-id",
        "job-2-id",
        "job-3-id"
    ]

    print("Example usage:")
    print("  job_ids = ['job-1', 'job-2', 'job-3']")
    print("  asyncio.run(track_multiple_jobs(job_ids))")

    # Example 3: Connection features
    print("\n" + "=" * 60)
    print("Example 3: WebSocket Features")
    print("=" * 60)

    print("\nKey Features:")
    print("  1. Real-time Updates: Instant notification of progress")
    print("  2. No Polling Overhead: Server pushes updates")
    print("  3. Keep-alive: Automatic ping/pong to maintain connection")
    print("  4. Automatic Cleanup: Connection closes on completion")
    print("  5. Multiple Clients: Multiple clients can track same job")

    print("\nMessage Types:")
    print("  - Status Updates: Job status changes")
    print("  - Progress Updates: Processing progress (0-1)")
    print("  - Stage Updates: Pipeline stage transitions")
    print("  - Completion: Final results when done")
    print("  - Keep-alive: Ping/pong heartbeat")

    # Example 4: Error handling
    print("\n" + "=" * 60)
    print("Example 4: Error Handling & Reconnection")
    print("=" * 60)

    print("\nWebSocket connections can fail. Best practices:")
    print("  1. Implement reconnection with exponential backoff")
    print("  2. Handle connection timeouts gracefully")
    print("  3. Fall back to polling if WebSocket unavailable")
    print("  4. Monitor connection health with keep-alive")

    # Example 5: Comparison with polling
    print("\n" + "=" * 60)
    print("Example 5: WebSocket vs Polling")
    print("=" * 60)

    print("\nPolling (REST API):")
    print("  ✓ Simple to implement")
    print("  ✓ Works with any HTTP client")
    print("  ✗ Delays in receiving updates")
    print("  ✗ Higher server load")
    print("  ✗ Wastes bandwidth with repeated requests")

    print("\nWebSocket:")
    print("  ✓ Real-time updates")
    print("  ✓ Lower server load")
    print("  ✓ Efficient bandwidth usage")
    print("  ✗ Requires WebSocket client library")
    print("  ✗ More complex error handling")

    print("\nRecommendation:")
    print("  - Use WebSocket for real-time UIs and long jobs")
    print("  - Use Polling for simple integrations and batch processing")

    # Example 6: JavaScript example
    print("\n" + "=" * 60)
    print("Example 6: JavaScript WebSocket Client")
    print("=" * 60)

    javascript_example = """
const ws = new WebSocket('ws://localhost:8000/v1/ws/job-id');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);

  console.log(`Status: ${update.status}`);
  console.log(`Progress: ${update.progress * 100}%`);

  if (update.status === 'completed') {
    console.log('Results:', update.result);
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};

// Keep-alive
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 30000);
    """

    print("\nJavaScript Example:")
    print(javascript_example)


if __name__ == "__main__":
    # Allow command-line usage
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
        api_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000"

        try:
            # Track job via WebSocket
            result = asyncio.run(track_job_websocket(job_id, api_url))

            # Display results
            if result:
                print("\n" + "=" * 60)
                print("FINAL RESULTS")
                print("=" * 60)
                print(json.dumps(result, indent=2))

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
    else:
        # Run examples
        main()
