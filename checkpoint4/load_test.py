"""
MedScan Diagnostics - Checkpoint 4
50-concurrent-request load test.

Start the API first:

    uvicorn checkpoint4.api:app --host 127.0.0.1 --port 8000

Then run:

    python checkpoint4/load_test.py

Before running, set the path to an approved test image:

Windows PowerShell:

    $env:PREDICTION_IMAGE="data/raw/normal/example.png"

Command Prompt:

    set PREDICTION_IMAGE=data/raw/normal/example.png

Linux/macOS:

    export PREDICTION_IMAGE=data/raw/normal/example.png

The script sends 50 concurrent prediction requests and
reports latency and throughput.
"""

import asyncio
import os
import statistics
import time

import httpx


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_URL = os.getenv(
    "MEDSCAN_API_URL",
    "http://127.0.0.1:8000",
)

CONCURRENT_REQUESTS = 50

TIMEOUT_SECONDS = 60

IMAGE_PATH = os.getenv(
    "PREDICTION_IMAGE"
)


# -------------------------------------------------------------------
# Load test image
# -------------------------------------------------------------------

def load_image():
    """
    Read the image that will be submitted to /predict.
    """

    if not IMAGE_PATH:
        raise SystemExit(
            "PREDICTION_IMAGE is not set.\n"
            "Example:\n"
            "PREDICTION_IMAGE=data/raw/normal/example.png"
        )

    if not os.path.exists(IMAGE_PATH):
        raise SystemExit(
            f"Prediction image does not exist: "
            f"{IMAGE_PATH}"
        )

    with open(
        IMAGE_PATH,
        "rb",
    ) as file:

        return file.read()


# -------------------------------------------------------------------
# Send prediction request
# -------------------------------------------------------------------

async def send_prediction_request(
    client,
    image_bytes,
):
    """
    Send one prediction request and record latency.
    """

    start_time = time.perf_counter()

    try:

        response = await client.post(
            f"{BASE_URL}/predict",
            files={
                "file": (
                    "load_test.png",
                    image_bytes,
                    "image/png",
                )
            },
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "status": response.status_code,
            "latency": elapsed,
            "error": None,
        }

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "status": 0,
            "latency": elapsed,
            "error": str(exc),
        }


# -------------------------------------------------------------------
# Main load test
# -------------------------------------------------------------------

async def main():

    image_bytes = load_image()

    limits = httpx.Limits(
        max_connections=CONCURRENT_REQUESTS,
        max_keepalive_connections=CONCURRENT_REQUESTS,
    )

    timeout = httpx.Timeout(
        TIMEOUT_SECONDS
    )

    print()
    print("=" * 60)
    print("MedScan Diagnostics - Checkpoint 4 Load Test")
    print("=" * 60)

    print(
        f"API endpoint: {BASE_URL}/predict"
    )

    print(
        f"Concurrent requests: "
        f"{CONCURRENT_REQUESTS}"
    )

    print(
        f"Test image: {IMAGE_PATH}"
    )

    print()
    print("Sending requests...")

    # ---------------------------------------------------------------
    # Create HTTP client
    # ---------------------------------------------------------------

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:

        start_time = time.perf_counter()

        tasks = [
            send_prediction_request(
                client,
                image_bytes,
            )
            for _ in range(
                CONCURRENT_REQUESTS
            )
        ]

        results = await asyncio.gather(
            *tasks
        )

        total_time = (
            time.perf_counter()
            - start_time
        )

    # ---------------------------------------------------------------
    # Calculate statistics
    # ---------------------------------------------------------------

    latencies = [
        result["latency"]
        for result in results
    ]

    successful = sum(
        result["status"] == 200
        for result in results
    )

    failed = (
        CONCURRENT_REQUESTS
        - successful
    )

    mean_latency = statistics.mean(
        latencies
    )

    median_latency = statistics.median(
        latencies
    )

    minimum_latency = min(
        latencies
    )

    maximum_latency = max(
        latencies
    )

    throughput = (
        CONCURRENT_REQUESTS
        / total_time
    )

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)

    print(
        f"Total requests       : "
        f"{CONCURRENT_REQUESTS}"
    )

    print(
        f"Successful requests  : "
        f"{successful}"
    )

    print(
        f"Failed requests      : "
        f"{failed}"
    )

    print(
        f"Total wall time      : "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Mean latency         : "
        f"{mean_latency:.4f} seconds"
    )

    print(
        f"Median latency       : "
        f"{median_latency:.4f} seconds"
    )

    print(
        f"Minimum latency      : "
        f"{minimum_latency:.4f} seconds"
    )

    print(
        f"Maximum latency      : "
        f"{maximum_latency:.4f} seconds"
    )

    print(
        f"Throughput           : "
        f"{throughput:.2f} requests/second"
    )

    print("=" * 60)

    # ---------------------------------------------------------------
    # Errors
    # ---------------------------------------------------------------

    errors = [
        result
        for result in results
        if result["error"] is not None
    ]

    if errors:

        print()
        print("ERRORS:")

        for error in errors[:10]:

            print(
                error["error"]
            )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    asyncio.run(main())