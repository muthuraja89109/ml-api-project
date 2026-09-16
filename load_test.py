import asyncio
import time
import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "9c3ee05ef58a17be33bd620d46b33e19ae568574eb11a4ed3802fbad6a72067f"
HEADERS = {"X-API-Key": API_KEY}

PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

NUM_REQUESTS = 100


async def send_one_request(client, index):
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{BASE_URL}/api/v1/predict",
            json=PAYLOAD,
            headers=HEADERS,
            timeout=10.0,
        )
        duration = time.perf_counter() - start
        return {
            "index": index,
            "status": response.status_code,
            "duration": duration,
            "error": None,
        }
    except Exception as exc:
        duration = time.perf_counter() - start
        return {
            "index": index,
            "status": None,
            "duration": duration,
            "error": str(exc),
        }


async def main():
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = [send_one_request(client, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start

    successes = [r for r in results if r["status"] == 200]
    failures = [r for r in results if r["status"] != 200]
    durations = [r["duration"] for r in results]

    print(f"\n{'='*50}")
    print(f"LOAD TEST RESULTS — {NUM_REQUESTS} concurrent requests")
    print(f"{'='*50}")
    print(f"Total wall-clock time : {total_duration:.2f}s")
    print(f"Successful (200)      : {len(successes)}")
    print(f"Failed                : {len(failures)}")
    print(f"Avg response time     : {sum(durations)/len(durations):.4f}s")
    print(f"Min response time     : {min(durations):.4f}s")
    print(f"Max response time     : {max(durations):.4f}s")

    if failures:
        print(f"\nFailure details (first 5):")
        for f in failures[:5]:
            print(f"  index={f['index']} status={f['status']} error={f['error']}")


if __name__ == "__main__":
    asyncio.run(main())