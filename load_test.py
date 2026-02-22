"""
Load test: sends N concurrent requests to the background removal API.
Usage: python load_test.py
"""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/api/remove-bg"
IMAGE_FILE = "test_portrait.jpg"
NUM_REQUESTS = 100  # Simulate 100 concurrent users


def send_request(i):
    start = time.time()
    try:
        with open(IMAGE_FILE, "rb") as f:
            resp = requests.post(URL, files={"image": f}, timeout=60)
        elapsed = time.time() - start
        return i, resp.status_code, elapsed, len(resp.content)
    except Exception as e:
        elapsed = time.time() - start
        return i, 0, elapsed, str(e)


if __name__ == "__main__":
    print(f"Sending {NUM_REQUESTS} concurrent requests to {URL}...")
    t0 = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as pool:
        futures = [pool.submit(send_request, i) for i in range(NUM_REQUESTS)]
        for f in as_completed(futures):
            results.append(f.result())

    total = time.time() - t0
    results.sort(key=lambda x: x[0])

    success = 0
    for idx, status, elapsed, size in results:
        tag = "OK" if status == 200 else "FAIL"
        print(f"  [{tag}] Request #{idx+1}: {status} | {elapsed:.2f}s | {size} bytes" if status else f"  [FAIL] Request #{idx+1}: {size}")
        if status == 200:
            success += 1

    print(f"\n--- Results ---")
    print(f"  Total time:  {total:.2f}s")
    print(f"  Success:     {success}/{NUM_REQUESTS}")
    print(f"  Throughput:  {success/total:.1f} req/s")
