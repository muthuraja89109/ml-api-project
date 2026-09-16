TESTING.md

Integration Testing

Ran tests/test_integration.py against the live, running Docker container
(via docker compose up --build), using real HTTP requests (httpx) rather
than FastAPI's in-process TestClient. This validates that the whole system
- container, config, model loading, routing - works together, not just the
application code in isolation.

Endpoints tested: /api/v1/health, /api/v1/predict, /api/v1/predict-batch, /metrics

Result: 4 of 4 passed.

Note: the first run failed with httpx.ConnectError [WinError 10061]
because the container was not actually running when the test was executed
(it was run in the same terminal as docker compose up, which blocks the
terminal). Fixed by running the stack and the tests in separate terminals,
confirming reachability with curl first.


Load Testing

Built load_test.py (project root) to send 100 concurrent requests to
/api/v1/predict using asyncio and httpx.AsyncClient, measuring per-request
response time and overall success and failure counts.

Baseline (single Uvicorn worker)

Requests               100
Successful             100
Failed                 0
Avg response time      1.1668s
Min response time      0.3980s
Max response time      1.8404s
Total wall-clock time  1.86s

No requests failed, but average response time under 100 concurrent requests
was about 20 times higher than a single isolated request (around 0.05s),
indicating the API does not scale well under concurrent load.

Diagnosis: ran the same test with 30 concurrent requests to see whether
this was a thread-pool queuing limit. Result avg 0.4722s, still far above
baseline, ruling out thread-pool size as the cause. Prediction is CPU-bound
work (Random Forest inference via predict_proba), and Python's GIL prevents
true parallelism across threads within a single process for CPU-bound work.


Bug Found and Fixed

Bug 1 (real, in production code path): Nothing broke functionally under
load - 0 failures at 100 concurrent requests. The issue found was a
performance bottleneck, not a crash.

Bug 2 (introduced during investigation, caught and fixed): while testing
a fix, app.on_event("startup") was added to app/main.py before app
was defined, causing NameError name 'app' is not defined on every worker
process at import time. The container failed to start entirely, and the
subsequent load test showed 100 of 100 failures with near-instant
ConnectErrors (connection refused). This is a good example of how
integration and load testing catches startup-time crashes that unit tests
never would, since unit tests never actually boot the real application
process.

Fix applied: removed the broken thread-limiter code entirely, since it was
based on a disproven hypothesis. Instead, updated the Dockerfile to run
Uvicorn with --workers 2 instead of the default single worker, so multiple
separate processes, each with their own Python interpreter and GIL, can
handle CPU-bound prediction work in true parallel.

After fix (2 Uvicorn workers)

Requests               100
Successful             100
Failed                 0
Avg response time      1.1023s
Min response time      0.4606s
Max response time      1.3745s
Total wall-clock time  1.39s

Result: about 25 percent reduction in total wall-clock time and max latency
under identical load. A meaningful, measurable improvement, though not a
complete fix - the API is still fundamentally constrained by available CPU
cores under heavy concurrent load, since prediction work is CPU-bound.


Conclusion

The system was verified to work correctly end to end via real HTTP against
the running Docker container, not just unit tests, and correctly handles
100 concurrent requests with zero failures in both configurations. A
genuine performance bottleneck under concurrent load was identified, root
caused as CPU-bound work plus the GIL rather than thread-pool exhaustion,
and partially mitigated via multi-worker deployment. A second, unrelated
startup-crash bug was introduced and caught during this investigation,
demonstrating exactly why integration and load testing exist: they catch
failures unit tests cannot.

Future improvement (beyond this task's scope): further increase worker
count in line with available CPU cores, or investigate batching concurrent
single-prediction requests together internally.