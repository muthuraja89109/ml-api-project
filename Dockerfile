FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 makes Uvicorn listen on all network interfaces inside the
# container, so Docker can forward the host's port 8000 to it.
# 127.0.0.1 would only accept connections from inside the container
# itself, so requests from the host machine (or your browser) would
# never reach it.
#
# --workers 2 runs two separate Uvicorn worker processes instead of
# one. Prediction is CPU-bound (tree traversal in the Random Forest),
# and Python's GIL means CPU-bound work doesn't truly parallelize
# across threads within a single process. Multiple worker processes
# each get their own interpreter (and GIL), letting real work happen
# in parallel — this was found to meaningfully reduce average
# response time under concurrent load (see TESTING.md).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]