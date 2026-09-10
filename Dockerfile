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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]