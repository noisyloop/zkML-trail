# zkml-trail — containerized proof + verification server.
#
# A single image runs the FastAPI server; proving happens in isolated
# subprocesses (the EZKL/Halo2 backend) spawned per request, so a stalled
# proof can be timed out and retried without taking down the API.
FROM python:3.11-slim

# EZKL ships as a prebuilt wheel; we only need a C toolchain for any
# source deps and curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Where the artifact store (keys, circuits, SRS) lives. Mount a volume here
# to persist registered agents across restarts.
ENV ZKML_TRAIL_HOME=/data
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["zkml-trail", "serve", "--host", "0.0.0.0", "--port", "8000"]
