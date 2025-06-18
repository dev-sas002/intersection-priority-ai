# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the dependency tree, including the heavy inference stack.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY smart-traffic-backend/requirements.txt smart-traffic-backend/requirements-ml.txt ./

# CPU-only torch: the default PyPI wheels would pull hundreds of megabytes of
# CUDA runtime that a CPU container can never use. The cache mount lives in the
# builder stage only, so it speeds up rebuilds without touching image size.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-ml.txt

# ---------------------------------------------------------------------------
# Stage 2: fetch the public checkpoint so the container needs no network at run
# time. A failure here is not fatal -- the app downloads on first use instead.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS weights

ARG ITCS_PUBLIC_MODEL=yolov8n.pt
ARG ULTRALYTICS_ASSETS=https://github.com/ultralytics/assets/releases/download/v8.3.0

WORKDIR /weights
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL --retry 3 -o "/weights/${ITCS_PUBLIC_MODEL}" \
        "${ULTRALYTICS_ASSETS}/${ITCS_PUBLIC_MODEL}" \
    || echo "WARNING: could not pre-fetch ${ITCS_PUBLIC_MODEL}; it will be downloaded at first use."

# ---------------------------------------------------------------------------
# Stage 3: runtime. No compilers, no pip cache, no build tooling.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ITCS_MODEL_CACHE_DIR=/app/weights \
    ITCS_WEIGHTS_PATH=/app/weights/best.pt \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libglib2.0-0 and libgl1 are what OpenCV links against. ultralytics depends on
# the non-headless opencv-python, so libgl1 has to be here even though this
# project itself asks for opencv-python-headless.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --create-home --uid 10001 itcs

COPY --from=builder /install /usr/local

WORKDIR /app
COPY smart-traffic-backend/app ./app
# Sample clips ship with the image so a fresh container has something to scan.
COPY test_files ./samples
COPY --from=weights /weights/ ./weights/

RUN chown -R itcs:itcs /app && mkdir -p /tmp/ultralytics && chown itcs:itcs /tmp/ultralytics

USER itcs

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
