# ─────────────────────────────────────────────
# Stage 1: deps — build venv + compile protos
# ─────────────────────────────────────────────
FROM python:3.12 AS deps

ENV PYTHONUNBUFFERED=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# PyTorch cu121 — wheels bundle cuDNN, no CUDA base image needed
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.base.txt ./
RUN pip install --no-cache-dir -r requirements.base.txt

# Compile both protos
COPY protos/ ./protos/
COPY run-grpc-compiler.sh ./
RUN mkdir -p compiler_generated \
 && sed -i 's/\r//' run-grpc-compiler.sh \
 && chmod +x run-grpc-compiler.sh \
 && PYTHON=/venv/bin/python ./run-grpc-compiler.sh

# ─────────────────────────────────────────────
# Stage 2: runtime — slim image
# ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=deps /venv /venv
ENV PATH="/venv/bin:$PATH"

# Runtime defaults — every value is overridable via docker-compose environment or -e flags
ENV SERVICE_TYPE=face_detection \
    GRPC_PORT=50053 \
    MODELS_DIR=/app/models \
    EMBEDDING_SERVICE_ADDR=face-embedding:50051 \
    EMBEDDING_USERNAME=verseye \
    EMBEDDING_PASSWORD=verseye1@3 \
    GRPC_USERNAME=verseye \
    GRPC_PASSWORD=verseye1@3

# App source
COPY host.py inference.py ./
COPY interceptors/ ./interceptors/
COPY services/ ./services/

# Generated stubs from deps stage
COPY --from=deps /build/compiler_generated ./compiler_generated/

EXPOSE 50053
CMD ["python", "host.py"]
