# ─────────────────────────────────────────────
# Stage 1: deps — build venv + compile protos
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS deps

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# onnxruntime-gpu — cached in its own layer, only re-runs if this line changes
# --only-binary :all: ensures pip uses a pre-built wheel and never compiles from source
RUN pip install --no-cache-dir --only-binary :all: onnxruntime-gpu==1.19.2

# Remaining deps — fast to install, cache invalidates when requirements.base.txt changes
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
# Stage 2: runtime — CUDA/cuDNN base for GPU inference
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# python3.12 only — CUDA/cuDNN runtime libs already come from the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=deps /venv /venv
ENV PATH="/venv/bin:$PATH"

# Runtime defaults — every value is overridable via docker-compose environment or -e flags
ENV SERVICE_TYPE=face_detection \
    GRPC_PORT=50051 \
    MODELS_DIR=/app/models \
    GRPC_USERNAME=verseye \
    GRPC_PASSWORD=verseye1@3

# App source
COPY host.py inference.py ./
COPY interceptors/ ./interceptors/
COPY services/ ./services/

# Generated stubs from deps stage
COPY --from=deps /build/compiler_generated ./compiler_generated/

EXPOSE 50051
CMD ["python", "host.py"]
