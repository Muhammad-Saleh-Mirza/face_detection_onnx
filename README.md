# Face Detection & Recognition Plugin

A GPU-accelerated gRPC microservice that detects faces in a frame and returns bounding
boxes, 5-point landmarks, and 512-dim face embeddings in a single call. It's one plugin
in a larger multi-service video-inference pipeline (see the sibling `gait_pipeline` and
`simple_object_detector` protos in `compiler_generated/`) — each service handles one
capability and speaks the same gRPC conventions (`Forward`, `HealthCheck`, `GetNames`).

## Pipeline at a glance

```
 request.frame (jpg/png bytes)
        │
        ▼
 ┌─────────────────┐
 │ 1. DETECT        │  SCRFDDetector.detect()      services/detector.py
 │   (SCRFD)        │  → bboxes (N,5) + 5 landmarks per face
 └─────────────────┘
        │  per face
        ▼
 ┌─────────────────┐
 │ 2. ALIGN         │  SCRFDDetector.align_face()  services/detector.py
 │   (Umeyama warp) │  → 112x112 canonical crop
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │ 3. EMBED         │  ArcFaceEmbedder.get_embedding()  services/detector.py
 │   (ArcFace)      │  → 512-dim, L2-normalized vector
 └─────────────────┘
        │
        ▼
 DetectionResponse { bbox, landmarks, embedding } per face   inference.py
```

Entry point is `FaceDetectionServicerImpl.Forward()` in [inference.py](inference.py) —
it decodes the incoming frame, calls `detector.detect()`, then loops over each detected
face to align + embed it before packing everything into the protobuf response.

## 1. Detection — SCRFD

**Model:** `models/buffalo_l/det_10g.onnx` (InsightFace's `buffalo_l` pack), run through
ONNX Runtime — no PyTorch/TensorFlow dependency at inference time.

SCRFD is an anchor-based detector that predicts faces at three feature-map resolutions
(strides `8`, `16`, `32` — see `_feat_stride_fpn` in `SCRFDDetector._init_vars`), so it
catches both large, close-up faces and small, far-away ones in the same pass. For each
stride, the model outputs three tensors: a confidence score, a bounding-box offset, and
5 facial-landmark offsets per anchor point.

Walking through `SCRFDDetector.detect()`:

1. **Letterbox resize** — the image is resized to fit inside 640x640 while preserving
   aspect ratio, then padded into a black canvas (`det_img`). `det_scale` records the
   resize factor so box/landmark coordinates can be mapped back to the original image
   size at the end.
2. **Blob prep** — `cv2.dnn.blobFromImage` normalizes pixels to roughly `[-1, 1]`
   (`mean=127.5, std=128.0`), swaps BGR→RGB, and reshapes to `NCHW` for the ONNX graph.
3. **Per-stride decode** — for each of the 3 strides: build (or fetch from
   `center_cache`) a grid of anchor center points, then turn the model's raw
   distance-from-anchor predictions into actual pixel coordinates via
   `_distance2bbox` / `_distance2kps`. Anchors below `det_thresh` (default `0.5`) are
   dropped immediately.
4. **NMS** — all surviving boxes across the three strides are merged, sorted by score,
   and passed through a custom greedy `_nms()` (IoU threshold `nms_thresh`, default
   `0.4`) to collapse duplicate/overlapping detections into one box per face.
5. Boxes and landmarks are divided by `det_scale` to map back to original image
   coordinates and returned as `(N, 5)` boxes `[x1, y1, x2, y2, score]` and `(N, 5, 2)`
   landmarks.

**Why an anchor cache?** `center_cache` memoizes anchor-grid coordinates per
`(height, width, stride)` key — the grid is identical across all requests as long as the
image is resized to the same `input_size`, so it's computed once and reused instead of
regenerated every request.

## 2. Alignment — Umeyama transform

Recognition models like ArcFace are trained on faces that are consistently cropped and
rotated into a fixed pose — feeding them an arbitrary crop hurts embedding quality. So
before embedding, `SCRFDDetector.align_face()` warps each detected face into that fixed
pose using the 5 landmarks SCRFD already gave us:

- `_ARCFACE_DST` is the canonical target position of each landmark (left eye, right eye,
  nose, left mouth corner, right mouth corner) inside a 112x112 frame — this is a fixed
  reference used by every ArcFace-compatible pipeline, not something we compute.
- `umeyama()` solves for the similarity transform (rotation + uniform scale +
  translation) that best maps the detected landmarks onto that canonical layout
  (least-squares, via SVD — Umeyama's 1991 method).
- `cv2.warpAffine` applies that transform to the original image, producing a
  112x112 crop where the eyes/nose/mouth sit in the same place for every face,
  regardless of the face's original size, rotation, or position in frame.

## 3. Embedding — ArcFace

**Model:** `models/buffalo_l/w600k_r50.onnx` — a ResNet-50 trained with the ArcFace
loss, which explicitly optimizes for angular separation between different identities'
embeddings (as opposed to a generic classification loss). That's what makes cosine
similarity between two embeddings a meaningful "same person / different person" signal.

`ArcFaceEmbedder.get_embedding()`:

1. Converts the aligned 112x112 BGR crop to RGB, normalizes to `[-1, 1]` the same way
   as detection preprocessing.
2. Reshapes `HWC → NCHW` and runs it through the ONNX session.
3. **L2-normalizes** the raw 512-dim output vector before returning it. This is what
   makes downstream comparisons simple: once every embedding has unit length, cosine
   similarity between two embeddings reduces to a plain dot product (see
   `cosine_similarity()` in [test_pipeline.py](test_pipeline.py)).

The resulting `list[float]` goes straight onto `DetectionObject.embedding` in the
response — this service does **not** store embeddings or do similarity search itself;
that's a downstream concern (e.g. a vector index in another service).

## gRPC API

Defined in [protos/face_detection.proto](protos/face_detection.proto), implemented in
[inference.py](inference.py):

| RPC | Purpose |
|---|---|
| `Forward(DetectionRequest) → DetectionResponse` | The main call — frame in, detections (bbox + landmarks + embedding) out. |
| `HealthCheck(Empty) → HealthCheckResponse` | Reports whether the ONNX sessions are loaded. |
| `GetNames(Empty) → NamesResponse` | Returns the class-id → class-name map (`{1: "face"}`). |

Every call must carry `username`/`password` gRPC metadata — enforced by
[interceptors/auth.py](interceptors/auth.py)'s `AuthInterceptor`, which aborts
unauthenticated calls with `UNAUTHENTICATED` before they reach the servicer.

`PreviousDetection` in the request lets an upstream pipeline stage (e.g. a person
tracker running SORT/ByteTrack/DeepSORT) hand this service prior-stage results, since
it's designed to slot into a chain of pipeline plugins rather than run standalone.

## Startup & model loading

[host.py](host.py) is the process entry point:

1. `services/model_loader.ensure_models()` checks for the two ONNX files under
   `MODELS_DIR`; if missing, it downloads and extracts them from S3
   (`services/model_loader.py`) — so the container is self-bootstrapping and doesn't
   need model weights baked into the image.
2. Both ONNX Runtime sessions are created with `CUDAExecutionProvider` preferred and
   `CPUExecutionProvider` as fallback — chosen automatically from
   `ort.get_available_providers()`, no manual GPU/CPU flag needed.
3. A `grpc.server` is started with the `AuthInterceptor`, a 4-worker thread pool, and
   32 MB max message size (frames/embeddings can exceed gRPC's 4 MB default).

## Docker build

Multi-stage build on an `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base (GPU
inference needs the CUDA/cuDNN runtime libs, so both stages share it — this isn't a
"switch to a slim base" pattern):

- **`deps` stage** — builds a venv, installs `onnxruntime-gpu` in its own cached layer,
  installs the rest of `requirements.base.txt`, and compiles `protos/*.proto` into
  `compiler_generated/` via `run-grpc-compiler.sh`.
- **`runtime` stage** — copies only the built venv and the generated `.py` stubs across;
  proto source files, the compiler script, and the `python3.12-venv` apt package never
  reach the image that ships. App source is copied last so editing `inference.py`
  doesn't invalidate the (slow) dependency-install layers.

## Repo layout

```
host.py                  gRPC server bootstrap (entry point)
inference.py              FaceDetectionServicerImpl — the Forward/HealthCheck/GetNames RPCs
services/
  detector.py              SCRFDDetector (detect + align) and ArcFaceEmbedder (embed)
  model_loader.py           Downloads/extracts ONNX weights from S3 if missing
interceptors/
  auth.py                   Username/password gRPC server interceptor
protos/
  face_detection.proto      Service + message definitions (source of truth for the API)
compiler_generated/         protoc/grpc_tools output — generated, not hand-edited
models/buffalo_l/           ONNX weights (det_10g.onnx, w600k_r50.onnx, ...)
test-client.py, test_pipeline.py   Manual/E2E test scripts against a running server
Dockerfile, docker-compose.yml     Container build + local run config
```
