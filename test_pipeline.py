"""
End-to-end pipeline test.

Two modes:

  --mode embedding  (default)
      Loads SCRFD locally, detects + aligns faces, sends each 112x112 crop
      directly to the face-embedding gRPC service and returns embeddings.
      Use this when face-detection is down.

  --mode detection
      Sends the raw image to the face-detection gRPC service which internally
      handles detection → alignment → embedding and returns everything.
      Use this when both services are running.

Usage:
    python test_pipeline.py image.jpg --host ai-srv.qbscocloud.net:30050
    python test_pipeline.py image.jpg --host ai-srv.qbscocloud.net:30050 --mode embedding
    python test_pipeline.py image.jpg --host ai-srv.qbscocloud.net:30153 --mode detection
    python test_pipeline.py image1.jpg image2.jpg --host ai-srv.qbscocloud.net:30050
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import cv2
import numpy as np
import grpc

import compiler_generated.face_embedding_pb2 as face_embedding_pb2
import compiler_generated.face_embedding_pb2_grpc as face_embedding_pb2_grpc
import compiler_generated.face_detection_pb2 as face_detection_pb2
import compiler_generated.face_detection_pb2_grpc as face_detection_pb2_grpc

from google.protobuf.empty_pb2 import Empty


# =====================================================
# VISUALIZATION HELPERS
# =====================================================

_KP_COLORS = [
    (255,   0,   0),
    (  0,   0, 255),
    (  0, 255, 255),
    (255,   0, 255),
    (  0, 165, 255),
]
_KP_LABELS = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def print_embedding(emb: np.ndarray) -> None:
    norm = float(np.linalg.norm(emb))
    normalized_ok = abs(norm - 1.0) < 1e-4
    print(f"  Embedding  : {len(emb)}-dim")
    print(f"  L2 norm    : {norm:.6f}  {'✓' if normalized_ok else '✗  NOT NORMALIZED'}")
    print(f"  First 6    : {[round(float(v), 5) for v in emb[:6]]}")


def run_similarity(path_a, path_b, embs_a, embs_b) -> None:
    if not embs_a or not embs_b:
        print("\n  Skipping similarity — one or both images had no embeddings.")
        return
    print(f"\n{'='*60}")
    print(f"  SIMILARITY TEST")
    print(f"  Image A : {path_a}  ({len(embs_a)} face(s))")
    print(f"  Image B : {path_b}  ({len(embs_b)} face(s))")
    print(f"{'='*60}")
    for ia, ea in enumerate(embs_a):
        for ib, eb in enumerate(embs_b):
            sim     = cosine_similarity(ea, eb)
            verdict = "SAME PERSON  (likely)" if sim > 0.40 else "DIFFERENT    (likely)"
            print(f"  A-face#{ia+1} vs B-face#{ib+1} : similarity={sim:.4f}  → {verdict}")


# =====================================================
# MODE 1 — EMBEDDING SERVICE DIRECT
# Local SCRFD detection + alignment, remote embedding call
# =====================================================

def run_embedding_mode(
    image_path: str,
    stub: face_embedding_pb2_grpc.FaceEmbeddingServiceStub,
    metadata: tuple,
    detector,
    save: bool,
) -> list[np.ndarray]:

    print(f"\n{'='*60}")
    print(f"  Image : {image_path}")
    print(f"{'='*60}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR : Cannot read image — {image_path}")
        return []

    h, w = image.shape[:2]
    print(f"  Resolution : {w} x {h}")

    bboxes, kpss = detector.detect(image)
    print(f"  Faces found : {len(bboxes)}")

    if len(bboxes) == 0:
        print("  No faces detected.")
        return []

    embeddings = []

    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        print(f"\n  ── Face #{i+1} ──────────────────────────────────")
        print(f"  BBox       : ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")
        print(f"  Confidence : {score:.4f}")
        print(f"  Landmarks  :")
        for label, kp in zip(_KP_LABELS, kpss[i]):
            print(f"               {label:<10} ({kp[0]:.1f}, {kp[1]:.1f})")

        aligned = detector.align_face(image, kpss[i])
        assert aligned.shape == (112, 112, 3), f"Expected (112,112,3), got {aligned.shape}"
        print(f"  Aligned    : {aligned.shape}  ✓")

        if save:
            crop_out = f"aligned_{Path(image_path).stem}_face{i+1}.jpg"
            cv2.imwrite(crop_out, aligned, [cv2.IMWRITE_JPEG_QUALITY, 99])
            print(f"  Crop saved : {crop_out}")

        success, encoded = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 99])
        if not success:
            print(f"  ERROR : Failed to JPEG-encode face #{i+1}")
            continue

        try:
            req  = face_embedding_pb2.DetectionRequest(image_data=encoded.tobytes())
            resp = stub.Forward(req, metadata=metadata, timeout=10.0)
        except grpc.RpcError as e:
            print(f"  gRPC ERROR : {e.code()} — {e.details()}")
            continue

        emb = np.array(resp.embedding, dtype=np.float32)
        print_embedding(emb)
        embeddings.append(emb)

    if save and len(bboxes) > 0:
        vis = image.copy()
        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            for kp, color in zip(kpss[i], _KP_COLORS):
                cv2.circle(vis, (int(kp[0]), int(kp[1])), 4, color, -1)
        cv2.imwrite(f"detection_{Path(image_path).stem}.jpg", vis)
        print(f"\n  Visualization saved : detection_{Path(image_path).stem}.jpg")

    return embeddings


# =====================================================
# MODE 2 — FACE-DETECTION SERVICE
# Raw image → face-detection service handles everything
# =====================================================

def run_detection_mode(
    image_path: str,
    stub: face_detection_pb2_grpc.FaceDetectionStub,
    metadata: tuple,
    save: bool,
) -> list[np.ndarray]:

    print(f"\n{'='*60}")
    print(f"  Image : {image_path}")
    print(f"{'='*60}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR : Cannot read image — {image_path}")
        return []

    h, w = image.shape[:2]
    print(f"  Resolution : {w} x {h}")

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        print("  ERROR : Failed to encode image as JPEG")
        return []

    request = face_detection_pb2.DetectionRequest(
        frame=encoded.tobytes(),
        height=h, width=w, channels=3,
        timestamp=int(time.time() * 1000),
        camera_id="test-client"
    )

    try:
        response = stub.Forward(request, metadata=metadata, timeout=30.0)
    except grpc.RpcError as e:
        print(f"  gRPC ERROR : {e.code()} — {e.details()}")
        return []

    print(f"  Faces found    : {len(response.objects)}")
    print(f"  Inference time : {response.inference_time_ms} ms")
    print(f"  Message        : {response.message}")

    embeddings = []
    for i, obj in enumerate(response.objects):
        print(f"\n  ── Face #{i+1} ──────────────────────────────────")
        print(f"  BBox       : ({obj.x1:.1f}, {obj.y1:.1f}) → ({obj.x2:.1f}, {obj.y2:.1f})")
        print(f"  Confidence : {obj.confidence:.4f}")
        if obj.landmarks:
            print(f"  Landmarks  :")
            for label, lm in zip(_KP_LABELS, obj.landmarks):
                print(f"               {label:<10} ({lm.x:.1f}, {lm.y:.1f})")
        if obj.embedding:
            emb = np.array(obj.embedding, dtype=np.float32)
            print_embedding(emb)
            embeddings.append(emb)
        else:
            print(f"  Embedding  : N/A (embedding service unreachable)")

    if save and response.objects:
        vis = image.copy()
        for obj in response.objects:
            cv2.rectangle(vis, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), (0, 255, 0), 2)
            for lm, color in zip(obj.landmarks, _KP_COLORS):
                cv2.circle(vis, (int(lm.x), int(lm.y)), 4, color, -1)
        out_path = f"detection_{Path(image_path).stem}.jpg"
        cv2.imwrite(out_path, vis)
        print(f"\n  Visualization saved : {out_path}")

    return embeddings


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline test")
    parser.add_argument("images", nargs="+",
                        help="One or two image paths.")
    parser.add_argument("--host", default="ai-srv.qbscocloud.net:30050",
                        help="Service address (default: ai-srv.qbscocloud.net:30050).")
    parser.add_argument("--mode", choices=["embedding", "detection"], default="embedding",
                        help="embedding: local detect + remote embed (default). "
                             "detection: send raw image to face-detection service.")
    parser.add_argument("--save", action="store_true",
                        help="Save aligned crops and detection visualization.")
    parser.add_argument("--models-dir", default="models",
                        help="Local models directory (embedding mode only).")
    parser.add_argument("--username", default="verseye")
    parser.add_argument("--password", default="verseye1@3")
    args = parser.parse_args()

    channel = grpc.insecure_channel(
        args.host,
        options=[
            ("grpc.max_send_message_length",    32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    metadata = (("username", args.username), ("password", args.password))

    results: dict[str, list[np.ndarray]] = {}

    if args.mode == "embedding":
        print(f"\n========== MODE: EMBEDDING SERVICE DIRECT ==========")
        print(f"  Address : {args.host}")

        from services.model_loader import ensure_models
        from services.detector import SCRFDDetector

        print(f"\n  Loading local detection model...")
        det_path = ensure_models(args.models_dir)
        detector = SCRFDDetector(model_path=det_path)
        print(f"  Detection model loaded ✓")

        stub = face_embedding_pb2_grpc.FaceEmbeddingServiceStub(channel)

        try:
            stub.Forward(
                face_embedding_pb2.DetectionRequest(image_data=b"\xff\xd8\xff"),
                metadata=metadata, timeout=5.0
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"\n  ERROR : Cannot reach embedding service at {args.host}")
                sys.exit(1)
            # Any other code (INVALID_ARGUMENT etc.) means service IS reachable
        print(f"  Embedding service reachable ✓")

        for img_path in args.images[:2]:
            results[img_path] = run_embedding_mode(
                img_path, stub, metadata, detector, args.save
            )

    else:
        print(f"\n========== MODE: FACE-DETECTION SERVICE ==========")
        print(f"  Address : {args.host}")

        stub = face_detection_pb2_grpc.FaceDetectionStub(channel)
        try:
            health = stub.HealthCheck(Empty(), metadata=metadata, timeout=10.0)
            print(f"  Status  : connected ✓")
            print(f"  Health  : {health.status} | model_loaded={health.model_loaded} | v{health.version}")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"  ERROR   : Cannot reach face-detection service at {args.host}")
                sys.exit(1)
            print(f"  Health  : WARNING — {e.code()} {e.details()}")

        for img_path in args.images[:2]:
            results[img_path] = run_detection_mode(img_path, stub, metadata, args.save)

    if len(args.images) >= 2:
        paths = args.images[:2]
        run_similarity(paths[0], paths[1], results[paths[0]], results[paths[1]])

    print(f"\n{'='*60}")
    print("  PIPELINE TEST COMPLETE")
    print(f"{'='*60}\n")
    channel.close()


if __name__ == "__main__":
    main()
