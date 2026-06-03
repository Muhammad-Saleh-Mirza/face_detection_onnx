"""
End-to-end pipeline test.

Runs face detection + alignment locally, then sends each aligned
112x112 crop to the face-embedding gRPC service and gets the
512-d embedding back.

Requires:
    - face-embedding service running on localhost:50051

Usage:
    python test_pipeline.py image.jpg
    python test_pipeline.py image.jpg --save
    python test_pipeline.py person1.jpg person2.jpg        # cosine similarity test
    python test_pipeline.py image.jpg --host 192.168.1.10:50051
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import cv2
import numpy as np
import grpc

import compiler_generated.face_embedding_pb2 as face_embedding_pb2
import compiler_generated.face_embedding_pb2_grpc as face_embedding_pb2_grpc

from services.model_loader import ensure_models
from services.detector import SCRFDDetector


# =====================================================
# VISUALIZATION HELPERS
# =====================================================

# Landmark colours: L-eye, R-eye, Nose, L-mouth, R-mouth
_KP_COLORS = [
    (255,   0,   0),   # blue    — left eye
    (  0,   0, 255),   # red     — right eye
    (  0, 255, 255),   # yellow  — nose tip
    (255,   0, 255),   # magenta — left mouth
    (  0, 165, 255),   # orange  — right mouth
]
_KP_LABELS = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]


def draw_detections(image: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray) -> np.ndarray:
    vis = image.copy()
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            vis, f"face {score:.2f}",
            (int(x1), max(int(y1) - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        for kp, color in zip(kpss[i], _KP_COLORS):
            cv2.circle(vis, (int(kp[0]), int(kp[1])), 4, color, -1)
    return vis


# =====================================================
# PIPELINE
# =====================================================

def run_pipeline(
    image_path: str,
    detector: SCRFDDetector,
    stub: face_embedding_pb2_grpc.FaceEmbeddingServiceStub,
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

    # ── Step 1: Detect faces locally ─────────────────────────
    bboxes, kpss = detector.detect(image)
    print(f"  Faces found : {len(bboxes)}")

    if len(bboxes) == 0:
        print("  No faces detected.")
        return []

    embeddings = []

    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        print(f"\n  ── Face #{i + 1} ──────────────────────────────────")
        print(f"  BBox       : ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")
        print(f"  Confidence : {score:.4f}")
        print(f"  Landmarks  :")
        for label, kp in zip(_KP_LABELS, kpss[i]):
            print(f"               {label:<10} ({kp[0]:.1f}, {kp[1]:.1f})")

        # ── Step 2: Align to 112x112 locally ─────────────────
        aligned = detector.align_face(image, kpss[i])
        assert aligned.shape == (112, 112, 3), \
            f"Expected (112,112,3), got {aligned.shape}"
        print(f"  Aligned    : {aligned.shape}  ✓")

        if save:
            stem     = Path(image_path).stem
            crop_out = f"aligned_{stem}_face{i + 1}.jpg"
            cv2.imwrite(crop_out, aligned, [cv2.IMWRITE_JPEG_QUALITY, 99])
            print(f"  Crop saved : {crop_out}")

        # ── Step 3: Encode and send to embedding service ─────
        success, encoded = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 99])
        if not success:
            print(f"  ERROR : Failed to JPEG-encode aligned crop for face #{i + 1}")
            continue

        try:
            request  = face_embedding_pb2.DetectionRequest(image_data=encoded.tobytes())
            response = stub.Forward(request, metadata=metadata, timeout=10.0)
        except grpc.RpcError as e:
            print(f"  gRPC ERROR : {e.code()} — {e.details()}")
            continue

        emb  = np.array(response.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        normalized_ok = abs(norm - 1.0) < 1e-4

        print(f"  Embedding  : {len(emb)}-dim")
        print(f"  L2 norm    : {norm:.6f}  {'✓' if normalized_ok else '✗  NOT NORMALIZED'}")
        print(f"  First 6    : {[round(float(v), 5) for v in emb[:6]]}")

        embeddings.append(emb)

    # ── Visualization ─────────────────────────────────────────
    if save:
        vis     = draw_detections(image, bboxes, kpss)
        vis_out = f"detection_{Path(image_path).stem}.jpg"
        cv2.imwrite(vis_out, vis)
        print(f"\n  Detection visualization saved : {vis_out}")

    return embeddings


# =====================================================
# SIMILARITY TEST
# =====================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Both vectors are already L2-normalized so dot == cosine similarity
    return float(np.dot(a, b))


def run_similarity(
    path_a: str,
    path_b: str,
    embs_a: list[np.ndarray],
    embs_b: list[np.ndarray],
) -> None:
    if not embs_a or not embs_b:
        print("\n  Skipping similarity — one or both images had no faces.")
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
            print(f"  A-face#{ia + 1} vs B-face#{ib + 1} : similarity = {sim:.4f}  → {verdict}")


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline test: local detection + remote embedding via gRPC"
    )
    parser.add_argument(
        "images", nargs="+",
        help="One or two image paths. Two images enables cosine similarity output."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save aligned 112x112 crops and detection visualization to disk."
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Root directory for model weights (default: models)."
    )
    parser.add_argument(
        "--host", default="localhost:50051",
        help="Face-embedding gRPC service address (default: localhost:50051)."
    )
    parser.add_argument(
        "--username", default="verseye",
        help="gRPC auth username (default: verseye)."
    )
    parser.add_argument(
        "--password", default="verseye1@3",
        help="gRPC auth password (default: verseye1@3)."
    )
    args = parser.parse_args()

    # ── Load detection model ──────────────────────────────────
    print("\n========== LOADING DETECTION MODEL ==========")
    det_path = ensure_models(args.models_dir)
    print(f"  Path   : {det_path}")
    detector = SCRFDDetector(model_path=det_path)
    print(f"  Status : loaded ✓")

    # ── Connect to embedding service ─────────────────────────
    print(f"\n========== CONNECTING TO EMBEDDING SERVICE ==========")
    print(f"  Address : {args.host}")

    channel = grpc.insecure_channel(
        args.host,
        options=[
            ("grpc.max_send_message_length",    32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )

    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        print(f"  Status  : connected ✓")
    except grpc.FutureTimeoutError:
        print(f"  ERROR   : Could not reach embedding service at {args.host}")
        print(f"            Make sure the face-embedding service is running.")
        sys.exit(1)

    stub     = face_embedding_pb2_grpc.FaceEmbeddingServiceStub(channel)
    metadata = (
        ("username", args.username),
        ("password", args.password),
    )

    # ── Run pipeline ──────────────────────────────────────────
    results: dict[str, list[np.ndarray]] = {}
    for img_path in args.images[:2]:
        results[img_path] = run_pipeline(img_path, detector, stub, metadata, args.save)

    # ── Similarity if two images given ────────────────────────
    if len(args.images) >= 2:
        paths = args.images[:2]
        run_similarity(paths[0], paths[1], results[paths[0]], results[paths[1]])

    print(f"\n{'='*60}")
    print("  PIPELINE TEST COMPLETE")
    print(f"{'='*60}\n")

    channel.close()


if __name__ == "__main__":
    main()
