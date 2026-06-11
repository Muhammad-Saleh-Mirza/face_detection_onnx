"""
End-to-end pipeline test.

Sends a raw image to the face-detection gRPC service.
The service handles detection → alignment → embedding internally and returns all results.

Usage:
    python test_pipeline.py image.jpg --host localhost:50053
    python test_pipeline.py image.jpg --host ai-srv.qbscocloud.net:30154
    python test_pipeline.py image1.jpg image2.jpg --host localhost:50053
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

import compiler_generated.face_detection_pb2 as face_detection_pb2
import compiler_generated.face_detection_pb2_grpc as face_detection_pb2_grpc

from google.protobuf.empty_pb2 import Empty


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
    print(f"  L2 norm    : {norm:.6f}  {'OK' if normalized_ok else 'NOT NORMALIZED'}")
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
            print(f"  A-face#{ia+1} vs B-face#{ib+1} : similarity={sim:.4f}  -> {verdict}")


def run_detection(
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
        camera_id="test-client",
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
        print(f"\n  -- Face #{i+1} --------------------------------------------")
        print(f"  BBox       : ({obj.x1:.1f}, {obj.y1:.1f}) -> ({obj.x2:.1f}, {obj.y2:.1f})")
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
            print(f"  Embedding  : N/A")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection + embedding pipeline test")
    parser.add_argument("images", nargs="+", help="One or two image paths.")
    parser.add_argument("--host", default="localhost:50051",
                        help="face-detection service address (default: localhost:50051).")
    parser.add_argument("--save", action="store_true",
                        help="Save detection visualization to disk.")
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
    stub = face_detection_pb2_grpc.FaceDetectionStub(channel)

    print(f"\n========== FACE DETECTION PIPELINE TEST ==========")
    print(f"  Address : {args.host}")

    try:
        health = stub.HealthCheck(Empty(), metadata=metadata, timeout=10.0)
        print(f"  Status  : connected OK")
        print(f"  Health  : {health.status} | model_loaded={health.model_loaded} | v{health.version}")
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            print(f"  ERROR   : Cannot reach face-detection service at {args.host}")
            sys.exit(1)
        print(f"  Health  : WARNING — {e.code()} {e.details()}")

    results: dict[str, list[np.ndarray]] = {}
    for img_path in args.images[:2]:
        results[img_path] = run_detection(img_path, stub, metadata, args.save)

    if len(args.images) >= 2:
        paths = args.images[:2]
        run_similarity(paths[0], paths[1], results[paths[0]], results[paths[1]])

    print(f"\n{'='*60}")
    print("  PIPELINE TEST COMPLETE")
    print(f"{'='*60}\n")
    channel.close()


if __name__ == "__main__":
    main()
