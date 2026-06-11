import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import logging
import os
from concurrent import futures

import grpc

from compiler_generated.face_detection_pb2_grpc import add_FaceDetectionServicer_to_server
from inference import FaceDetectionServicerImpl
from services.model_loader import ensure_models
from services.detector import SCRFDDetector, ArcFaceEmbedder
from interceptors.auth import AuthInterceptor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def serve() -> None:
    port = int(os.environ.get("GRPC_PORT", "50051"))
    models_dir = os.environ.get("MODELS_DIR", "models")

    det_model_path, rec_model_path = ensure_models(models_dir)

    log.info("Loading SCRFD detection model: %s", det_model_path)
    detector = SCRFDDetector(model_path=det_model_path)

    log.info("Loading ArcFace recognition model: %s", rec_model_path)
    embedder = ArcFaceEmbedder(model_path=rec_model_path)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[AuthInterceptor()],
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )

    servicer = FaceDetectionServicerImpl(detector=detector, embedder=embedder)
    add_FaceDetectionServicer_to_server(servicer, server)

    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    log.info("Face Detection gRPC server running on :%d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
