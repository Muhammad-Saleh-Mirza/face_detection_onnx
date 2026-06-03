import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import asyncio
import logging
import os
import time
import urllib.request
from concurrent import futures

import grpc
import numpy as np


# Face detection imports
from compiler_generated.face_detection_pb2_grpc import (
    add_FaceDetectionServicer_to_server,
)
from inference import FaceDetectionServicerImpl
from services.model_loader import ensure_models
from services.detector import SCRFDDetector
from interceptors.auth import AuthInterceptor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _download_checkpoint(url: str, dest: str) -> None:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    last_pct = [-1]

    def _progress(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(count * block_size * 100 / total_size)
        pct = min(pct, 100)
        if pct - last_pct[0] >= 5:
            last_pct[0] = pct
            filled  = pct // 5
            empty   = 20 - filled
            bar     = "█" * filled + "░" * empty
            done_mb = count * block_size / 1024 / 1024
            tot_mb  = total_size / 1024 / 1024
            log.info("[%s] %d%%  %.1f / %.1f MB", bar, pct, done_mb, tot_mb)

    log.info("Downloading checkpoint from %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    log.info("Download complete: %s", dest)


def _resolve_checkpoint(ckpt_path: str, ckpt_url: str) -> str:
    path = Path(ckpt_path)
    if path.exists():
        return str(path)

    # Confirm the parent directory is writable; fall back to /app/models/ if not.
    try:
        probe = path.parent / ".write_probe"
        probe.touch()
        probe.unlink()
        target = str(path)
    except OSError:
        fallback = Path("/app/models") / path.name
        log.warning("Cannot write to %s — falling back to %s", path.parent, fallback.parent)
        if fallback.exists():
            return str(fallback)
        target = str(fallback)

    _download_checkpoint(ckpt_url, target)
    return target



def serve_face_detection() -> None:
    port = int(os.environ.get("GRPC_PORT", "50053"))
    models_dir = os.environ.get("MODELS_DIR", "models")
    embedding_service_addr = os.environ.get("EMBEDDING_SERVICE_ADDR", "localhost:50051")
    
    # 1. Model safety check & download if missing
    resolved_model_path = ensure_models(models_dir)
    log.info("Loading SCRFD face detection model: %s", resolved_model_path)
    
    # 2. Instantiate detector (validates model availability during startup)
    detector = SCRFDDetector(model_path=resolved_model_path)
    
    # 3. Create gRPC server with AuthInterceptor registered
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=(AuthInterceptor(),),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    
    # 4. Register face detection servicer
    servicer = FaceDetectionServicerImpl(
        detector=detector,
        embedding_service_addr=embedding_service_addr
    )
    add_FaceDetectionServicer_to_server(servicer, server)
    
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    log.info("Face Detection gRPC server running on :%d", port)
    server.wait_for_termination()


def serve() -> None:
    serve_face_detection()


if __name__ == "__main__":
    serve()
