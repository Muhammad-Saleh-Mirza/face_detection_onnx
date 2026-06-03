import sys
from pathlib import Path

# Ensure compiler_generated is visible
sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import logging
import time
import os
import cv2
import numpy as np
import grpc

from compiler_generated.simple_object_detector_pb2 import (
    DetectionObject,
    DetectionResponse,
    HealthCheckResponse as ObjHealthCheckResponse,
    NamesResponse as ObjNamesResponse
)
from compiler_generated.simple_object_detector_pb2_grpc import ObjectDetectionServicer

from compiler_generated.face_detection_pb2 import (
    DetectionObject,
    DetectionResponse,
    HealthCheckResponse,
    NamesResponse,
    Landmark
)
from compiler_generated.face_detection_pb2_grpc import FaceDetectionServicer
import compiler_generated.face_embedding_pb2 as face_embedding_pb2
import compiler_generated.face_embedding_pb2_grpc as face_embedding_pb2_grpc

from services.detector import SCRFDDetector

log = logging.getLogger(__name__)


class NewImplObjectDetectionServicer(
    ObjectDetectionServicer
):

    def Forward(self, request, context):

        start = time.time()

        # Decode image
        np_arr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return DetectionResponse(
                message="invalid image"
            )

        h, w, _ = image.shape

        # Dummy lightweight detection
        detections = [
            DetectionObject(
                x1=w * 0.25,
                y1=h * 0.25,
                x2=w * 0.75,
                y2=h * 0.75,
                class_name="mock_object",
                confidence=0.95,
                track_id=1,
                attributes={
                    "source": "mock_detector"
                }
            )
        ]

        inference_time = int((time.time() - start) * 1000)

        return DetectionResponse(
            objects=detections,
            message="success",
            inference_time_ms=inference_time,
            model_version="v1",
            frame_id=request.frame_id
        )

    def HealthCheck(self, request, context):

        return ObjHealthCheckResponse(
            status="healthy",
            service="simple_object_detector",
            timestamp=int(time.time() * 1000),
            model_loaded=True,
            version="v1"
        )
    def GetNames(self, request, context):
        return ObjNamesResponse(
            names={1: "mock_object"}
        )


class FaceDetectionServicerImpl(FaceDetectionServicer):
    def __init__(self, detector: SCRFDDetector, embedding_service_addr: str) -> None:
        self.detector = detector
        self.embedding_service_addr = embedding_service_addr
        
        log.info("Connecting to remote embedding service at %s", self.embedding_service_addr)
        self.channel = grpc.insecure_channel(
            self.embedding_service_addr,
            options=[
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            ]
        )
        self.embedding_stub = face_embedding_pb2_grpc.FaceEmbeddingServiceStub(self.channel)

    def _get_embedding_with_retry(self, image_data: bytes, retries: int = 3, backoff: float = 0.5) -> list[float]:
        req = face_embedding_pb2.DetectionRequest(image_data=image_data)
        metadata = (
            ("username", os.environ.get("EMBEDDING_USERNAME", "verseye")),
            ("password", os.environ.get("EMBEDDING_PASSWORD", "verseye1@3")),
        )

        for attempt in range(retries):
            try:
                resp = self.embedding_stub.Forward(req, metadata=metadata, timeout=5.0)
                embedding_arr = np.array(resp.embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_arr)
                if norm > 0:
                    embedding_arr = embedding_arr / norm
                return embedding_arr.tolist()
            except grpc.RpcError as e:
                log.warning("gRPC error when calling embedding service (attempt %d/%d): %s - %s", 
                            attempt + 1, retries, e.code(), e.details())
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
            except Exception as e:
                log.warning("Unexpected error when calling embedding service (attempt %d/%d): %s", 
                            attempt + 1, retries, str(e))
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    
        log.error("Failed to retrieve embedding after %d attempts", retries)
        return []

    def Forward(self, request, context):
        start = time.time()
        
        # Decode image
        np_arr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            log.error("Failed to decode image from request.frame")
            return DetectionResponse(
                message="invalid image"
            )
            
        # Detect faces
        bboxes, kpss = self.detector.detect(image)
        
        detections = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            kps = kpss[i]
            
            x1, y1, x2, y2, score = bbox
            
            # Align face crop using the 5 landmark coordinates
            aligned_face = self.detector.align_face(image, kps)
            
            # Encode face crop to JPEG
            success, encoded_crop = cv2.imencode(".jpg", aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 99])
            if not success:
                log.error("Failed to encode aligned face crop to JPEG")
                crop_bytes = b""
            else:
                crop_bytes = encoded_crop.tobytes()
                
            # Retrieve embedding from remote gRPC service
            embedding = []
            if len(crop_bytes) > 0:
                embedding = self._get_embedding_with_retry(crop_bytes)
                
            # Construct protobuf Landmark list
            pb_landmarks = []
            for kp in kps:
                pb_landmarks.append(Landmark(x=float(kp[0]), y=float(kp[1])))

            detections.append(
                DetectionObject(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_name="face",
                    confidence=float(score),
                    track_id=0,
                    attributes={"source": "scrfd"},
                    landmarks=pb_landmarks,
                    embedding=embedding
                )
            )

        inference_time = int((time.time() - start) * 1000)

        return DetectionResponse(
            objects=detections,
            message="success",
            inference_time_ms=inference_time,
            model_version="v1"
        )

    def HealthCheck(self, request, context):
        # Validate model availability during runtime
        model_loaded = False
        try:
            # Check if session is initialized and running
            if self.detector and self.detector.session:
                model_loaded = True
        except Exception:
            pass

        return HealthCheckResponse(
            status="healthy" if model_loaded else "unhealthy",
            service="face_detection",
            timestamp=int(time.time() * 1000),
            model_loaded=model_loaded,
            version="v1"
        )

    def GetNames(self, request, context):
        return NamesResponse(
            names={1: "face"}
        )
