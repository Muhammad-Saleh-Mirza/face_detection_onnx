import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import logging
import time
import cv2
import numpy as np

from compiler_generated.face_detection_pb2 import (
    DetectionObject,
    DetectionResponse,
    HealthCheckResponse,
    NamesResponse,
    Landmark,
)
from compiler_generated.face_detection_pb2_grpc import FaceDetectionServicer

from services.detector import SCRFDDetector, ArcFaceEmbedder

log = logging.getLogger(__name__)


class FaceDetectionServicerImpl(FaceDetectionServicer):
    def __init__(self, detector: SCRFDDetector, embedder: ArcFaceEmbedder) -> None:
        self.detector = detector
        self.embedder = embedder

    def Forward(self, request, context):
        start = time.time()

        np_arr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            log.error("Failed to decode image from request.frame")
            return DetectionResponse(message="invalid image")

        bboxes, kpss = self.detector.detect(image)

        detections = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i]
            kps = kpss[i]

            aligned_face = self.detector.align_face(image, kps)
            embedding = self.embedder.get_embedding(aligned_face)

            pb_landmarks = [Landmark(x=float(kp[0]), y=float(kp[1])) for kp in kps]

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
                    embedding=embedding,
                )
            )

        inference_time = int((time.time() - start) * 1000)

        return DetectionResponse(
            objects=detections,
            message="success",
            inference_time_ms=inference_time,
            model_version="v1",
        )

    def HealthCheck(self, request, context):
        model_loaded = False
        try:
            if self.detector and self.detector.session:
                model_loaded = True
        except Exception:
            pass

        return HealthCheckResponse(
            status="healthy" if model_loaded else "unhealthy",
            service="face_detection",
            timestamp=int(time.time() * 1000),
            model_loaded=model_loaded,
            version="v1",
        )

    def GetNames(self, request, context):
        return NamesResponse(names={1: "face"})
