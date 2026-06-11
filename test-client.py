#Rule: can be change according to use cases
import time
import grpc
import sys
from pathlib import Path
from google.protobuf.empty_pb2 import Empty

# protoc emits top-level imports; make compiler_generated visible on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import requests
import cv2
import numpy as np

import compiler_generated.face_detection_pb2 as face_detection_pb2
import compiler_generated.face_detection_pb2_grpc as face_detection_pb2_grpc


# =====================================================
# CONFIG
# =====================================================

GRPC_HOST = "localhost:50051"

USERNAME = "verseye"
PASSWORD = "verseye1@3"

IMAGE_URL = "https://images.pexels.com/photos/10528434/pexels-photo-10528434.jpeg?cs=srgb&dl=pexels-qaarif-10528434.jpg"


# =====================================================
# CREATE CHANNEL
# =====================================================

channel = grpc.insecure_channel(
    GRPC_HOST,
    options=[
        ("grpc.max_send_message_length", 32 * 1024 * 1024),
        ("grpc.max_receive_message_length", 32 * 1024 * 1024),
    ],
)

stub = face_detection_pb2_grpc.FaceDetectionStub(channel)

# Authentication metadata
metadata = (
    ("username", USERNAME),
    ("password", PASSWORD),
)


# =====================================================
# DOWNLOAD IMAGE
# =====================================================

def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download image")
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# =====================================================
# HEALTH CHECK
# =====================================================

def health_check():
    response = stub.HealthCheck(Empty(), metadata=metadata)

    print("\n========== HEALTH CHECK ==========")
    print("Status        :", response.status)
    print("Service       :", response.service)
    print("Timestamp     :", response.timestamp)
    print("Model Loaded  :", response.model_loaded)
    print("Version       :", response.version)


# =====================================================
# DETECTION TEST
# =====================================================

def detect():
    print("\nDownloading image...")
    image = download_image(IMAGE_URL)

    if image is None:
        print("Failed to decode image")
        return

    print("Image downloaded successfully")

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        print("Failed to encode image")
        return

    request = face_detection_pb2.DetectionRequest(
        frame=encoded.tobytes(),
        height=image.shape[0],
        width=image.shape[1],
        channels=image.shape[2],
        timestamp=int(time.time() * 1000),
        camera_id="camera-01"
    )

    response = stub.Forward(request, metadata=metadata)

    print("\n========== FACE DETECTION RESPONSE ==========")
    print("Message            :", response.message)
    print("Inference Time(ms) :", response.inference_time_ms)
    print("Model Version      :", response.model_version)
    print("Frame ID           :", response.frame_id)
    print(f"Faces Detected     : {len(response.objects)}")

    print("\nDetected Faces:")
    print("-" * 60)

    for idx, obj in enumerate(response.objects):
        print(f"Face #{idx + 1}")
        print("Class       :", obj.class_name)
        print("Confidence  :", round(obj.confidence, 4))
        print("Track ID    :", obj.track_id)
        print(
            "BBox        : "
            f"({obj.x1:.1f}, {obj.y1:.1f}) -> ({obj.x2:.1f}, {obj.y2:.1f})"
        )
        print(
            "Landmarks   :",
            [(round(lm.x, 1), round(lm.y, 1)) for lm in obj.landmarks]
        )
        if obj.embedding:
            print(f"Embedding   : {len(obj.embedding)}-dim vector (norm={round(float(np.linalg.norm(obj.embedding)), 4)})")
        else:
            print("Embedding   : N/A (embedding service unavailable)")
        print("Attributes  :", dict(obj.attributes))
        print("-" * 60)


# =====================================================
# NAMES
# =====================================================

def get_names():
    response = stub.GetNames(Empty(), metadata=metadata)

    print("\n========== NAMES RESPONSE ==========")
    for id, name in response.names.items():
        print(f"ID: {id}, Name: {name}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    try:
        health_check()
        detect()
        get_names()

    except grpc.RpcError as e:
        print("\nGRPC ERROR")
        print("Code    :", e.code())
        print("Details :", e.details())

    except Exception as e:
        print("\nERROR")
        print(str(e))
