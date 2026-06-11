import os
import logging
import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# Template landmarks for 112x112 ArcFace
_ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def umeyama(src, dst):
    """
    Estimate similarity transform parameters (r, t, s) such that dst = s * R * src + t.
    Based on Shinji Umeyama's 1991 paper: Least-squares estimation of transformation parameters between two point patterns.
    """
    num_pts = src.shape[0]
    dim = src.shape[1]
    
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    A = np.dot(dst_demean.T, src_demean) / num_pts
    
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
        
    U, S, V = np.linalg.svd(A)
    
    S_det = np.linalg.det(U) * np.linalg.det(V)
    if S_det < 0:
        d[dim - 1] = -1
        
    R = np.dot(U, np.dot(np.diag(d), V))
    src_var = src_demean.var(axis=0).sum()
    s = 1.0 / src_var * np.dot(S, d)
    t = dst_mean - s * np.dot(R, src_mean)
    
    M = np.zeros((2, 3), dtype=np.double)
    M[0:2, 0:2] = s * R
    M[0:2, 2] = t
    return M

class SCRFDDetector:
    def __init__(self, model_path: str, nms_thresh: float = 0.4, det_thresh: float = 0.5) -> None:
        self.model_path = model_path
        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        
        # Select execution providers (CUDA if available, else CPU)
        providers = ort.get_available_providers()
        selected_providers = []
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")
        
        log.info("Initializing ONNX Runtime session for SCRFD with providers: %s", selected_providers)
        self.session = ort.InferenceSession(self.model_path, providers=selected_providers)
        self.center_cache = {}
        self._init_vars()

    def _init_vars(self) -> None:
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape
        
        # Get output names
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        self.input_mean = 127.5
        self.input_std = 128.0
        
        # Buffalo L det_10g.onnx has 9 outputs:
        # stride 8: score, bbox, kps
        # stride 16: score, bbox, kps
        # stride 32: score, bbox, kps
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        
    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _nms(self, dets: np.ndarray) -> list:
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray, input_size=(640, 640), max_num: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect faces in the image.
        Returns:
            bboxes: array of shape (N, 5) with [x1, y1, x2, y2, score]
            kpss: array of shape (N, 5, 2) with landmark keypoints
        """
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Preprocessing image (blob)
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            (input_size[0], input_size[1]),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].ravel()
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = self._distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        if not scores_list or sum(score.size for score in scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        scores = np.hstack(scores_list)
        order = scores.argsort()[::-1]
        
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        
        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        kpss = kpss[order, :, :]

        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[keep, :, :]
        
        if max_num > 0 and det.shape[0] > max_num:
            det = det[:max_num, :]
            kpss = kpss[:max_num, :, :]
            
        return det, kpss

    def align_face(self, img: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
        """
        Align face crops using Umeyama similarity transform to standard 112x112 coordinates.
        """
        M = umeyama(landmarks, _ARCFACE_DST)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped


class ArcFaceEmbedder:
    def __init__(self, model_path: str) -> None:
        providers = ort.get_available_providers()
        selected: list[str] = []
        if "CUDAExecutionProvider" in providers:
            selected.append("CUDAExecutionProvider")
        selected.append("CPUExecutionProvider")

        log.info("Initializing ONNX Runtime session for ArcFace with providers: %s", selected)
        self.session = ort.InferenceSession(model_path, providers=selected)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, aligned_face: np.ndarray) -> list[float]:
        """
        aligned_face: (112, 112, 3) BGR uint8 image (output of SCRFDDetector.align_face)
        Returns: 512-dim L2-normalized embedding as a float list.
        """
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5          # normalize to [-1, 1]
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW (1, 3, 112, 112)

        outputs = self.session.run(None, {self.input_name: img})
        embedding = outputs[0].flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
