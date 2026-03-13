"""
FaceLandmarker (Tasks API)
──────────────────────────
Wrapper quanh MediaPipe Tasks API FaceLandmarker – đúng như notebook dùng.
Hỗ trợ cả IMAGE mode (ảnh tĩnh) và VIDEO mode (xử lý frame-by-frame).

Model cần download trước:
    wget -O face_landmarker_v2_with_blendshapes.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass, field
from typing import List, Optional
import cv2


@dataclass
class FaceLandmarks:
    """
    Landmarks của một khuôn mặt.

    Attributes
    ----------
    xy  : (N, 2) float32  –  tọa độ chuẩn hóa [0,1] trong ảnh (x = col, y = row)
    xyz : (N, 3) float32  –  idem + z metric (âm = gần camera)
    frame_shape : (H, W, C)  –  kích thước frame gốc
    transform_matrix : (4, 4) float32 hoặc None
        Ma trận facial transformation từ MediaPipe (face model → camera space).
        Dùng để bổ chính pose 3D nếu cần.
    """

    xy: np.ndarray
    xyz: np.ndarray
    frame_shape: tuple = field(default=(0, 0, 3))
    transform_matrix: Optional[np.ndarray] = None

    def to_pixels(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Trả về tọa độ pixel (col, row) float32 cho các landmark được chỉ định."""
        h, w = self.frame_shape[:2]
        pts = self.xy if indices is None else self.xy[indices]
        return (pts * np.array([w, h], dtype=np.float32)).astype(np.float32)

    def to_pixels_3d(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Trả về (col, row, z_scaled) float32."""
        h, w = self.frame_shape[:2]
        pts = self.xyz if indices is None else self.xyz[indices]
        return (pts * np.array([w, h, w], dtype=np.float32)).astype(np.float32)


class FaceLandmarker:
    """
    Phát hiện facial landmarks bằng MediaPipe Tasks API.

    Parameters
    ----------
    model_path : str
        Đường dẫn tới file .task (face_landmarker_v2_with_blendshapes.task).
    num_faces : int
    mode : str
        'image'  – dùng detect()         (ảnh độc lập)
        'video'  – dùng detect_video()   (cần timestamp_ms tăng dần)
    """

    def __init__(
        self,
        model_path: str = "face_landmarker_v2_with_blendshapes.task",
        num_faces: int = 1,
        mode: str = "video",
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)

        running_mode = (
            vision.RunningMode.VIDEO
            if mode == "video"
            else vision.RunningMode.IMAGE
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,   # dùng cho 3D correction
            num_faces=num_faces,
            running_mode=running_mode,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mode = mode

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[FaceLandmarks]:
        """Detect trên ảnh tĩnh (IMAGE mode)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        return self._parse_result(result, frame_bgr.shape)

    def detect_video(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> List[FaceLandmarks]:
        """Detect trên frame video (VIDEO mode). timestamp_ms phải tăng dần."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._parse_result(result, frame_bgr.shape)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(result, frame_shape: tuple) -> List[FaceLandmarks]:
        if not result.face_landmarks:
            return []

        faces = []
        for i, landmarks in enumerate(result.face_landmarks):
            n = len(landmarks)
            xy  = np.empty((n, 2), dtype=np.float32)
            xyz = np.empty((n, 3), dtype=np.float32)
            for j, lm in enumerate(landmarks):
                xy[j]  = (lm.x, lm.y)
                xyz[j] = (lm.x, lm.y, lm.z)

            mat = None
            if (
                result.facial_transformation_matrixes
                and i < len(result.facial_transformation_matrixes)
            ):
                mat = np.array(
                    result.facial_transformation_matrixes[i],
                    dtype=np.float32,
                ).reshape(4, 4)

            faces.append(
                FaceLandmarks(
                    xy=xy,
                    xyz=xyz,
                    frame_shape=frame_shape,
                    transform_matrix=mat,
                )
            )
        return faces
