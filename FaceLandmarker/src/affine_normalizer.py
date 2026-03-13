"""
AffineNormalizer – Affine Transform căn chỉnh vùng miệng
──────────────────────────────────────────────────────────

Output chuẩn: 96×96 px, grayscale.

Chiến lược căn chỉnh (Similarity Transform):
  ┌──────────────────────────────────────────┐
  │  Bước 1  – Lấy 3 điểm landmark miệng    │
  │    • L  = mép trái  (index 61)           │
  │    • R  = mép phải  (index 291)          │
  │    • T  = đỉnh môi  (index 0)            │
  │                                          │
  │  Bước 2  – 3D roll/yaw correction        │
  │    Dùng z-coord của MediaPipe để bổ      │
  │    chính khi đầu nghiêng sang bên.       │
  │                                          │
  │  Bước 3  – Similarity Transform          │
  │    src: [L, R, T] trong ảnh             │
  │    dst: canonical positions trong 96×96  │
  │    estimateAffinePartial2D (no shear)    │
  │                                          │
  │  Bước 4  – warpAffine → BGR → grayscale │
  └──────────────────────────────────────────┘

Canonical layout trong patch 96×96:
    L ──────────── R        row 48  (y = 0.50)
           T                row 19  (y = 0.20)
    col:  18      78        x = 0.1875, 0.8125
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from .face_detector import FaceLandmarks
from .landmarks import MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, MOUTH_TOP_CENTER

# ── Canonical target positions trong 96×96 ────────────────────────────────────
OUTPUT_SIZE = 96   # vuông

_CAN_LEFT  = np.array([18.0, 48.0], dtype=np.float32)   # mép trái
_CAN_RIGHT = np.array([78.0, 48.0], dtype=np.float32)   # mép phải
_CAN_TOP   = np.array([48.0, 19.0], dtype=np.float32)   # đỉnh môi trên


class AffineNormalizer:
    """
    Warp vùng miệng về patch chuẩn 96×96 grayscale.

    Parameters
    ----------
    use_3d : bool
        True  → bổ chính roll và yaw từ z-coordinates (mặc định).
        False → chỉ dùng tọa độ 2D (đủ khi đầu nhìn thẳng).
    """

    def __init__(self, use_3d: bool = True):
        self.use_3d = use_3d
        self._dst = np.stack([_CAN_LEFT, _CAN_RIGHT, _CAN_TOP])   # (3, 2)

    # ── public ────────────────────────────────────────────────────────────────

    def normalize(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
    ) -> Optional[np.ndarray]:
        """
        Trả về mouth patch 96×96 grayscale đã căn chỉnh.
        Trả về None nếu không tính được transform.
        """
        M = self._compute_matrix(frame, face)
        if M is None:
            return None

        warped_bgr = cv2.warpAffine(
            frame,
            M,
            (OUTPUT_SIZE, OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # ── internal ──────────────────────────────────────────────────────────────

    def _compute_matrix(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
    ) -> Optional[np.ndarray]:
        indices = [MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, MOUTH_TOP_CENTER]

        if self.use_3d:
            src = self._src_3d_corrected(frame, face, indices)
        else:
            src = face.to_pixels(indices)   # (3, 2) float32

        if src is None:
            return None

        # Similarity Transform: scale + rotate + translate, NO shear
        M, _ = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2),
            self._dst.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        return M   # None nếu thất bại

    def _src_3d_corrected(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
        indices,
    ) -> Optional[np.ndarray]:
        """
        Lấy 3 điểm nguồn và bổ chính góc nghiêng đầu (roll + yaw).

        Roll  → rotate 2D quanh tâm miệng sao cho đường nối 2 mép nằm ngang.
        Yaw   → mặt quay ngang làm thu ngắn vector ngang; dùng z-delta để
                 scale bù khoảng cách giữa 2 mép.
        """
        pts3 = face.to_pixels_3d(indices)   # (3, 3): [col, row, z_scaled]
        L, R, T = pts3[0], pts3[1], pts3[2]

        # ── Roll correction ───────────────────────────────────────────────────
        roll = np.arctan2(R[1] - L[1], R[0] - L[0])
        cx = (L[0] + R[0]) / 2.0
        cy = (L[1] + R[1]) / 2.0

        cr, sr = np.cos(-roll), np.sin(-roll)
        Rot = np.array([[cr, -sr], [sr, cr]], dtype=np.float32)

        def rot2d(p):
            v = np.array([p[0] - cx, p[1] - cy], dtype=np.float32)
            rv = Rot @ v
            return np.array([rv[0] + cx, rv[1] + cy], dtype=np.float32)

        Lr, Rr, Tr = rot2d(L), rot2d(R), rot2d(T)

        # ── Yaw compensation ──────────────────────────────────────────────────
        # z là metric depth (scale theo width của frame → tương đối)
        h, w = frame.shape[:2]
        z_delta = (R[2] - L[2]) / w           # chuẩn hóa về chiều rộng frame
        yaw = np.arcsin(np.clip(z_delta, -1.0, 1.0))
        cos_yaw = max(np.cos(yaw), 0.15)       # tránh chia gần 0

        dist2d = np.linalg.norm(Rr - Lr)
        if dist2d > 1.0:
            dist_corr = dist2d / cos_yaw
            scale = dist_corr / dist2d
            mid = (Lr + Rr) / 2.0
            Lr = mid + (Lr - mid) * scale
            Rr = mid + (Rr - mid) * scale

        return np.stack([Lr, Rr, Tr]).astype(np.float32)
