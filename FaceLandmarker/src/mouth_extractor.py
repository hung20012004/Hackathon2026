"""
MouthExtractor
──────────────
Kỹ thuật 1: Cắt vùng miệng bằng bounding-box từ landmarks (đơn giản, nhanh).

Output là ảnh BGR đã resize về output_width × output_height.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from .face_detector import FaceLandmarks
from .landmarks import ALL_LIP_INDICES


class MouthExtractor:
    """
    Trích xuất vùng miệng bằng bounding-box quanh toàn bộ lip landmarks.

    Parameters
    ----------
    output_width, output_height : int
        Kích thước patch đầu ra (pixel).
    bbox_pad : float
        Tỉ lệ padding quanh bounding box (0.25 = thêm 25% mỗi cạnh).
    """

    def __init__(
        self,
        output_width: int = 128,
        output_height: int = 64,
        bbox_pad: float = 0.25,
    ):
        self.out_w = output_width
        self.out_h = output_height
        self.pad = bbox_pad

    # ── public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
    ) -> Optional[np.ndarray]:
        """
        Cắt và resize vùng miệng từ frame.

        Returns
        -------
        np.ndarray shape (out_h, out_w, 3) BGR, hoặc None nếu bbox nằm ngoài ảnh.
        """
        bbox = self._compute_bbox(frame, face)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

    def get_bbox(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Trả về (x1, y1, x2, y2) của bounding box vùng miệng."""
        return self._compute_bbox(frame, face)

    # ── internal ─────────────────────────────────────────────────────────────

    def _compute_bbox(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        pts = face.to_pixels(ALL_LIP_INDICES)  # (N, 2) float32, cols/rows

        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        bw = xmax - xmin
        bh = ymax - ymin

        x1 = int(xmin - bw * self.pad)
        y1 = int(ymin - bh * self.pad)
        x2 = int(xmax + bw * self.pad)
        y2 = int(ymax + bh * self.pad)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
