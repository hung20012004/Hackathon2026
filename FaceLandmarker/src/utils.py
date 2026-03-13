"""
Tiện ích vẽ / debug / I/O.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .face_detector import FaceLandmarks
from .landmarks import ALL_LIP_INDICES, AFFINE_SRC_INDICES


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_lip_landmarks(
    frame: np.ndarray,
    face: FaceLandmarks,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1,
) -> np.ndarray:
    """Vẽ toàn bộ lip landmark lên frame (in-place copy)."""
    vis = frame.copy()
    pts = face.to_pixels(ALL_LIP_INDICES).astype(int)
    for x, y in pts:
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis


def draw_affine_keypoints(
    frame: np.ndarray,
    face: FaceLandmarks,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 4,
) -> np.ndarray:
    """Vẽ 3 điểm key dùng cho Affine Transform."""
    vis = frame.copy()
    pts = face.to_pixels(AFFINE_SRC_INDICES).astype(int)
    labels = ["L", "R", "T"]
    for (x, y), label in zip(pts, labels):
        cv2.circle(vis, (x, y), radius, color, -1)
        cv2.putText(vis, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 165, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ bounding box lên frame."""
    vis = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def make_debug_frame(
    original: np.ndarray,
    bbox_crop: Optional[np.ndarray],
    affine_crop: Optional[np.ndarray],
    target_size: Tuple[int, int] = (128, 64),
) -> np.ndarray:
    """
    Ghép [original_with_overlay | bbox_crop | affine_crop] thành một ảnh debug.
    Cả 3 ảnh được resize về cùng chiều cao để dễ so sánh.
    """
    h_target = 200

    def _resize_to_h(img, h):
        if img is None:
            return np.zeros((h, int(h * target_size[0] / target_size[1]), 3), np.uint8)
        ratio = h / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * ratio)), h))

    orig_small = _resize_to_h(original, h_target)
    bbox_small  = _resize_to_h(bbox_crop, h_target)
    affine_small = _resize_to_h(affine_crop, h_target)

    # Add labels
    for img, txt in zip(
        [orig_small, bbox_small, affine_small],
        ["Original", "BBox crop", "Affine crop"],
    ):
        cv2.putText(img, txt, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([orig_small, bbox_small, affine_small])


# ── Video I/O ─────────────────────────────────────────────────────────────────

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def make_video_writer(
    path: str,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
