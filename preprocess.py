"""
preprocess.py
Input : video MP4 (face crop 160x160 hoặc bất kỳ)
Output: video MP4 grayscale 96x96 mouth crop (affine stabilized)

Usage:
    python preprocess.py --input input.mp4 --output mouth.mp4
    python preprocess.py --input input.mp4 --output mouth.mp4 --task face_landmarker.task
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ── Landmark indices (MediaPipe 478-point mesh) ──────────────────────────────
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
MOUTH_LEFT      = 61
MOUTH_RIGHT     = 291
MOUTH_TOP       = 13
MOUTH_BOTTOM    = 14

# ── Output spec ───────────────────────────────────────────────────────────────
OUT_SIZE   = 96          # final output resolution
MOUTH_W_RATIO = 0.60     # mouth width as fraction of output width


def get_landmarks(landmarker, frame_rgb):
    """Run FaceLandmarker on a single RGB frame. Returns (478,2) array or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    h, w = frame_rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts


def compute_affine(pts, out_size=OUT_SIZE):
    """
    Compute affine transform so that:
    - Eyes are horizontal
    - Mouth center is at output center
    - Mouth width fills MOUTH_W_RATIO of output width
    Returns 2x3 affine matrix M.
    """
    left_eye  = (pts[LEFT_EYE_INNER]  + pts[LEFT_EYE_OUTER])  / 2
    right_eye = (pts[RIGHT_EYE_INNER] + pts[RIGHT_EYE_OUTER]) / 2
    mouth_l   = pts[MOUTH_LEFT]
    mouth_r   = pts[MOUTH_RIGHT]

    # rotation angle from eye line
    eye_vec = right_eye - left_eye
    angle   = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

    # scale so mouth width = MOUTH_W_RATIO * out_size
    mouth_width = np.linalg.norm(mouth_r - mouth_l)
    if mouth_width < 1:
        return None
    scale = (MOUTH_W_RATIO * out_size) / mouth_width

    # mouth center → output center
    mouth_center = (mouth_l + mouth_r) / 2
    cx, cy = out_size / 2, out_size / 2

    # build affine: rotate around mouth_center, then scale, then translate
    cos_a, sin_a = np.cos(np.radians(-angle)), np.sin(np.radians(-angle))
    M = np.array([
        [scale * cos_a, -scale * sin_a,
         cx - scale * (cos_a * mouth_center[0] - sin_a * mouth_center[1])],
        [scale * sin_a,  scale * cos_a,
         cy - scale * (sin_a * mouth_center[0] + cos_a * mouth_center[1])],
    ], dtype=np.float64)
    return M


def warp_frame(frame_bgr, M, out_size=OUT_SIZE):
    """Apply affine warp + convert to grayscale 96x96."""
    warped = cv2.warpAffine(frame_bgr, M, (out_size, out_size),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray


def process_video(input_path, output_path, task_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FaceLandmarker (IMAGE mode for frame-by-frame)
    base_opts = python.BaseOptions(model_asset_path=task_path)
    opts = FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(opts)

    # VideoWriter — grayscale saved as BGR (OpenCV requirement)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_SIZE, OUT_SIZE), isColor=False)

    prev_M      = None
    detected    = 0
    frame_idx   = 0

    print(f"Processing {total} frames @ {fps:.1f}fps → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = get_landmarks(landmarker, frame_rgb)

        if pts is not None:
            M = compute_affine(pts)
            if M is not None:
                prev_M = M
                detected += 1

        if prev_M is not None:
            gray = warp_frame(frame, prev_M)
        else:
            # fallback: center crop + resize
            h, w = frame.shape[:2]
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = frame[y0:y0+s, x0:x0+s]
            gray = cv2.cvtColor(cv2.resize(crop, (OUT_SIZE, OUT_SIZE)), cv2.COLOR_BGR2GRAY)

        out.write(gray)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total}  detection rate: {detected/frame_idx*100:.1f}%")

    cap.release()
    out.release()
    landmarker.close()

    rate = detected / max(frame_idx, 1) * 100
    print(f"Done. {frame_idx} frames, detection rate: {rate:.1f}%")
    if rate < 90:
        print(f"  WARNING: detection rate {rate:.1f}% < 90% — check input video quality")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output mouth crop video path")
    parser.add_argument("--task",   default="face_landmarker.task",
                        help="Path to face_landmarker.task model file")
    args = parser.parse_args()

    if not os.path.exists(args.task):
        raise FileNotFoundError(f"Task file not found: {args.task}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_video(args.input, args.output, args.task)


if __name__ == "__main__":
    main()