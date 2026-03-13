"""
demo.py – Pipeline trích xuất vùng miệng từ video MP4
══════════════════════════════════════════════════════

Input : Video MP4 bất kỳ resolution (mặt nhìn thẳng hoặc hơi nghiêng)
Output:
  • output/frames/  – từng frame PNG  (96×96, grayscale)
  • output/mouth_aligned.mp4 – video tổng hợp (grayscale, 25 fps)

Usage:
    python demo.py --video input.mp4
    python demo.py --video input.mp4 --out results/ --preview
    python demo.py --video input.mp4 --no3d          # tắt 3D correction
    python demo.py --video input.mp4 --max-frames 200
"""

from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

from src.face_detector import FaceLandmarker
from src.affine_normalizer import AffineNormalizer, OUTPUT_SIZE
from src.landmarks import ALL_LIP_INDICES, AFFINE_SRC_INDICES

# ── Hằng số đầu ra ────────────────────────────────────────────────────────────
OUT_FPS  = 25
OUT_SIZE = OUTPUT_SIZE   # 96

# ── Model mặc định ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "face_landmarker_v2_with_blendshapes.task"


# ══════════════════════════════════════════════════════════════════════════════
# Vẽ overlay debug lên frame gốc
# ══════════════════════════════════════════════════════════════════════════════

def draw_debug_overlay(frame: np.ndarray, face) -> np.ndarray:
    """Vẽ lip landmarks + 3 điểm affine key lên frame gốc (để preview)."""
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Tất cả lip landmarks – xanh lá
    lip_pts = face.to_pixels(ALL_LIP_INDICES).astype(int)
    for x, y in lip_pts:
        cv2.circle(vis, (x, y), 1, (0, 220, 0), -1)

    # 3 điểm affine key – đỏ + nhãn
    key_pts = face.to_pixels(AFFINE_SRC_INDICES).astype(int)
    labels  = ["L", "R", "T"]
    colors  = [(0, 0, 255), (255, 0, 0), (0, 165, 255)]
    for (x, y), label, color in zip(key_pts, labels, colors):
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.putText(vis, label, (x + 6, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def make_preview_tile(
    original: np.ndarray,
    mouth_gray: np.ndarray | None,
) -> np.ndarray:
    """
    Ghép [frame gốc (thu nhỏ) | mouth patch (phóng to)] ngang nhau.
    Dùng để hiển thị preview hoặc lưu debug video.
    """
    TILE_H = 240

    # Frame gốc → thu nhỏ về TILE_H
    ratio = TILE_H / original.shape[0]
    orig_small = cv2.resize(original, (max(1, int(original.shape[1] * ratio)), TILE_H))

    # Mouth patch → 96×96 grayscale → BGR → phóng to cho dễ xem
    scale_m = TILE_H / OUT_SIZE
    if mouth_gray is not None:
        m_bgr = cv2.cvtColor(mouth_gray, cv2.COLOR_GRAY2BGR)
        m_big = cv2.resize(m_bgr, (int(OUT_SIZE * scale_m), TILE_H),
                           interpolation=cv2.INTER_NEAREST)
    else:
        m_big = np.zeros((TILE_H, TILE_H, 3), dtype=np.uint8)
        cv2.putText(m_big, "NO FACE", (8, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # Separator
    sep = np.full((TILE_H, 2, 3), 60, dtype=np.uint8)

    tile = np.hstack([orig_small, sep, m_big])

    # Nhãn
    cv2.putText(tile, "Original + landmarks",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(tile, f"Affine 96x96 gray",
                (orig_small.shape[1] + 10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return tile


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════

def process_video(args) -> None:
    # ── Kiểm tra model ────────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Download bằng lệnh:")
        print(
            "  wget -O face_landmarker_v2_with_blendshapes.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        sys.exit(1)

    # ── Mở video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_src = min(total_src, args.max_frames)

    # Tỉ lệ resample: giữ 25 fps từ nguồn bất kỳ fps
    frame_step = max(1, round(src_fps / OUT_FPS))

    print(f"Video  : {args.video}  [{src_w}×{src_h} @ {src_fps:.1f}fps]")
    print(f"Frames : {total_src}  →  output ~{total_src // frame_step} frames @ {OUT_FPS}fps")
    print(f"Output : {args.out}/")
    print(f"3D     : {'ON' if not args.no3d else 'OFF'}")
    print()

    # ── Chuẩn bị thư mục output ──────────────────────────────────────────────
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # VideoWriter cho mouth patch (grayscale → lưu dưới dạng BGR gray)
    mouth_video_path = os.path.join(args.out, "mouth_aligned.mp4")
    vw_mouth = cv2.VideoWriter(
        mouth_video_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        OUT_FPS,
        (OUT_SIZE, OUT_SIZE),
        isColor=False,   # grayscale
    )

    # VideoWriter cho preview tile (optional)
    preview_video_path = os.path.join(args.out, "preview.mp4")
    tile_w = int(240 * src_w / src_h) + 2 + 240   # rough estimate
    vw_preview = None   # khởi tạo sau khi có tile thật đầu tiên

    # ── Khởi tạo model ───────────────────────────────────────────────────────
    with FaceLandmarker(
        model_path=args.model,
        num_faces=1,
        mode="video",
    ) as detector:
        normalizer = AffineNormalizer(use_3d=not args.no3d)

        src_idx   = 0   # index frame nguồn
        out_idx   = 0   # index frame output đã ghi
        no_face_n = 0

        pbar = tqdm(total=total_src, unit="frame", desc="Processing")

        while True:
            ret, frame = cap.read()
            if not ret or src_idx >= total_src:
                break

            # Resample về 25 fps
            if src_idx % frame_step != 0:
                src_idx += 1
                pbar.update(1)
                continue

            # timestamp theo src_fps (bắt buộc tăng dần trong VIDEO mode)
            ts_ms = int(src_idx * 1000 / src_fps)

            # ── Detect ───────────────────────────────────────────────────────
            faces = detector.detect_video(frame, ts_ms)

            mouth_gray = None
            if faces:
                mouth_gray = normalizer.normalize(frame, faces[0])

            # ── Ghi frame output ─────────────────────────────────────────────
            if mouth_gray is not None:
                # 1) PNG riêng lẻ
                cv2.imwrite(
                    os.path.join(frames_dir, f"{out_idx:06d}.png"),
                    mouth_gray,
                )
                # 2) Video mouth (grayscale)
                vw_mouth.write(mouth_gray)
            else:
                # Ghi frame đen để giữ đồng bộ thời gian
                black = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
                vw_mouth.write(black)
                no_face_n += 1

            # ── Preview tile ─────────────────────────────────────────────────
            if args.preview or vw_preview is not None or True:
                vis = draw_debug_overlay(frame, faces[0]) if faces else frame.copy()
                tile = make_preview_tile(vis, mouth_gray)

                if vw_preview is None:
                    th, tw = tile.shape[:2]
                    vw_preview = cv2.VideoWriter(
                        preview_video_path,
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        OUT_FPS,
                        (tw, th),
                    )
                vw_preview.write(tile)

                if args.preview:
                    cv2.imshow("FaceLandmarker – Affine mouth (press q to stop)", tile)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            out_idx += 1
            src_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    vw_mouth.release()
    if vw_preview:
        vw_preview.release()
    if args.preview:
        cv2.destroyAllWindows()

    # ── Tóm tắt ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Tổng frames output : {out_idx}")
    print(f"Frames không face  : {no_face_n} ({100*no_face_n/max(out_idx,1):.1f}%)")
    print(f"Frames PNG         → {frames_dir}/")
    print(f"Video grayscale    → {mouth_video_path}")
    print(f"Preview video      → {preview_video_path}")
    print(f"{'='*50}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Mouth region extractor – Affine Transform → 96×96 grayscale"
    )
    p.add_argument("--video",       required=True, help="Input MP4 video")
    p.add_argument("--out",         default="output", help="Output directory")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help="Path to MediaPipe .task model file")
    p.add_argument("--no3d",        action="store_true",
                   help="Disable 3D head-pose correction")
    p.add_argument("--preview",     action="store_true",
                   help="Show real-time preview window")
    p.add_argument("--max-frames",  type=int, default=0,
                   help="Limit processing to first N source frames (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
