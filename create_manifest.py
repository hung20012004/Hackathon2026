"""
create_manifest.py
Input : original video (có audio) + mouth crop video
Output: test.tsv + test.wrd trong output_dir

Usage:
    python create_manifest.py \
        --video  /path/to/original.mp4 \
        --mouth  /path/to/mouth.mp4 \
        --outdir /path/to/manifest_dir \
        [--whisper_model base]
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import whisper


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract_audio(video_path, wav_path):
    """Extract audio from video to wav 16kHz mono."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)


def transcribe(video_path, whisper_model="base"):
    """Use Whisper to transcribe audio from video. Returns uppercase text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        extract_audio(video_path, wav_path)
        model = whisper.load_model(whisper_model)
        result = model.transcribe(wav_path, language="en")
        text = result["text"].strip().upper()
        # clean: keep only letters, spaces, apostrophes
        text = " ".join(text.split())
        return text
    finally:
        os.unlink(wav_path)


def create_manifest(video_path, mouth_path, outdir, whisper_model="base"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    video_path = str(Path(video_path).resolve())
    mouth_path = str(Path(mouth_path).resolve())

    print(f"Transcribing {video_path} with Whisper {whisper_model}...")
    transcript = transcribe(video_path, whisper_model)
    print(f"Transcript: {transcript}")

    n_frames = count_frames(mouth_path)
    print(f"Mouth video frames: {n_frames}")

    # TSV format (from dataset.py):
    # line 0: root path (dummy, absolute paths used)
    # line 1+: id \t video_path \t audio_path \t n_video_frames \t n_audio_frames
    # audio not used (video-only mode) — use mouth video path as dummy
    root = str(outdir.resolve())
    tsv_path = outdir / "test.tsv"
    wrd_path = outdir / "test.wrd"

    with open(tsv_path, "w") as f:
        f.write(root + "\n")
        # video_path relative to root, audio_path same (dummy), frames, frames
        video_rel = os.path.relpath(mouth_path, root)
        audio_rel = os.path.relpath(mouth_path, root)
        f.write(f"test_sample\t{video_rel}\t{audio_rel}\t{n_frames}\t{n_frames}\n")

    with open(wrd_path, "w") as f:
        f.write(transcript + "\n")

    print(f"Saved: {tsv_path}")
    print(f"Saved: {wrd_path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          required=True, help="Original video (with audio)")
    parser.add_argument("--mouth",          required=True, help="Mouth crop video (96x96)")
    parser.add_argument("--outdir",         required=True, help="Output directory for manifest")
    parser.add_argument("--whisper_model",  default="base", help="Whisper model size")
    args = parser.parse_args()

    create_manifest(args.video, args.mouth, args.outdir, args.whisper_model)


if __name__ == "__main__":
    main()