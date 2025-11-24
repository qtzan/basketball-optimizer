"""Manual CLI for running the pose pipeline without the API layer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from ml.scoring import score_pose_dataframe
from models.pose_pipeline import PosePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a video and score the detected poses.")
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/annotated_frame.png"),
        help="Path where the first annotated frame will be stored.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional custom YOLOv8 pose weights (defaults to yolov8n-pose).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to analyze (default: entire video).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser()
    if not video_path.exists() or not video_path.is_file():
        print(f"Error: {video_path} does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)

    weights = str(args.model.expanduser()) if args.model else "yolov8n-pose.pt"
    pipeline = PosePipeline(path=weights)

    results_df, overlay_frames = pipeline.process_video(
        video_path, max_frames=args.max_frames, overlay_limit=1
    )

    score, strengths, weaknesses = score_pose_dataframe(results_df)

    frames_processed = (
        int(results_df["frame_index"].nunique()) if "frame_index" in results_df.columns else 0
    )
    print(f"Frames processed: {frames_processed}")
    print(f"Detections captured: {len(results_df.index)}")
    print(f"Score: {score}")
    print("Strengths:")
    for item in strengths or ["(none)"]:
        print(f"  - {item}")
    print("Weaknesses:")
    for item in weaknesses or ["(none)"]:
        print(f"  - {item}")

    if overlay_frames:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(args.output), overlay_frames[0]):
            print(f"Annotated frame saved to {args.output}")
        else:
            print(f"Failed to write annotated frame to {args.output}", file=sys.stderr)
    else:
        print("No overlay frames returned; skipping PNG export.", file=sys.stderr)


if __name__ == "__main__":
    main()
