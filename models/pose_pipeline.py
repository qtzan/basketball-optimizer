from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class PosePipeline:
    """Lightweight wrapper around the YOLO pose model for offline experimentation."""

    def __init__(self, path: str = "yolov8n-pose.pt") -> None:
        self.path = path
        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Eagerly load YOLO and provide clearer error surfaces."""
        if self.model is not None:
            return

        try:
            self.model = YOLO(self.path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError(f"Model file not found: {self.path}") from None
        except Exception as exc:  # pragma: no cover - hardware/env specific
            raise RuntimeError(f"Failed to load model from {self.path}: {exc}") from exc

    def process_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        overlay_limit: int = 25,
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        """Run pose extraction frame-by-frame.

        Returns a tuple containing:
        1. pandas.DataFrame with flattened keypoints + confidence metrics
        2. List of annotated frames (BGR) that can be persisted with cv2.imwrite
        """
        if self.model is None:
            self._load_model()

        path = Path(video_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Video path does not exist: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")

        records: list[dict[str, float]] = []
        overlay_frames: list[np.ndarray] = []
        frame_index = 0

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                success, frame = capture.read()
                if not success:
                    break

                results = self.model(frame, verbose=False)
                if not isinstance(results, (list, tuple)):
                    results = [results]

                for result in results:
                    annotated = result.plot() if hasattr(result, "plot") else None
                    if annotated is not None and len(overlay_frames) < overlay_limit:
                        overlay_frames.append(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    keypoints = getattr(result, "keypoints", None)
                    if keypoints is None or keypoints.xy is None:
                        continue

                    xy = _to_numpy(keypoints.xy)
                    conf = _to_numpy(keypoints.conf) if keypoints.conf is not None else None

                    for person_idx in range(xy.shape[0]):
                        record = {
                            "frame_index": frame_index,
                            "person_index": person_idx,
                        }

                        if conf is not None:
                            confidences = conf[person_idx]
                            record["mean_confidence"] = float(np.mean(confidences))
                            record["min_confidence"] = float(np.min(confidences))
                            record["max_confidence"] = float(np.max(confidences))
                        else:
                            record["mean_confidence"] = 0.0
                            record["min_confidence"] = 0.0
                            record["max_confidence"] = 0.0

                        for kp_idx, (x_coord, y_coord) in enumerate(xy[person_idx]):
                            record[f"kpt_{kp_idx}_x"] = float(x_coord)
                            record[f"kpt_{kp_idx}_y"] = float(y_coord)
                            if conf is not None:
                                record[f"kpt_{kp_idx}_conf"] = float(conf[person_idx][kp_idx])

                        records.append(record)

                frame_index += 1
        finally:
            capture.release()

        return pd.DataFrame.from_records(records), overlay_frames


def _to_numpy(value: np.ndarray | "torch.Tensor" | None) -> np.ndarray:
    """Convert tensors to numpy without forcing torch import in callers."""
    if value is None:
        return np.empty((0, 0, 0))

    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)