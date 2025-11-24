"""Heuristic scoring for offline pose experiments."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def score_pose_dataframe(df: pd.DataFrame) -> Tuple[float, list[str], list[str]]:
    """Compute a simple score plus qualitative feedback from pose keypoints."""
    if df.empty:
        return (
            0.0,
            ["No detections were generated from the supplied clip."],
            ["Unable to score an empty result set."],
        )

    mean_conf = float(df["mean_confidence"].mean()) if "mean_confidence" in df.columns else 0.0
    frame_counts = df["frame_index"].nunique() if "frame_index" in df.columns else 0
    frame_conf_series = (
        df.groupby("frame_index")["mean_confidence"].mean()
        if {"frame_index", "mean_confidence"}.issubset(df.columns)
        else pd.Series(dtype=float)
    )
    stability = float(frame_conf_series.std()) if not frame_conf_series.empty else 0.0

    normalized_score = (mean_conf * 0.7) + (max(0.0, 1.0 - stability) * 0.3)
    normalized_score = max(0.0, min(1.0, normalized_score))
    score = round(normalized_score * 100, 2)

    strengths: list[str] = []
    weaknesses: list[str] = []

    if mean_conf >= 0.6:
        strengths.append("Detections show consistently high keypoint confidence.")
    else:
        weaknesses.append("Keypoint confidence is low; improve camera position or lighting.")

    if frame_counts >= 30:
        strengths.append("Sufficient frame coverage captured for analysis.")
    else:
        weaknesses.append("Video is short; longer clips improve scoring stability.")

    if stability <= 0.1:
        strengths.append("Pose appears stable across frames.")
    else:
        weaknesses.append("Pose varies significantly frame-to-frame.")

    return score, strengths, weaknesses
