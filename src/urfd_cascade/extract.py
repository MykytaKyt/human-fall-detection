"""End-to-end feature extraction: YOLOv8-Pose inference over every URFD
frame, followed by DDL feature computation and temporal-derivative
augmentation. Produces the features.parquet file consumed by evaluate.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import frame_number, list_sequence_frames, list_sequences, load_labels
from .ddl_features import compute_ddl_features

logger = logging.getLogger(__name__)


def extract_dataset_features(
    data_dir: str | Path,
    frames_dir: str | Path,
    model_name: str = "yolov8n-pose.pt",
    device: int | str = 0,
) -> pd.DataFrame:
    """Run YOLOv8-Pose over all sequences and compute DDL features.

    Returns a DataFrame with one row per frame: seq, frame, label, valid,
    and the DDL feature columns (see ddl_features.FEATURE_NAMES).
    """
    from ultralytics import YOLO  # local import: heavy, GPU-only dependency

    labels = load_labels(data_dir)
    model = YOLO(model_name)
    rows: list[dict] = []

    for seq in list_sequences(frames_dir):
        imgs = list_sequence_frames(frames_dir, seq)
        if not imgs:
            logger.warning("no images found for sequence %s", seq)
            continue
        results = model.predict(imgs, verbose=False, stream=True, device=device)
        for img_path, res in zip(imgs, results):
            fn = frame_number(Path(img_path).name)
            h, w = res.orig_shape
            rec = dict(seq=seq, frame=fn, label=labels.get((seq, fn), 0), H=h, W=w)

            has_person = (
                res.keypoints is not None and len(res.keypoints) > 0
                and res.boxes is not None and len(res.boxes) > 0
            )
            if not has_person:
                rec["valid"] = 0
                rows.append(rec)
                continue

            # pick the largest detected person by bbox area
            areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
            idx = int(np.argmax(areas))
            kp = res.keypoints.data[idx].cpu().numpy()
            box = res.boxes.xywh[idx].cpu().numpy()

            feats = compute_ddl_features(kp, box, frame_w=w, frame_h=h)
            rec["valid"] = 1
            rec.update(feats)
            rows.append(rec)
        logger.info("%s: %d frames processed", seq, len(imgs))

    df = pd.DataFrame(rows).sort_values(["seq", "frame"]).reset_index(drop=True)
    return _add_temporal_derivatives(df)


def _add_temporal_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("vy", "vx", "ay"):
        if col not in df:
            df[col] = 0.0
    for seq, group in df.groupby("seq"):
        cy = group["com_y"].ffill().fillna(0).to_numpy()
        cx = group["com_x"].ffill().fillna(0).to_numpy()
        vy = np.gradient(cy)
        vx = np.gradient(cx)
        ay = np.gradient(vy)
        df.loc[group.index, "vy"] = vy
        df.loc[group.index, "vx"] = vx
        df.loc[group.index, "ay"] = ay
    return df
