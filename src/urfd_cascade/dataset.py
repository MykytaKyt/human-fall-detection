"""UR Fall Detection (URFD) dataset access: label loading and frame
enumeration. See scripts/download_urfd.sh for acquiring the raw data.

Dataset: Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded
platform using depth maps and wireless accelerometer. Computer Methods and
Programs in Biomedicine, 117(3), 489-501.
https://doi.org/10.1016/j.cmpb.2014.09.005
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

FALLS_CSV = "urfall-cam0-falls.csv"
ADLS_CSV = "urfall-cam0-adls.csv"


def load_labels(data_dir: str | Path) -> dict[tuple[str, int], int]:
    """Load per-frame binary labels from the URFD annotation CSVs.

    URFD label column semantics: 1 = person on the ground (fall), -1 = not
    fallen, 0 = transitional. We binarise to {1: fall, 0: not-fall}, which
    is the standard protocol used in prior URFD-based work.
    """
    labels: dict[tuple[str, int], int] = {}
    for csv_name in (FALLS_CSV, ADLS_CSV):
        path = os.path.join(data_dir, csv_name)
        df = pd.read_csv(path, header=None)
        for _, row in df.iterrows():
            seq, frame, lab = str(row[0]), int(row[1]), int(row[2])
            labels[(seq, frame)] = 1 if lab == 1 else 0
    return labels


def frame_number(filename: str) -> int:
    """Extract the integer frame index from a URFD RGB filename such as
    'fall-01-cam0-rgb-023.png'."""
    m = re.search(r"-(\d+)\.png$", filename)
    return int(m.group(1)) if m else -1


def list_sequence_frames(frames_dir: str | Path, seq: str) -> list[str]:
    """List (sorted) RGB frame paths for one sequence, handling the
    nested '<seq>/<seq>-cam0-rgb/*.png' layout produced by the URFD zips.
    """
    subdir = os.path.join(frames_dir, seq, f"{seq}-cam0-rgb")
    if not os.path.isdir(subdir):
        candidates = glob.glob(os.path.join(frames_dir, seq, "*"))
        subdir = candidates[0] if candidates and os.path.isdir(candidates[0]) else os.path.join(frames_dir, seq)
    return sorted(glob.glob(os.path.join(subdir, "*.png")))


def list_sequences(frames_dir: str | Path) -> list[str]:
    """List all downloaded sequence names (e.g. ['fall-01', ..., 'adl-40'])."""
    return sorted(
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    )
