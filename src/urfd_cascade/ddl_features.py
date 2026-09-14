"""Dynamics Decoupling Layer (DDL): converts COCO-17 keypoints from
YOLOv8-Pose into scale-invariant inverted-pendulum features (CoM, BoS,
MoS, torso angle) as described in the paper's kinematic cascade.
"""
from __future__ import annotations

import numpy as np

from .geometry import convex_hull, margin_of_stability, polygon_area

# COCO-17 keypoint indices
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

CONTACT_JOINTS = [L_KNEE, R_KNEE, L_ANK, R_ANK]
FEATURE_NAMES = [
    "com_x", "com_y", "angle", "mos", "bos_area",
    "vy", "vx", "ay", "aspect", "bbox_h", "mean_conf",
]


def compute_ddl_features(
    keypoints: np.ndarray,
    bbox_xywh: np.ndarray,
    frame_w: int,
    frame_h: int,
    visibility_threshold: float = 0.3,
) -> dict:
    """Compute a single frame's DDL features from one detected person.

    Args:
        keypoints: (17, 3) array of (x, y, confidence) in pixel coords.
        bbox_xywh: (4,) array (cx, cy, w, h) in pixel coords.
        frame_w, frame_h: frame dimensions, used for spatial normalisation.
        visibility_threshold: minimum confidence for a contact joint to be
            included in the base-of-support polygon.

    Returns:
        Dict with keys: com_x, com_y (normalised [0,1]), angle (degrees
        from vertical), mos (normalised by bbox height), bos_area
        (normalised by bbox height squared), mean_conf, bbox_h (normalised),
        aspect (w/h). Velocity/acceleration keys (vy, vx, ay) are left at
        0.0 here; they are filled in by add_temporal_derivatives() once a
        full sequence is available.
    """
    xy = keypoints[:, :2]
    v = keypoints[:, 2]
    bh = bbox_xywh[3] + 1e-6

    # CoM: visibility-weighted midpoint of the hips (Eq. eq:com)
    hips = [L_HIP, R_HIP]
    wsum = v[hips].sum() + 1e-6
    com = (v[hips, None] * xy[hips]).sum(0) / wsum

    # Torso angle from vertical (shoulder midpoint -> hip midpoint)
    sh = xy[[L_SH, R_SH]].mean(0)
    hp = xy[[L_HIP, R_HIP]].mean(0)
    vec = hp - sh
    angle = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))

    # BoS: convex hull of visible knee/ankle joints (Eq. eq:mos)
    contact = [i for i in CONTACT_JOINTS if v[i] > visibility_threshold]
    poly = convex_hull(xy[contact]) if len(contact) >= 3 else np.array([])
    ground_y = xy[contact][:, 1].max() if contact else com[1]
    com_proj = np.array([com[0], ground_y])
    d_mos = margin_of_stability(com_proj, poly) if len(poly) >= 3 else 0.0
    bos_area = polygon_area(poly) if len(poly) >= 3 else 0.0

    return dict(
        com_x=com[0] / frame_w,
        com_y=com[1] / frame_h,
        angle=angle,
        mos=d_mos / bh,
        bos_area=bos_area / (bh * bh),
        vy=0.0, vx=0.0, ay=0.0,
        mean_conf=float(v.mean()),
        bbox_h=bh / frame_h,
        aspect=float(bbox_xywh[2] / bh),
    )


def add_temporal_derivatives(com_x: np.ndarray, com_y: np.ndarray) -> dict:
    """Compute CoM velocity/acceleration via finite differences over a
    per-sequence time series (called once per sequence, after all frames'
    static DDL features have been extracted).
    """
    vy = np.gradient(com_y)
    vx = np.gradient(com_x)
    ay = np.gradient(vy)
    return dict(vy=vy, vx=vx, ay=ay)
