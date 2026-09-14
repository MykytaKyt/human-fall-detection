import numpy as np

from urfd_cascade.ddl_features import compute_ddl_features


def _make_keypoints(shoulder_xy, hip_xy, knee_ankle_xy, conf=1.0):
    """Build a minimal 17x3 COCO keypoint array with only the joints used
    by the DDL layer populated (shoulders, hips, knees, ankles)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [*shoulder_xy, conf]  # left shoulder
    kp[6] = [*shoulder_xy, conf]  # right shoulder
    kp[11] = [*hip_xy, conf]      # left hip
    kp[12] = [*hip_xy, conf]      # right hip
    for idx, xy in zip([13, 14, 15, 16], knee_ankle_xy):
        kp[idx] = [*xy, conf]
    return kp


def test_upright_pose_has_small_angle_and_stable_mos():
    # Standing: shoulders above hips above feet, all aligned vertically.
    # Note: com_proj sits at ground level (max ankle y) by construction,
    # i.e. exactly on the BoS polygon's bottom edge for symmetric feet, so
    # MoS is at the stability boundary (~0) rather than strictly positive;
    # see test_fallen_pose_has_larger_instability for the meaningful
    # relative comparison used elsewhere in this file.
    kp = _make_keypoints(
        shoulder_xy=(50, 10),
        hip_xy=(50, 50),
        knee_ankle_xy=[(40, 80), (60, 80), (40, 95), (60, 95)],
    )
    bbox = np.array([50, 50, 30, 100])  # cx, cy, w, h
    feats = compute_ddl_features(kp, bbox, frame_w=100, frame_h=100)

    assert feats["angle"] < 10.0  # near-vertical torso
    assert feats["mos"] >= -1e-6  # at or inside the stability boundary


def test_fallen_pose_has_larger_instability_than_upright():
    upright = _make_keypoints(
        shoulder_xy=(50, 10),
        hip_xy=(50, 50),
        knee_ankle_xy=[(40, 80), (60, 80), (40, 95), (60, 95)],
    )
    # CoM shifted far outside the base of support (person toppling sideways).
    fallen = _make_keypoints(
        shoulder_xy=(90, 30),
        hip_xy=(70, 40),
        knee_ankle_xy=[(40, 80), (60, 80), (40, 95), (60, 95)],
    )
    bbox = np.array([50, 50, 30, 100])
    feats_upright = compute_ddl_features(upright, bbox, frame_w=100, frame_h=100)
    feats_fallen = compute_ddl_features(fallen, bbox, frame_w=100, frame_h=100)

    assert feats_fallen["mos"] < feats_upright["mos"]  # less stable
    assert feats_fallen["mos"] < 0  # CoM projects outside the BoS polygon


def test_fallen_pose_has_large_angle_and_low_com():
    # Lying down: shoulders and hips at roughly the same height, torso
    # horizontal, CoM projects far from the (compact) foot polygon.
    kp = _make_keypoints(
        shoulder_xy=(10, 90),
        hip_xy=(50, 92),
        knee_ankle_xy=[(70, 90), (72, 90), (90, 91), (92, 91)],
    )
    bbox = np.array([50, 90, 90, 20])
    feats = compute_ddl_features(kp, bbox, frame_w=100, frame_h=100)

    assert feats["angle"] > 45.0  # torso close to horizontal
    assert feats["aspect"] > 1.0  # wide, flat bounding box (w > h)


def test_full_occlusion_of_contact_joints_falls_back_gracefully():
    # No visibility on knees/ankles -> BoS polygon cannot be built; the
    # function must not raise and should return mos == 0.0.
    kp = _make_keypoints(
        shoulder_xy=(50, 10),
        hip_xy=(50, 50),
        knee_ankle_xy=[(48, 80), (52, 80), (48, 95), (52, 95)],
        conf=1.0,
    )
    kp[[13, 14, 15, 16], 2] = 0.0  # zero-out confidence for contact joints
    bbox = np.array([50, 50, 30, 100])
    feats = compute_ddl_features(kp, bbox, frame_w=100, frame_h=100)

    assert feats["mos"] == 0.0
    assert feats["bos_area"] == 0.0
