"""Kinematic cascade for posture-anomaly (fall) detection.

Implements the Dynamics Decoupling Layer (DDL) inverted-pendulum features
(CoM, BoS, MoS) on top of YOLOv8-Pose keypoints, and three temporal
classifiers (physics threshold, LSTM, LTC) evaluated on UR Fall Detection.
"""

__version__ = "0.1.0"
