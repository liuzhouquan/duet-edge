"""
17-metric Laban Movement Analysis (LMA) feature extraction.

Follows Zhou et al. "Real-Time Full-body Interaction with AI Dance Models"
(IUI '25) Section 4.2: Body (8) + Effort (7) + Space (2) = 17 metrics.

Shape component (f20-f29) is INTENTIONALLY excluded, matching the paper's
methodology: "redundancies in measuring space covered, where we focused on
kinesphere metrics (relative body and limb distances) rather than pure
distance metrics."

The mapping to the paper:

  Body component (8 metrics, indices f1-f8)
    f1   feet to hips distance
    f2   hands to shoulders distance
    f3   left hand to right hand distance
    f4   hands to head distance
    f5   hands to hips distance
    f6   pelvis height
    f7   hips to ground minus feet to hips distance
    f8   gait distance

  Effort component (7 metrics, indices f11, f13-f15, f17-f19)
    f11  pelvis velocity
    f13  hands velocity
    f14  feet velocity
    f15  pelvis acceleration
    f17  hands acceleration
    f18  feet acceleration
    f19  pelvis rate of acceleration (jerk)

  Space component (2 metrics, indices f30-f31)
    f30  distance covered over time
    f31  area on ground covered over time

Aggregation per paper Section 4.2 ("computed the max, mean, and standard
deviation within a 3.7-second segment"): each metric -> 3 scalars,
giving 51 scalars per segment.
"""

from __future__ import annotations

import numpy as np


# Same SMPL 24-joint map as eval/lma_features.py and the upstream
# dance-ai-research-project/feature_utils.py.
JOINT_IDX = {
    "Pelvis": 0,    "RHip": 1,      "LHip": 2,       "spine1": 3,
    "RKnee": 4,     "LKnee": 5,     "spine2": 6,     "RAnkle": 7,
    "LAnkle": 8,    "spine3": 9,    "RFoot": 10,     "LFoot": 11,
    "Neck": 12,     "RCollar": 13,  "LCollar": 14,   "Head": 15,
    "RShoulder": 16,"LShoulder": 17,"RElbow": 18,    "LElbow": 19,
    "RWrist": 20,   "LWrist": 21,   "RHand": 22,     "LHand": 23,
}

# Ordered metric names — index here is the metric index in the [T, 17] tensor.
METRIC_NAMES = [
    # Body 8
    "f1_feet_to_hips",
    "f2_hands_to_shoulders",
    "f3_lhand_to_rhand",
    "f4_hands_to_head",
    "f5_hands_to_hips",
    "f6_pelvis_height",
    "f7_hipground_minus_feethip",
    "f8_gait_distance",
    # Effort 7
    "f11_pelvis_velocity",
    "f13_hands_velocity",
    "f14_feet_velocity",
    "f15_pelvis_acceleration",
    "f17_hands_acceleration",
    "f18_feet_acceleration",
    "f19_pelvis_jerk",
    # Space 2
    "f30_distance_covered",
    "f31_area_covered",
]
N_METRICS = len(METRIC_NAMES)
assert N_METRICS == 17

# Group memberships for grouped reporting in the appendix.
COMPONENT_OF = (
    ["Body"]   * 8
  + ["Effort"] * 7
  + ["Space"]  * 2
)

# Window for the area-covered rolling stat (f31). 35 frames at 30fps ≈ 1.17s,
# matching the upstream feature_utils.py choice of `period="35i"`.
AREA_WINDOW = 35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _j(joints: np.ndarray, name: str) -> np.ndarray:
    """Take joint `name`'s [T, 3] position trajectory."""
    return joints[:, JOINT_IDX[name], :]


def _norm(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm: [..., 3] -> [...]."""
    return np.linalg.norm(v, axis=-1)


def _speed(seq: np.ndarray) -> np.ndarray:
    """Per-frame speed of a [T, 3] trajectory. First frame is 0."""
    v = np.zeros(seq.shape[0])
    v[1:] = _norm(np.diff(seq, axis=0))
    return v


def _shoelace_area_xz(pts: np.ndarray) -> float:
    """|signed area| of the polygon traced by `pts` ([N, 2], xz-coords)."""
    if len(pts) < 3:
        return 0.0
    x, z = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_17_timeseries(joints: np.ndarray) -> np.ndarray:
    """Per-frame value of each of the 17 LMA metrics.

    Args
    ----
    joints : np.ndarray [T, 24, 3]
        World-coordinate SMPL joint positions (z-up; the same convention as
        EDGE's SMPLSkeleton output after the y→z rotation).

    Returns
    -------
    np.ndarray [T, 17]
        Column k holds the per-frame trajectory of `METRIC_NAMES[k]`.

    Notes
    -----
    * f31 ("area on ground covered over time") is not a true per-frame
      quantity. We assign it the shoelace area of the pelvis xz trajectory
      over a trailing `AREA_WINDOW`-frame window ending at frame i. This is
      faithful to the upstream `calculate_space_component`, where the area
      is computed inside a sliding 35-frame window.
    * Frames before AREA_WINDOW-1 use a shorter window (the area starts at 0
      and grows). Pearson over the f31 column should be interpreted with
      this warm-up in mind for very short clips.
    """
    T = joints.shape[0]
    out = np.zeros((T, N_METRICS), dtype=np.float64)

    # ── Body 8 ───────────────────────────────────────────────────────
    f1 = (_norm(_j(joints, "LFoot") - _j(joints, "LHip")) +
          _norm(_j(joints, "RFoot") - _j(joints, "RHip"))) / 2
    f2 = (_norm(_j(joints, "LHand") - _j(joints, "LShoulder")) +
          _norm(_j(joints, "RHand") - _j(joints, "RShoulder"))) / 2
    f3 = _norm(_j(joints, "RHand") - _j(joints, "LHand"))
    f4 = (_norm(_j(joints, "LHand") - _j(joints, "Head")) +
          _norm(_j(joints, "RHand") - _j(joints, "Head"))) / 2
    f5 = (_norm(_j(joints, "LHand") - _j(joints, "LHip")) +
          _norm(_j(joints, "RHand") - _j(joints, "RHip"))) / 2
    f6 = _j(joints, "Pelvis")[:, 2]   # z is up after EDGE's rotation
    hip_height = (_j(joints, "LHip")[:, 2] + _j(joints, "RHip")[:, 2]) / 2
    f7 = f1 - hip_height
    f8 = _norm(_j(joints, "LFoot") - _j(joints, "RFoot"))

    # ── Effort 7 ─────────────────────────────────────────────────────
    pelvis_v = _speed(_j(joints, "Pelvis"))
    lhand_v  = _speed(_j(joints, "LHand"))
    rhand_v  = _speed(_j(joints, "RHand"))
    lfoot_v  = _speed(_j(joints, "LFoot"))
    rfoot_v  = _speed(_j(joints, "RFoot"))

    f11 = pelvis_v
    f13 = (lhand_v + rhand_v) / 2
    f14 = (lfoot_v + rfoot_v) / 2

    f15 = np.zeros_like(f11); f15[1:] = np.diff(f11)
    f17 = np.zeros_like(f13); f17[1:] = np.diff(f13)
    f18 = np.zeros_like(f14); f18[1:] = np.diff(f14)
    f19 = np.zeros_like(f15); f19[1:] = np.diff(f15)

    # ── Space 2 ──────────────────────────────────────────────────────
    pelvis_xy = _j(joints, "Pelvis")[:, [0, 1]]  # xy-plane in z-up coords
    disp = np.zeros(T)
    disp[1:] = _norm(np.diff(pelvis_xy, axis=0))
    f30 = disp

    f31 = np.zeros(T)
    for i in range(T):
        s = max(0, i - AREA_WINDOW + 1)
        f31[i] = _shoelace_area_xz(pelvis_xy[s:i + 1])

    # ── Stack ────────────────────────────────────────────────────────
    out[:,  0] = f1
    out[:,  1] = f2
    out[:,  2] = f3
    out[:,  3] = f4
    out[:,  4] = f5
    out[:,  5] = f6
    out[:,  6] = f7
    out[:,  7] = f8
    out[:,  8] = f11
    out[:,  9] = f13
    out[:, 10] = f14
    out[:, 11] = f15
    out[:, 12] = f17
    out[:, 13] = f18
    out[:, 14] = f19
    out[:, 15] = f30
    out[:, 16] = f31
    return out


def aggregate_paper(ts: np.ndarray) -> np.ndarray:
    """Paper-style aggregation: max, mean, std per metric.

    Args
    ----
    ts : np.ndarray [T, 17]
        Output of `extract_17_timeseries`.

    Returns
    -------
    np.ndarray [17, 3]
        Row k = (max, mean, std) of metric k. Flatten for a 51-dim segment
        descriptor.
    """
    return np.stack([ts.max(axis=0), ts.mean(axis=0), ts.std(axis=0)], axis=-1)


__all__ = [
    "METRIC_NAMES",
    "COMPONENT_OF",
    "N_METRICS",
    "extract_17_timeseries",
    "aggregate_paper",
]
