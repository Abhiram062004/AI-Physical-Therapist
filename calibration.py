"""
calibration.py
--------------
Guides the user through a short calibration phase to measure their personal
range of motion and sets adaptive thresholds for the rep counter and
form checker.

Usage (called from app.py):
    cal = Calibrator()
    thresholds = cal.run(frames_list)  # list of (33,3) pose frames
"""

import numpy as np
import json
import os

L_HIP, R_HIP   = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE,R_ANKLE= 27, 28
L_WRIST,R_WRIST= 15, 16
L_SHOULDER,R_SHOULDER = 11, 12
NOSE = 0

CALIBRATION_FILE = "user_calibration.json"


def _pt(frame, idx):
    return frame[idx, :3]


def _angle(a, b, c):
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class Calibrator:
    """
    Collects ~5 seconds of the user standing naturally and doing one max-effort
    movement per calibration exercise, then extracts personal thresholds.
    """

    INSTRUCTIONS = [
        ("Stand tall and relaxed for 3 seconds.", 3),
        ("Do ONE slow, deep squat and come back up.", 5),
        ("Lift one knee as HIGH as you can, then the other.", 4),
    ]

    def __init__(self):
        self.thresholds = {}

    def compute_from_frames(self, frames: list) -> dict:
        """
        Parameters
        ----------
        frames : list of np.ndarray, each shape (33, 3)

        Returns a dict of personal thresholds.
        """
        if len(frames) < 10:
            return self._defaults()

        arr = np.array(frames)   # (T, 33, 3)

        # ── Standing hip height (baseline) ──────────────────────────
        hip_ys   = (arr[:, L_HIP, 1] + arr[:, R_HIP, 1]) / 2
        knee_ys  = (arr[:, L_KNEE, 1] + arr[:, R_KNEE, 1]) / 2
        ankle_ys = (arr[:, L_ANKLE, 1] + arr[:, R_ANKLE, 1]) / 2

        standing_hip_y = float(np.percentile(hip_ys, 10))   # highest (lowest y)
        max_hip_y      = float(np.percentile(hip_ys, 90))   # deepest position

        # Squat depth threshold: midpoint between standing and deepest
        squat_depth = float((standing_hip_y + max_hip_y) / 2)

        # ── High-knee threshold ──────────────────────────────────────
        hip_knee_diffs = hip_ys - knee_ys   # positive when knee is higher
        high_knee_thresh = float(np.percentile(hip_knee_diffs, 75) * 0.6)

        # ── Arm reach (shoulder-to-wrist y distance) ─────────────────
        wrist_ys    = (arr[:, L_WRIST, 1] + arr[:, R_WRIST, 1]) / 2
        sho_ys      = (arr[:, L_SHOULDER, 1] + arr[:, R_SHOULDER, 1]) / 2
        arm_reach   = float(np.min(wrist_ys - sho_ys))   # most negative = arms highest

        self.thresholds = {
            "squat_depth":         squat_depth,
            "high_knee_threshold": max(0.04, high_knee_thresh),
            "butt_kick_threshold": 0.08,
            "arm_reach":           arm_reach,
            "standing_hip_y":      standing_hip_y,
        }
        return self.thresholds

    def _defaults(self) -> dict:
        return {
            "squat_depth":         0.55,
            "high_knee_threshold": 0.06,
            "butt_kick_threshold": 0.08,
            "arm_reach":           -0.1,
            "standing_hip_y":      0.45,
        }

    def save(self, path: str = CALIBRATION_FILE):
        with open(path, "w") as f:
            json.dump(self.thresholds, f, indent=2)

    @staticmethod
    def load(path: str = CALIBRATION_FILE) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}
