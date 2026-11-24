"""
core/form_checker.py
--------------------
Provides per-joint feedback messages based on deviation from the
trained rest-position pose.

FIXES in this version
---------------------
1. Only checks form BETWEEN reps (when is_in_rep=False). During movement
   the person is never at the rest position, so checking during a rep always
   produced false positives. Now form is only checked when the user is
   standing at rest — which is when the form_stats baseline is meaningful.

2. Uses x and y coordinates only (not z). MediaPipe depth (z) is estimated
   from image proportions and is unreliable across cameras and distances.

3. DEVIATION_THRESHOLD raised 2.0 → 2.5 — slightly more forgiving now that
   the form_stats baseline is correctly trained on rest frames only.

Overall form SCORING (path deviation during reps) is handled by RepCounter.
This module supplies feedback message strings and skeleton color only.
"""

import numpy as np
from core.exercise_loader import LoadedExercise

DEVIATION_THRESHOLD = 2.5   # std deviations before a joint triggers feedback


def check_form(exercise: LoadedExercise, pose_frame: np.ndarray,
               is_in_rep: bool = False) -> dict:
    """
    Parameters
    ----------
    exercise   : LoadedExercise instance
    pose_frame : np.ndarray shape (33, 3)
    is_in_rep  : bool — if True, skip form checking (person is mid-movement)

    Returns
    -------
    dict:
        feedback   (list[str])
        deviations (dict)  — {landmark_index: z_score}
    """
    if is_in_rep:
        # During a rep: person is mid-movement, not at rest.
        # Form stats are trained on rest position — don't compare.
        return {"feedback": [], "deviations": {}}

    if exercise.form_stats is None:
        return _untrained_result()

    stats       = exercise.form_stats
    key_indices = set(exercise.key_joint_indices)
    feedback    = []
    deviations  = {}

    for idx in range(33):
        if idx not in stats:
            continue

        z_score        = _z_score(pose_frame[idx], stats[idx])
        deviations[idx] = z_score

        if z_score > DEVIATION_THRESHOLD and idx in key_indices:
            messages = exercise.feedback_for_joint(idx)
            if messages:
                msg = _pick_message(messages, pose_frame[idx], stats[idx])
                if msg and msg not in feedback:
                    feedback.append(msg)

    return {
        "feedback":   feedback if feedback else ["Good posture — ready to go!"],
        "deviations": deviations,
    }


def get_color(score: int) -> tuple:
    """BGR skeleton color from a 0-100 form score."""
    if score >= 80: return (0, 255, 0)
    if score >= 50: return (0, 200, 255)
    return (0, 0, 255)


def _z_score(landmark: np.ndarray, stats: dict) -> float:
    """
    Compute z-score using x and y coordinates only.
    z (depth) is excluded — MediaPipe depth is unreliable across cameras.
    """
    mean = np.array(stats["mean"])[:2]   # x, y only
    std  = np.array(stats["std"])[:2]
    std  = np.where(std < 1e-6, 1e-6, std)
    return float(np.mean(np.abs((landmark[:2] - mean) / std)))


def _pick_message(messages: list, landmark: np.ndarray, stats: dict) -> str:
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0]

    mean = np.array(stats["mean"])[:2]
    std  = np.array(stats["std"])[:2]
    std  = np.where(std < 1e-6, 1e-6, std)

    worst_axis    = int(np.argmax(np.abs((landmark[:2] - mean) / std)))
    axis_keywords = {
        0: ["side", "lateral", "inward", "outward", "left", "right", "cave", "wide"],
        1: ["deep", "lower", "higher", "height", "low", "up", "down", "bend", "lift"],
    }
    for msg in messages:
        if any(kw in msg.lower() for kw in axis_keywords.get(worst_axis, [])):
            return msg
    return messages[0]


def _untrained_result() -> dict:
    return {
        "feedback":   ["Exercise not trained yet — form checking unavailable."],
        "deviations": {},
    }
