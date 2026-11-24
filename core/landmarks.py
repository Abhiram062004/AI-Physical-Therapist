"""
core/landmarks.py
-----------------
Single source of truth for MediaPipe landmark indices and names.
Used by the builder UI (joint selector) and the form checker.

Derived joints (index 100+) are computed at runtime from real landmarks
and are not directly from MediaPipe.
"""

# Full MediaPipe Pose landmark map
LANDMARK_NAMES = {
    0:  "Nose",
    1:  "Left Eye Inner",
    2:  "Left Eye",
    3:  "Left Eye Outer",
    4:  "Right Eye Inner",
    5:  "Right Eye",
    6:  "Right Eye Outer",
    7:  "Left Ear",
    8:  "Right Ear",
    9:  "Mouth Left",
    10: "Mouth Right",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    17: "Left Pinky",
    18: "Right Pinky",
    19: "Left Index",
    20: "Right Index",
    21: "Left Thumb",
    22: "Right Thumb",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle",
    29: "Left Heel",
    30: "Right Heel",
    31: "Left Foot Index",
    32: "Right Foot Index",
}

# Derived joints — computed from combinations of real landmarks
# Index 100+ to avoid collision with MediaPipe indices
DERIVED_JOINTS = {
    100: {
        "name":        "Torso Lean",
        "description": "Forward/sideways lean — midpoint of shoulders vs midpoint of hips",
        "landmarks":   [11, 12, 23, 24],
    },
    101: {
        "name":        "Head Position",
        "description": "Forward head lean — nose vs midpoint of shoulders",
        "landmarks":   [0, 11, 12],
    },
    102: {
        "name":        "Spine Alignment",
        "description": "Overall spinal straightness — nose, shoulder midpoint, hip midpoint",
        "landmarks":   [0, 11, 12, 23, 24],
    },
    103: {
        "name":        "Hip Level",
        "description": "Whether hips are even — left hip vs right hip height",
        "landmarks":   [23, 24],
    },
    104: {
        "name":        "Shoulder Level",
        "description": "Whether shoulders are even — left vs right shoulder height",
        "landmarks":   [11, 12],
    },
}

# Joints grouped by body region for the dropdown
JOINT_GROUPS = {
    "Head & Neck":   [0],
    "Shoulders":     [11, 12],
    "Elbows":        [13, 14],
    "Wrists":        [15, 16],
    "Hips":          [23, 24],
    "Knees":         [25, 26],
    "Ankles":        [27, 28],
    "Heels":         [29, 30],
    "Feet":          [31, 32],
    "Derived":       list(DERIVED_JOINTS.keys()),
}

# Suggested joints per exercise — used for the suggestion banner.
# Keys are canonical names; lookup is case-insensitive (see get_suggestions()).
EXERCISE_JOINT_SUGGESTIONS = {
    # Lower body ──────────────────────────────────────────────────────────────
    "Squats": [
        25, 26,   # L/R Knee   — primary movement, depth tracking
        23, 24,   # L/R Hip    — hip hinge & depth
        100,      # Torso Lean — keep chest up, don't fold forward
        103,      # Hip Level  — catch uneven squat (one hip dropping)
    ],
    "Lunges": [
        25, 26,   # L/R Knee   — front knee angle & alignment
        23, 24,   # L/R Hip    — hip drop check
        27, 28,   # L/R Ankle  — front foot stability
        100,      # Torso Lean — upright torso
        103,      # Hip Level  — pelvis stays level during split stance
    ],
    "Glute Bridges": [
        23, 24,   # L/R Hip    — primary — hip extension tracked here
        25, 26,   # L/R Knee   — 90° knee angle at start position
        100,      # Torso Lean — neutral spine
        103,      # Hip Level  — both sides rise equally (common error)
    ],
    "Glute Kickbacks": [
        23, 24,   # L/R Hip    — hip extension of kicking leg
        25, 26,   # L/R Knee   — knee angle of kicking leg
        27, 28,   # L/R Ankle  — foot position check
        103,      # Hip Level  — hips stay square, don't rotate open
    ],
    "High Knees": [
        25, 26,   # L/R Knee   — knee lift height
        23, 24,   # L/R Hip    — hip flexion drive
        100,      # Torso Lean — stay upright, don't lean back
        101,      # Head Pos   — head neutral (common to crane neck forward)
    ],
    # Upper body ──────────────────────────────────────────────────────────────
    "Push-ups": [
        13, 14,   # L/R Elbow  — primary — elbow angle tracks the push
        11, 12,   # L/R Shoulder — shoulder position, flare check
        23, 24,   # L/R Hip    — plank position, hips not sagging/piking
        100,      # Torso Lean — full-body alignment
        104,      # Shoulder Level — both sides working equally
    ],
    "Shoulder Press": [
        11, 12,   # L/R Shoulder — primary — shoulder angle tracks the press
        13, 14,   # L/R Elbow  — elbow position at start (90°)
        15, 16,   # L/R Wrist  — wrist stays neutral, not bent
        100,      # Torso Lean — no excessive back arch at top
        104,      # Shoulder Level — both arms rising evenly
    ],
    "Bicep Curls": [
        13, 14,   # L/R Elbow  — primary — full curl range
        11, 12,   # L/R Shoulder — shoulder must stay still (no swinging)
        15, 16,   # L/R Wrist  — wrist neutral, not curling wrist to cheat
    ],
    "Lateral Raises": [
        11, 12,   # L/R Shoulder — primary — abduction angle
        13, 14,   # L/R Elbow  — slight elbow bend maintained throughout
        15, 16,   # L/R Wrist  — wrists level with elbows at top
        104,      # Shoulder Level — both arms rise to the same height
    ],
    # Full body ───────────────────────────────────────────────────────────────
    "Jumping Jacks": [
        11, 12,   # L/R Shoulder — arm abduction tracked here
        15, 16,   # L/R Wrist  — arms fully extended overhead
        27, 28,   # L/R Ankle  — feet land wide symmetrically
        104,      # Shoulder Level — both arms moving together
        103,      # Hip Level  — pelvis stays level on landing
    ],
}

# Recommended tracking joint per exercise — the joint whose angle most clearly
# marks one full rep (goes low → high or high → low once per rep).
# Shown in the builder so the user knows what to select in the annotator.
EXERCISE_TRACKING_JOINTS = {
    "Squats":          (25, "Left Knee — angle drops on the way down, rises on the way up"),
    "Lunges":          (25, "Left Knee — front knee angle defines rep depth"),
    "Glute Bridges":   (23, "Left Hip — hip extension angle peaks at the top of each rep"),
    "Glute Kickbacks": (25, "Left Knee — kick-leg knee angle extends on each rep"),
    "High Knees":      (25, "Left Knee — knee rises above threshold once per stride"),
    "Push-ups":        (13, "Left Elbow — flexes down and extends up each rep"),
    "Shoulder Press":  (11, "Left Shoulder — shoulder angle opens from 90° to full extension"),
    "Bicep Curls":     (13, "Left Elbow — full curl = elbow fully flexed, rest = fully extended"),
    "Lateral Raises":  (11, "Left Shoulder — abduction angle goes from 0° to ~90° each rep"),
    "Jumping Jacks":   (11, "Left Shoulder — arm sweeps out and back once per jack"),
}


def get_suggestions(exercise_name: str) -> list:
    """
    Case-insensitive, punctuation-tolerant lookup for joint suggestions.
    Returns [] if no match found.
    """
    name_clean = exercise_name.strip().lower().replace("-", " ").replace("_", " ")
    for key, joints in EXERCISE_JOINT_SUGGESTIONS.items():
        if key.lower().replace("-", " ") == name_clean:
            return joints
    # Partial match fallback — handles "squat" matching "Squats" etc.
    for key, joints in EXERCISE_JOINT_SUGGESTIONS.items():
        if name_clean in key.lower() or key.lower().rstrip("s") in name_clean:
            return joints
    return []


def get_tracking_suggestion(exercise_name: str):
    """
    Returns (landmark_index, explanation_str) or None.
    Case-insensitive lookup.
    """
    name_clean = exercise_name.strip().lower().replace("-", " ").replace("_", " ")
    for key, val in EXERCISE_TRACKING_JOINTS.items():
        if key.lower().replace("-", " ") == name_clean:
            return val
    for key, val in EXERCISE_TRACKING_JOINTS.items():
        if name_clean in key.lower() or key.lower().rstrip("s") in name_clean:
            return val
    return None

# Angle triplets — (point_a_idx, vertex_idx, point_c_idx)
ANGLE_TRIPLETS = {
    11: (13, 11, 23),   # L Shoulder
    12: (14, 12, 24),   # R Shoulder
    13: (11, 13, 15),   # L Elbow
    14: (12, 14, 16),   # R Elbow
    15: (13, 15, 19),   # L Wrist
    16: (14, 16, 20),   # R Wrist
    23: (11, 23, 25),   # L Hip
    24: (12, 24, 26),   # R Hip
    25: (23, 25, 27),   # L Knee
    26: (24, 26, 28),   # R Knee
    27: (25, 27, 31),   # L Ankle
    28: (26, 28, 32),   # R Ankle
}

# Bilateral pairs — maps each joint to its mirror joint on the other side.
# Used by rep_trainer and rep_counter to average both sides for bilateral exercises,
# reducing noise from minor asymmetries.
BILATERAL_PAIRS = {
    11: 12, 12: 11,   # shoulders
    13: 14, 14: 13,   # elbows
    15: 16, 16: 15,   # wrists
    23: 24, 24: 23,   # hips
    25: 26, 26: 25,   # knees
    27: 28, 28: 27,   # ankles
}


def compute_angle(a_idx: int, b_idx: int, c_idx: int, pose_frame) -> float:
    """
    Compute the angle in degrees at vertex b_idx,
    formed by landmarks a_idx → b_idx → c_idx.
    pose_frame : np.ndarray shape (33, 3)

    Uses x and y coordinates ONLY — z (depth) is excluded.
    MediaPipe depth is estimated from image proportions, not real geometry.
    It is noisy and inconsistent across cameras and distances. Including it
    distorts angle calculations significantly — a fully extended arm that
    should read ~165° can compute as ~120° purely due to bad z values.
    x+y (2D projected angle) is stable, consistent, and sufficient for
    detecting joint flexion/extension in all exercises.
    """
    import numpy as np
    a  = pose_frame[a_idx, :2]   # x, y only
    b  = pose_frame[b_idx, :2]
    c  = pose_frame[c_idx, :2]
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def get_name(idx: int) -> str:
    if idx in DERIVED_JOINTS:
        return DERIVED_JOINTS[idx]["name"]
    return LANDMARK_NAMES.get(idx, f"Landmark {idx}")


def get_description(idx: int) -> str:
    if idx in DERIVED_JOINTS:
        return DERIVED_JOINTS[idx]["description"]
    return ""


def is_derived(idx: int) -> bool:
    return idx in DERIVED_JOINTS


def get_body_landmarks() -> list:
    return list(range(11, 33))


def get_all_selectable() -> dict:
    result = {}
    for group, indices in JOINT_GROUPS.items():
        entries = []
        for idx in indices:
            name = get_name(idx)
            desc = get_description(idx)
            entries.append((idx, name, desc))
        result[group] = entries
    return result


def compute_derived(idx: int, pose_frame) -> float:
    import numpy as np

    if idx == 100:  # Torso Lean
        sho_mid = (pose_frame[11, :2] + pose_frame[12, :2]) / 2
        hip_mid = (pose_frame[23, :2] + pose_frame[24, :2]) / 2
        return float(np.linalg.norm(sho_mid - hip_mid))

    elif idx == 101:  # Head Position
        sho_mid = (pose_frame[11, :2] + pose_frame[12, :2]) / 2
        nose    = pose_frame[0, :2]
        return float(abs(nose[0] - sho_mid[0]))

    elif idx == 102:  # Spine Alignment
        nose    = pose_frame[0, :2]
        sho_mid = (pose_frame[11, :2] + pose_frame[12, :2]) / 2
        hip_mid = (pose_frame[23, :2] + pose_frame[24, :2]) / 2
        v1    = sho_mid - nose
        v2    = hip_mid - sho_mid
        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
        return float(cross)

    elif idx == 103:  # Hip Level
        return float(abs(pose_frame[23, 1] - pose_frame[24, 1]))

    elif idx == 104:  # Shoulder Level
        return float(abs(pose_frame[11, 1] - pose_frame[12, 1]))

    return 0.0
