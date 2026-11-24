"""
qevd_trainer.py
---------------
Trains a multi-class mistake classifier for each exercise using:
  - QEVD mistake poses  (from qevd_ingestor.py)          → mistake classes
  - User's correct form poses (from existing builder)    → "correct" class

The result is a single SVM pipeline (mistake_classifier.pkl) that replaces
the old one-class SVM. At runtime, form_checker.py runs this classifier on
every live pose frame and returns the QEVD descriptive label for whatever
mistake it detects — or "Good form" if the pose looks correct.

Usage:
    # Train a single exercise:
    python qevd_trainer.py --exercise "squat" --data_dir exercises_data

    # Train all exercises that have QEVD data:
    python qevd_trainer.py --all --data_dir exercises_data

Output (per exercise):
    exercises_data/<exercise>/
        mistake_classifier.pkl   — sklearn Pipeline (scaler + multi-class SVM)
        config.json              — updated with mistake_types list
"""

import os
import json
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.svm        import SVC
from sklearn.pipeline   import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re as _re


# ── Junk class filter ─────────────────────────────────────────────────────────
# QEVD uses numeric labels for speed/angle/ROM parameter variations.
# These are not form mistakes — they are measurement annotations.
# Filtering them significantly improves accuracy for high knees and jumping jacks.

_JUNK_PATTERNS = [
    _re.compile(r'^speed=[0-9.]+'),
    _re.compile(r'^angle=[0-9.]+'),
    _re.compile(r'^arms_rom=[0-9.]+'),
    _re.compile(r'^legs_rom=[0-9.]+'),
    _re.compile(r'^rom=[0-9.]+'),
]

_KEEP_EXCEPTIONS = {
    "90 degrees",  # pushups: elbows at 90 degrees IS a real form mistake
}

def is_junk_class(key: str) -> bool:
    if key in _KEEP_EXCEPTIONS:
        return False
    return any(p.match(key) for p in _JUNK_PATTERNS)


# ── Feature extraction (must match form_checker.py exactly) ──────────────────

def frame_to_features(pose_frame: np.ndarray) -> np.ndarray:
    """
    Hip-relative, scale-normalised x+y positions of body landmarks 11-32.
    Returns shape (44,) — 22 landmarks × 2 axes.

    Invariant to:
      - camera distance (scale normalised by torso height)
      - horizontal flip (used consistently in both training and live feed)
    NOT invariant to:
      - large camera angle differences
      - person size vs frame size (handled by scale normalisation)
    """
    frame   = np.array(pose_frame).reshape(33, 3)
    hip_mid = (frame[23] + frame[24]) / 2.0
    sho_mid = (frame[11] + frame[12]) / 2.0
    scale   = np.linalg.norm(sho_mid - hip_mid)
    if scale < 1e-6:
        scale = 1e-6
    norm = (frame - hip_mid) / scale
    return norm[11:33, :2].flatten().astype(np.float32)


def pose_sequences_to_features(pose_sequences: list,
                                max_frames: int = None) -> np.ndarray:
    """
    Flatten a list of pose sequences into a feature matrix.
    Each sequence is a list of (33,3) frames.
    Returns shape (N_frames, 44).
    """
    rows = []
    for seq in pose_sequences:
        arr = np.array(seq).reshape(-1, 33, 3)
        for frame in arr:
            rows.append(frame_to_features(frame))
    if not rows:
        return np.empty((0, 44), dtype=np.float32)

    X = np.vstack(rows)

    if max_frames and len(X) > max_frames:
        idx = np.random.RandomState(42).choice(len(X), max_frames, replace=False)
        X   = X[idx]

    return X


# ── Training ──────────────────────────────────────────────────────────────────

def train_exercise(exercise_dir: str,
                   max_frames_per_class: int = 2000,
                   cv_folds: int = 3) -> bool:
    """
    Train and save a mistake classifier for one exercise.
    Returns True on success, False if data is missing.
    """
    exercise_name = os.path.basename(exercise_dir).replace("_", " ")
    print(f"\n{'='*60}")
    print(f"Training: {exercise_name}")
    print(f"{'='*60}")

    # ── Load QEVD mistake poses ───────────────────────────────────────────────
    mistake_poses_path  = os.path.join(exercise_dir, "qevd_mistake_poses.npy")
    mistake_labels_path = os.path.join(exercise_dir, "qevd_mistake_labels.json")

    if not os.path.exists(mistake_poses_path):
        print(f"  ❌ qevd_mistake_poses.npy not found — run qevd_ingestor.py first")
        return False

    mistake_poses = np.load(mistake_poses_path, allow_pickle=True).item()

    if not mistake_poses:
        print(f"  ❌ mistake_poses is empty")
        return False

    with open(mistake_labels_path) as f:
        mistake_labels = json.load(f)

    # Filter out junk classes (speed/angle/ROM parameter labels)
    junk = [k for k in mistake_poses if is_junk_class(k)]
    for k in junk:
        del mistake_poses[k]
    if junk:
        print(f"  Filtered {len(junk)} junk classes: {junk}")

    print(f"  Mistake types:  {len(mistake_poses)}")

    # ── Load correct form poses ───────────────────────────────────────────────
    # Look for poses.npy saved by the existing builder pipeline
    correct_poses_path = os.path.join(exercise_dir, "poses.npy")
    correct_poses = []

    if os.path.exists(correct_poses_path):
        raw = np.load(correct_poses_path, allow_pickle=True)
        correct_poses = list(raw)
        print(f"  Correct form:   {len(correct_poses)} clips (from poses.npy)")
    else:
        print(f"  ⚠️  No poses.npy found — 'correct' class will be absent.")
        print(f"       Record correct form using the builder, then re-run.")
        print(f"       Training will continue WITHOUT a correct class.")
        print(f"       (Feedback will still work — it just can't confirm good form.)")

    # ── Build feature matrix ──────────────────────────────────────────────────
    X_parts = []
    y_parts = []

    # Class 0 — correct form
    label_names  = ["correct"]
    label_map    = {"correct": 0}

    if correct_poses:
        X_correct = pose_sequences_to_features(correct_poses, max_frames_per_class)
        X_parts.append(X_correct)
        y_parts.extend([0] * len(X_correct))
        print(f"\n  Class 0 (correct):  {len(X_correct):5} frames")
    else:
        print(f"\n  Class 0 (correct):  —  (no data)")

    # Classes 1..N — QEVD mistakes
    for mistake_key, pose_seqs in sorted(mistake_poses.items()):
        class_id = len(label_map)
        label_map[mistake_key]  = class_id
        label_names.append(mistake_key)

        X_mistake = pose_sequences_to_features(pose_seqs, max_frames_per_class)
        if len(X_mistake) == 0:
            print(f"  Class {class_id} ('{mistake_key[:40]}'): 0 frames — SKIPPED")
            label_map.pop(mistake_key)
            label_names.pop()
            continue

        X_parts.append(X_mistake)
        y_parts.extend([class_id] * len(X_mistake))
        print(f"  Class {class_id:2} ('{mistake_key[:40]}'):  {len(X_mistake):5} frames")

    if not X_parts:
        print("  ❌ No features at all — aborting")
        return False

    X = np.vstack(X_parts)
    y = np.array(y_parts)

    class_dist = Counter(y)
    print(f"\n  Total frames: {len(X)}  |  Classes: {len(label_map)}")
    print(f"  Class distribution: { {label_names[k]: v for k, v in sorted(class_dist.items())} }")

    # ── Minimum data check ────────────────────────────────────────────────────
    min_class_size = min(class_dist.values())
    if min_class_size < cv_folds:
        print(f"  ⚠️  Smallest class has {min_class_size} frames — reducing CV folds to 2")
        cv_folds = 2

    # ── Train SVM pipeline ────────────────────────────────────────────────────
    print(f"\n  Training SVC (rbf, class_weight=balanced)...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)

    # ── Cross-validation ──────────────────────────────────────────────────────
    try:
        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  Cross-val accuracy ({cv_folds}-fold): "
              f"{scores.mean():.1%} ± {scores.std():.1%}")
    except Exception as e:
        print(f"  ⚠️  Cross-val skipped: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    bundle = {
        "pipeline":             pipeline,
        "label_map":            label_map,
        "label_names":          label_names,
        "mistake_descriptions": mistake_labels,   # {mistake_key: human text}
    }

    out_path = os.path.join(exercise_dir, "mistake_classifier.pkl")
    joblib.dump(bundle, out_path)
    print(f"\n  💾 Saved mistake_classifier.pkl")

    # Update config.json
    config_path = os.path.join(exercise_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["mistake_classifier_trained"] = True
        cfg["mistake_types"]              = label_names
        cfg["n_mistake_types"]            = len(label_names) - 1  # excl. correct
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  💾 Updated config.json")

    print(f"  ✅ Done — {exercise_name}")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train QEVD mistake classifiers")
    parser.add_argument("--exercise",  default=None,
                        help="Single exercise folder name under --data_dir")
    parser.add_argument("--all",       action="store_true",
                        help="Train all exercises that have qevd_mistake_poses.npy")
    parser.add_argument("--data_dir",  default="exercises_data")
    parser.add_argument("--max_frames_per_class", type=int, default=2000,
                        help="Max pose frames per class (default: 2000)")
    args = parser.parse_args()

    if not args.exercise and not args.all:
        parser.print_help()
        return

    if args.all:
        # Find all exercise folders with QEVD mistake data
        targets = []
        for name in sorted(os.listdir(args.data_dir)):
            d = os.path.join(args.data_dir, name)
            if os.path.isdir(d) and os.path.exists(
                    os.path.join(d, "qevd_mistake_poses.npy")):
                targets.append(d)

        if not targets:
            print(f"❌ No exercise folders with qevd_mistake_poses.npy found in {args.data_dir}")
            return

        print(f"Training {len(targets)} exercises: {[os.path.basename(t) for t in targets]}")

        success = 0
        for d in targets:
            if train_exercise(d, max_frames_per_class=args.max_frames_per_class):
                success += 1

        print(f"\n{'='*60}")
        print(f"✅ Trained {success}/{len(targets)} exercises successfully")

    else:
        exercise_dir = os.path.join(args.data_dir, args.exercise)
        if not os.path.isdir(exercise_dir):
            # Try safe name
            safe = args.exercise.replace(" ", "_").lower()
            exercise_dir = os.path.join(args.data_dir, safe)

        if not os.path.isdir(exercise_dir):
            print(f"❌ Exercise folder not found: {exercise_dir}")
            return

        train_exercise(exercise_dir, max_frames_per_class=args.max_frames_per_class)

    print(f"\nNext step:  python app.py")


if __name__ == "__main__":
    main()