"""
exercise_builder/form_trainer.py
---------------------------------
Trains the form model from extracted correct-form poses.

REST POSITION STRATEGY
----------------------
We don't look for a rest period in the video — there isn't one.
Instead: the annotator marks rep START when the person is at full rest.
So frame 0 of every annotated rep window IS the rest position, guaranteed.

We take the first FIRST_FRAMES_N frames of each annotation and repeat them
(stretch) to build a statistically stable pool of rest-position samples.
This is completely honest — we're just oversampling a known-good rest frame
rather than averaging frames that might be mid-movement.

Saves:
  form_stats.npy  — per-joint statistics (from true rest frames)
  svm_model.pkl   — trained One-Class SVM (from true rest frames)
"""

import numpy as np
import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# How many frames from the START of each annotated rep to use as rest.
# Frame 0 = annotator-confirmed rest position.
# Taking 3 frames handles the case where frame 0 is slightly early.
FIRST_FRAMES_N = 3

# How many times to repeat those frames so stats are stable.
# 40 annotations × 3 frames × 15 repeats = 1800 rest samples — plenty.
REPEAT_COUNT = 15


def train_form_model(
    poses: list,
    output_dir: str,
    key_joint_indices: list,
    annotations: list = None,
    video_poses: list = None,
    progress_callback=None
) -> dict:
    """
    Parameters
    ----------
    poses             : list of np.ndarray, each shape (T, 33, 3)
    output_dir        : where to save form_stats.npy and svm_model.pkl
    key_joint_indices : list of landmark indices marked as key joints
    annotations       : list of {video_index, start_frame, end_frame}
    video_poses       : list of per-video pose lists (np.ndarray or None per frame)
    progress_callback : optional function(step: str)
    """
    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback("Extracting rest-position frames...")

    rest_frames = []

    # ── Primary: first frames of each annotated rep ───────────────────────────
    if annotations and video_poses:
        for ann in annotations:
            vid_idx     = ann.get("video_index", -1)
            start_frame = ann.get("start_frame", 0)
            end_frame   = ann.get("end_frame", 0)

            if vid_idx < 0 or vid_idx >= len(video_poses):
                continue
            raw = video_poses[vid_idx]
            if not raw:
                continue

            # Collect first FIRST_FRAMES_N valid pose frames from this annotation.
            # These are the closest frames to the rest position we will ever have.
            collected = []
            for i in range(start_frame, min(start_frame + FIRST_FRAMES_N + 5, end_frame)):
                if i < len(raw) and raw[i] is not None:
                    collected.append(raw[i])
                if len(collected) >= FIRST_FRAMES_N:
                    break

            # Stretch: repeat each captured frame so stats are numerically stable
            for frame in collected:
                for _ in range(REPEAT_COUNT):
                    rest_frames.append(frame)

        print(f"  Rest frames from annotation starts: {len(rest_frames)} "
              f"({len(annotations)} annotations × ~{FIRST_FRAMES_N} frames × {REPEAT_COUNT} repeats)")

    # ── Fallback: if no annotations, use first frame of each training video ───
    all_frames_list = []
    for pose_seq in poses:
        arr = np.array(pose_seq).reshape(-1, 33, 3)
        all_frames_list.append(arr)

        if not rest_frames:
            # Stretch the first few frames of each video
            for frame in arr[:FIRST_FRAMES_N]:
                for _ in range(REPEAT_COUNT):
                    rest_frames.append(frame)

    if not rest_frames:
        print("  ⚠️  No rest frames at all — using all frames as fallback")
        rest_frames = [f for arr in all_frames_list for f in arr]

    rest_all = np.array(rest_frames).reshape(-1, 33, 3)   # (N_rest, 33, 3)
    all_all  = np.vstack(all_frames_list)                  # (N_all,  33, 3)

    print(f"  Total rest pool: {len(rest_all)}  |  Total training frames: {len(all_all)}")

    # ── Step 1: per-joint statistics from rest frames ─────────────────────────
    if progress_callback:
        progress_callback("Computing per-joint statistics (rest position)...")

    form_stats = {}
    for idx in range(33):
        joint_data = rest_all[:, idx, :]
        form_stats[idx] = {
            "mean":         joint_data.mean(axis=0).tolist(),
            "std":          joint_data.std(axis=0).tolist(),
            "is_key_joint": idx in key_joint_indices,
        }

    stats_path = os.path.join(output_dir, "form_stats.npy")
    np.save(stats_path, form_stats)
    print(f"  ✅ Saved form_stats.npy")

    # ── Step 2: One-Class SVM from rest frames ────────────────────────────────
    if progress_callback:
        progress_callback("Training One-Class SVM...")

    body_frames = rest_all[:, 11:33, :].reshape(len(rest_all), -1)

    MAX_SAMPLES = 8000
    if len(body_frames) > MAX_SAMPLES:
        indices     = np.random.choice(len(body_frames), MAX_SAMPLES, replace=False)
        body_frames = body_frames[indices]

    scaler      = StandardScaler()
    body_scaled = scaler.fit_transform(body_frames)

    svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    svm.fit(body_scaled)

    svm_bundle = {"svm": svm, "scaler": scaler}
    svm_path   = os.path.join(output_dir, "svm_model.pkl")
    joblib.dump(svm_bundle, svm_path)
    print(f"  ✅ Saved svm_model.pkl")

    if progress_callback:
        progress_callback("Done!")

    return {
        "total_frames":  len(all_all),
        "rest_frames":   len(rest_all),
        "total_videos":  len(poses),
        "key_joints":    key_joint_indices,
        "stats_path":    stats_path,
        "svm_path":      svm_path,
    }
