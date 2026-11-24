"""
exercise_builder/pose_extractor.py
-----------------------------------
Extracts MediaPipe pose landmarks from every video in a folder.
Saves results to a .npy file for use by the form trainer.

CRITICAL FIX: Frames are horizontally flipped before MediaPipe processing
to match app.py which does cv2.flip(frame, 1) on the live webcam feed.
Without this, x-coordinates are mirrored between training and live, making
the form checker compare landmarks on the wrong side of the body.
Joint angles are invariant to horizontal flip so rep counting is unaffected.
"""

import cv2
import numpy as np
import os
import mediapipe as mp

mp_pose = mp.solutions.pose


def extract_poses_from_folder(
    folder_path: str,
    output_path: str,
    progress_callback=None
) -> dict:
    """
    Parameters
    ----------
    folder_path       : path to folder containing .mp4 / .avi / .mov videos
    output_path       : where to save the extracted poses .npy file
    progress_callback : optional function(current, total, filename)

    Returns
    -------
    dict with poses, total_frames, total_videos, failed_videos
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files      = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(video_extensions)
    ])

    if not video_files:
        raise ValueError(f"No video files found in {folder_path}")

    all_poses    = []
    failed       = []
    total_frames = 0

    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for i, filename in enumerate(video_files):
        video_path = os.path.join(folder_path, filename)

        if progress_callback:
            progress_callback(i, len(video_files), filename)

        try:
            frames = _extract_single_video(video_path, pose_estimator)
            if len(frames) >= 10:
                all_poses.append(np.array(frames))
                total_frames += len(frames)
            else:
                failed.append(f"{filename} (too short: {len(frames)} frames)")
        except Exception as e:
            failed.append(f"{filename} ({str(e)})")

    pose_estimator.close()

    if progress_callback:
        progress_callback(len(video_files), len(video_files), "Done")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(all_poses, dtype=object))

    return {
        "poses":         all_poses,
        "total_frames":  total_frames,
        "total_videos":  len(all_poses),
        "failed_videos": failed,
    }


def _extract_single_video(video_path: str, pose_estimator) -> list:
    """
    Extract pose landmarks from a single video.
    Frames are flipped to match the live webcam feed in app.py.
    """
    cap    = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Match live feed: app.py does cv2.flip(frame, 1) before MediaPipe
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_estimator.process(rgb)

        if res.pose_landmarks:
            row = []
            for lm in res.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            frames.append(row)

    cap.release()

    if not frames:
        return []

    return np.array(frames).reshape(-1, 33, 3).tolist()


def load_poses(path: str) -> list:
    data = np.load(path, allow_pickle=True)
    return list(data)
