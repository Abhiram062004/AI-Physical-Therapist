"""
qevd_ingestor.py
----------------
Reads fine_grained_labels.json, finds all videos across QEVD Parts 1-4,
extracts MediaPipe pose landmarks, validates quality, and saves mistake
pose datasets per exercise into exercises_data/.

CRASH RECOVERY / RESUME
------------------------
Progress is saved after EVERY mistake type completes — not at the end.
Each mistake type is checkpointed individually to:
    exercises_data/<exercise>/checkpoints/<mistake_key>.npy
    exercises_data/<exercise>/checkpoints/<mistake_key>_meta.json

If the script crashes or is interrupted, re-run the exact same command.
Already-completed mistake types are detected and skipped automatically.
Only the remaining ones are processed.

QUALITY VALIDATION (5 checks per clip)
---------------------------------------
1. Pose detection rate  >= 0.70  (MediaPipe must find pose in 70%+ of frames)
2. Landmark visibility  >= 0.60  (body landmarks must be clearly visible)
3. Min detected frames  >= 5     (clip must be long enough)
4. Body size            >= 0.10  (person must fill enough of the frame)
5. Pose jitter          <= 0.15  (tracking must be stable)

Usage:
    python qevd_ingestor.py \
        --qevd_root D:/QEVD \
        --exercises "squats" "pushups" "glutes bridge" "high knees" \
                    "jumping jacks" "alternating forward lunges" \
                    "kickback (right leg)" \
        --max_per_mistake 40
"""

import os
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict


mp_pose = mp.solutions.pose

# ── Quality thresholds ─────────────────────────────────────────────────────────
MIN_DETECTION_RATE = 0.70
MIN_VISIBILITY     = 0.60
MIN_FRAMES         = 5
MIN_TORSO_FRACTION = 0.10
MAX_JITTER         = 0.15


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_label(label):
    if " - " in label:
        parts = label.split(" - ", 1)
        return parts[0].strip().lower(), parts[1].strip()
    return label.strip().lower(), None


def safe_name(name):
    return (name.replace(" ", "_").replace("/", "_")
               .replace("(", "").replace(")", "").lower())


def find_video(filename, part_dirs):
    fname = os.path.basename(filename)
    for part_dir in part_dirs:
        path = os.path.join(part_dir, fname)
        if os.path.exists(path):
            return path
    return None


# ── Pose extraction + quality validation ───────────────────────────────────────

def extract_and_validate(video_path, pose_estimator,
                         min_detection_rate=MIN_DETECTION_RATE,
                         min_visibility=MIN_VISIBILITY,
                         min_frames=MIN_FRAMES,
                         min_torso_fraction=MIN_TORSO_FRACTION,
                         max_jitter=MAX_JITTER):
    cap          = cv2.VideoCapture(video_path)
    total_frames = 0
    pose_frames  = []
    vis_scores   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose_estimator.process(rgb)

        if res.pose_landmarks:
            row = []
            vis = []
            for lm in res.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
                vis.append(lm.visibility)
            pose_frames.append(row)
            vis_scores.append(float(np.mean(vis[11:33])))

    cap.release()

    quality = {
        "total_video_frames": total_frames,
        "detected_frames":    len(pose_frames),
        "detection_rate":     len(pose_frames) / max(total_frames, 1),
        "mean_visibility":    float(np.mean(vis_scores)) if vis_scores else 0.0,
        "passed":             False,
        "fail_reason":        None,
    }

    if len(pose_frames) < min_frames:
        quality["fail_reason"] = f"too_few_frames ({len(pose_frames)} < {min_frames})"
        return [], quality

    if quality["detection_rate"] < min_detection_rate:
        quality["fail_reason"] = (
            f"low_detection_rate ({quality['detection_rate']:.2f} < {min_detection_rate})"
        )
        return [], quality

    if quality["mean_visibility"] < min_visibility:
        quality["fail_reason"] = (
            f"low_visibility ({quality['mean_visibility']:.2f} < {min_visibility})"
        )
        return [], quality

    arr = np.array(pose_frames).reshape(-1, 33, 3)

    sho_y = (arr[:, 11, 1] + arr[:, 12, 1]) / 2.0
    hip_y = (arr[:, 23, 1] + arr[:, 24, 1]) / 2.0
    mean_torso = float(np.mean(np.abs(hip_y - sho_y)))
    quality["mean_torso_fraction"] = mean_torso

    if mean_torso < min_torso_fraction:
        quality["fail_reason"] = (
            f"person_too_small (torso={mean_torso:.3f} < {min_torso_fraction})"
        )
        return [], quality

    if len(arr) >= 2:
        diffs  = np.diff(arr[:, 11:33, :2], axis=0)
        jitter = float(np.mean(np.linalg.norm(diffs, axis=2)))
        quality["mean_jitter"] = jitter
        if jitter > max_jitter:
            quality["fail_reason"] = f"high_jitter ({jitter:.3f} > {max_jitter})"
            return [], quality
    else:
        quality["mean_jitter"] = 0.0

    quality["passed"] = True
    return arr.tolist(), quality


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def get_ckpt_paths(ckpt_dir, mistake_key):
    safe = safe_name(mistake_key)
    return (
        os.path.join(ckpt_dir, f"{safe}.npy"),
        os.path.join(ckpt_dir, f"{safe}_meta.json"),
    )


def save_checkpoint(ckpt_dir, mistake_key, pose_seqs, label, split_info, quality_info):
    npy_path, meta_path = get_ckpt_paths(ckpt_dir, mistake_key)
    np.save(npy_path, np.array(pose_seqs, dtype=object))
    with open(meta_path, "w") as f:
        json.dump({
            "label":        label,
            "split_counts": split_info,
            "quality":      quality_info,
        }, f, indent=2)


def load_checkpoint(ckpt_dir, mistake_key):
    npy_path, meta_path = get_ckpt_paths(ckpt_dir, mistake_key)
    if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
        return None
    try:
        seqs = np.load(npy_path, allow_pickle=True).tolist()
        with open(meta_path) as f:
            meta = json.load(f)
        return seqs, meta
    except Exception:
        return None


def merge_checkpoints(out_dir, ckpt_dir, mistake_poses, mistake_labels,
                      split_counts, quality_report):
    """Write the final merged files from all checkpoints."""
    np.save(
        os.path.join(out_dir, "qevd_mistake_poses.npy"),
        np.array(mistake_poses, dtype=object)
    )
    with open(os.path.join(out_dir, "qevd_mistake_labels.json"), "w") as f:
        json.dump(mistake_labels, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "qevd_split_counts.json"), "w") as f:
        json.dump(split_counts, f, indent=2)
    with open(os.path.join(out_dir, "qevd_quality_report.json"), "w") as f:
        json.dump(quality_report, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qevd_root",           required=True)
    parser.add_argument("--exercises",            nargs="+", required=True)
    parser.add_argument("--output",               default="exercises_data")
    parser.add_argument("--split",                default="train")
    parser.add_argument("--max_per_mistake",      type=int,   default=50)
    parser.add_argument("--min_detection_rate",   type=float, default=MIN_DETECTION_RATE)
    parser.add_argument("--min_visibility",       type=float, default=MIN_VISIBILITY)
    parser.add_argument("--min_frames",           type=int,   default=MIN_FRAMES)
    parser.add_argument("--min_torso_fraction",   type=float, default=MIN_TORSO_FRACTION)
    parser.add_argument("--max_jitter",           type=float, default=MAX_JITTER)
    args = parser.parse_args()

    print(f"Quality thresholds:")
    print(f"  Detection rate  >= {args.min_detection_rate}")
    print(f"  Visibility      >= {args.min_visibility}")
    print(f"  Min frames      >= {args.min_frames}")
    print(f"  Torso fraction  >= {args.min_torso_fraction}")
    print(f"  Max jitter      <= {args.max_jitter}\n")

    # ── Find part directories ─────────────────────────────────────────────────
    part_dirs = []
    for i in range(1, 5):
        d = os.path.join(args.qevd_root, f"QEVD-FIT-300k-Part-{i}")
        if os.path.isdir(d):
            part_dirs.append(d)
        else:
            print(f"⚠️  Part {i} not found at {d}")

    if not part_dirs:
        print("❌ No QEVD part directories found.")
        return

    print(f"✅ Found {len(part_dirs)} part directories")

    # ── Load annotations ──────────────────────────────────────────────────────
    json_path = os.path.join(
        args.qevd_root, "QEVD-FIT-300k-Part-4", "fine_grained_labels.json"
    )
    with open(json_path) as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} annotations\n")

    # ── Build lookup ──────────────────────────────────────────────────────────
    target_exercises      = {e.lower(): e for e in args.exercises}
    exercise_clips        = defaultdict(lambda: defaultdict(list))
    exercise_descriptions = defaultdict(dict)

    for entry in data:
        if entry.get("split") != args.split:
            continue
        if not entry.get("labels"):
            continue

        label       = entry["labels"][0]
        descriptive = (entry["labels_descriptive"][0]
                       if entry.get("labels_descriptive") else label)

        exercise_raw, mistake = parse_label(label)
        if exercise_raw not in target_exercises or mistake is None:
            continue

        display_name = target_exercises[exercise_raw]
        desc_mistake = (descriptive.split(" - ", 1)[1].strip()
                        if " - " in descriptive else descriptive.strip())

        exercise_clips[display_name][mistake].append({
            "video_path":  entry["video_path"],
            "descriptive": desc_mistake,
        })
        if mistake not in exercise_descriptions[display_name]:
            exercise_descriptions[display_name][mistake] = desc_mistake

    if not exercise_clips:
        print(f"❌ None of the requested exercises found in split='{args.split}'.")
        return

    print(f"Exercises matched: {list(exercise_clips.keys())}")

    # ── Extract + validate ────────────────────────────────────────────────────
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for exercise_name, mistakes in exercise_clips.items():
        print(f"\n{'='*60}")
        print(f"📦 {exercise_name}  ({len(mistakes)} mistake types)")
        print(f"{'='*60}")

        out_dir  = os.path.join(args.output, safe_name(exercise_name))
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(out_dir,  exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        mistake_poses  = {}
        mistake_labels = {}
        split_counts   = {}
        quality_report = {}

        # ── Resume: reload completed checkpoints ──────────────────────────────
        skipped = []
        for mk in sorted(mistakes.keys()):
            result = load_checkpoint(ckpt_dir, mk)
            if result is not None:
                seqs, meta = result
                mistake_poses[mk]  = seqs
                mistake_labels[mk] = meta.get("label", mk)
                split_counts[mk]   = meta.get("split_counts", {})
                quality_report[mk] = meta.get("quality", {})
                skipped.append(mk)

        if skipped:
            print(f"\n  ♻️  Resuming — {len(skipped)}/{len(mistakes)} already done:")
            for mk in skipped:
                n = len(mistake_poses[mk])
                print(f"       ✅ {mk}  ({n} clips)")

        # ── Process remaining mistake types ───────────────────────────────────
        remaining = [mk for mk in sorted(mistakes.keys()) if mk not in skipped]
        print(f"\n  ▶️  Processing {len(remaining)} remaining mistake types...")

        for mistake_key in remaining:
            clips        = mistakes[mistake_key]
            clips_to_use = clips[:args.max_per_mistake]
            n_attempted  = len(clips_to_use)

            print(f"\n  [{mistake_key}]  ({n_attempted} clips)")

            pose_seqs    = []
            fail_reasons = defaultdict(int)
            not_found    = 0
            det_rates    = []
            vis_scores   = []

            for i, clip in enumerate(clips_to_use):
                video_path = find_video(clip["video_path"], part_dirs)

                if video_path is None:
                    not_found += 1
                    fail_reasons["video_not_found"] += 1
                    continue

                frames, quality = extract_and_validate(
                    video_path, pose_estimator,
                    min_detection_rate = args.min_detection_rate,
                    min_visibility     = args.min_visibility,
                    min_frames         = args.min_frames,
                    min_torso_fraction = args.min_torso_fraction,
                    max_jitter         = args.max_jitter,
                )

                if quality.get("detection_rate") is not None:
                    det_rates.append(quality["detection_rate"])
                if quality.get("mean_visibility") is not None:
                    vis_scores.append(quality["mean_visibility"])

                if frames:
                    pose_seqs.append(frames)
                else:
                    reason = quality.get("fail_reason", "unknown")
                    bucket = reason.split("(")[0].strip() if "(" in reason else reason
                    fail_reasons[bucket] += 1

                if (i + 1) % 10 == 0 or (i + 1) == n_attempted:
                    processed = i + 1 - not_found
                    pct = int(len(pose_seqs) / max(processed, 1) * 100)
                    print(
                        f"    [{i+1}/{n_attempted}]  "
                        f"✅ {len(pose_seqs)} passed  "
                        f"❌ {sum(fail_reasons.values())} rejected  "
                        f"({pct}%)",
                        end="\r"
                    )

            print()

            # ── Quality summary ───────────────────────────────────────────────
            processed     = n_attempted - not_found
            pass_rate_pct = int(len(pose_seqs) / max(processed, 1) * 100)

            print(f"    Pass rate: {pass_rate_pct}%  ({len(pose_seqs)}/{processed})")

            if fail_reasons:
                print(f"    Rejections:")
                for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                    print(f"      {count:3}x  {reason}")
            if det_rates:
                print(f"    Avg detection: {np.mean(det_rates):.2f}  "
                      f"Avg visibility: {np.mean(vis_scores):.2f}")
            if pass_rate_pct < 30 and processed >= 10:
                print(f"    ⚠️  Low pass rate — try: "
                      f"--min_detection_rate 0.5 --min_visibility 0.5")

            # ── Build metadata ────────────────────────────────────────────────
            q_info = {
                "attempted":      n_attempted,
                "not_found":      not_found,
                "passed":         len(pose_seqs),
                "failed":         sum(fail_reasons.values()),
                "pass_rate_pct":  pass_rate_pct,
                "fail_reasons":   dict(fail_reasons),
                "avg_detection":  float(np.mean(det_rates))  if det_rates  else None,
                "avg_visibility": float(np.mean(vis_scores)) if vis_scores else None,
            }
            s_info = {
                "attempted": n_attempted,
                "success":   len(pose_seqs),
                "failed":    sum(fail_reasons.values()),
            }
            label_text = exercise_descriptions[exercise_name].get(
                mistake_key, mistake_key
            )

            if pose_seqs:
                # ── Save checkpoint immediately ───────────────────────────────
                save_checkpoint(ckpt_dir, mistake_key,
                                pose_seqs, label_text, s_info, q_info)

                mistake_poses[mistake_key]  = pose_seqs
                mistake_labels[mistake_key] = label_text
                split_counts[mistake_key]   = s_info
                quality_report[mistake_key] = q_info

                print(f"    💾 Checkpoint saved  ({len(pose_seqs)} clips)")
            else:
                print(f"    ⚠️  No valid clips — skipping")
                quality_report[mistake_key] = q_info

        # ── Merge all checkpoints into final files ────────────────────────────
        if mistake_poses:
            merge_checkpoints(out_dir, ckpt_dir,
                              mistake_poses, mistake_labels,
                              split_counts, quality_report)

            total_clips   = sum(v.get("success", 0) for v in split_counts.values())
            total_attempt = sum(v.get("attempted", 0) for v in split_counts.values())
            overall_pass  = int(total_clips / max(total_attempt, 1) * 100)

            print(f"\n  💾 Final save: {len(mistake_poses)} mistake types  "
                  f"| {total_clips} clips  "
                  f"| {overall_pass}% pass rate")
            print(f"     → {out_dir}")
        else:
            print(f"\n  ❌ No valid clips for {exercise_name}")

    pose_estimator.close()

    print(f"\n{'='*60}")
    print(f"✅ Ingest complete!")
    print(f"\nNext:")
    print(f"  python qevd_trainer.py --all --data_dir {args.output}")


if __name__ == "__main__":
    main()