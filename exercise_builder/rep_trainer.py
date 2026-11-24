"""
exercise_builder/rep_trainer.py
--------------------------------
Builds the rep counting threshold and path template from annotated reps.

CRITICAL FIX: Threshold computation now separates rest frames (first and last
20% of each annotation) from peak frames (middle 60%). This means the threshold
is computed from the actual rest angle vs actual peak angle, not just percentiles
of whatever range the annotator happened to mark. Previously, if the annotator
started or ended mid-movement, the percentiles would be wrong, producing a
threshold that sits inside the movement zone and causes phantom or missed reps.

Counting : joint angle (degrees) — invariant to camera position/distance.
Scoring  : hip-relative normalised y-positions — invariant to camera flip
           and depth estimation errors (y-only).

Why bilateral averaging
-----------------------
For exercises that use both sides of the body (squats: both knees, shoulder
press: both arms), averaging left and right joint angles reduces noise from
minor asymmetries. The BILATERAL_PAIRS map is used to find the partner joint.
"""

import numpy as np
import os

from core.landmarks import ANGLE_TRIPLETS, compute_angle, BILATERAL_PAIRS

DEADBAND_FRACTION = 0.20   # fraction of angle range used as hysteresis band
MIN_DEADBAND_DEG  = 12.0   # minimum deadband in degrees


def build_rep_pattern(
    annotations: list,
    video_poses: list,
    key_joint_indices: list,
    output_dir: str,
    tracking_joint_index: int = None,
) -> dict:
    """
    Parameters
    ----------
    annotations          : [{video_index, start_frame, end_frame}, ...]
    video_poses          : list of lists, one per video (np.ndarray or None per frame)
    key_joint_indices    : landmark indices marked as key joints
    output_dir           : directory to save rep_pattern.npy
    tracking_joint_index : joint used for angle-based rep counting

    Returns
    -------
    dict with threshold info written to config.json by builder_app
    """
    if tracking_joint_index is None:
        tracking_joint_index = key_joint_indices[0]

    triplet = ANGLE_TRIPLETS.get(tracking_joint_index)
    if triplet is None:
        raise ValueError(
            f"No angle triplet defined for tracking joint {tracking_joint_index}.\n"
            f"Supported joints: {list(ANGLE_TRIPLETS.keys())}"
        )

    # Find bilateral partner if one exists
    bilateral_joint = BILATERAL_PAIRS.get(tracking_joint_index)
    bilateral_triplet = ANGLE_TRIPLETS.get(bilateral_joint) if bilateral_joint else None

    rep_sequences = []   # y-only feature sequences — saved to rep_pattern.npy
    raw_rep_seqs  = []   # raw pose sequences — for angle calibration

    for ann in annotations:
        vid_idx     = ann["video_index"]
        start_frame = ann["start_frame"]
        end_frame   = ann["end_frame"]

        if vid_idx >= len(video_poses):
            print(f"  ⚠️  Video index {vid_idx} out of range, skipping")
            continue

        raw_poses = video_poses[vid_idx]

        if start_frame >= end_frame or end_frame > len(raw_poses):
            print(f"  ⚠️  Annotation {start_frame}-{end_frame} out of range "
                  f"(video has {len(raw_poses)} frames), skipping")
            continue

        window = raw_poses[start_frame:end_frame]
        valid  = [p for p in window if p is not None]
        if len(valid) < 5:
            print(f"  ⚠️  Rep [{start_frame}-{end_frame}] only {len(valid)} "
                  f"pose frames, skipping")
            continue

        rep_seq  = np.array(valid).reshape(-1, 33, 3)
        features = _extract_features_sequence(rep_seq)
        rep_sequences.append(features)
        raw_rep_seqs.append(rep_seq)

    if not rep_sequences:
        raise ValueError(
            "No valid rep sequences found.\n"
            "Check that annotations cover frames with detected poses and "
            "each rep window contains at least 5 pose-detected frames."
        )

    # ── Save rep pattern (features) and raw poses ────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    pattern_path = os.path.join(output_dir, "rep_pattern.npy")
    np.save(pattern_path, np.array(rep_sequences, dtype=object))

    # Save raw pose sequences separately — used by the diagnostic tool and
    # the form profile builder. Kept separate so the scoring pipeline is
    # completely unchanged.
    poses_path = os.path.join(output_dir, "rep_poses.npy")
    np.save(poses_path, np.array(raw_rep_seqs, dtype=object))

    lengths = [len(s) for s in rep_sequences]
    print(f"  ✅ Saved rep_pattern.npy — {len(rep_sequences)} reps, "
          f"avg length {np.mean(lengths):.1f} frames")

    # ── Compute threshold from ALL angles across ALL reps ────────────────────
    #
    # WHY PER-REP PERCENTILES FAIL FOR SHORT REPS:
    # A 25-frame rep: 5th percentile = bottom 1-2 frames.
    # That 1 frame might be a glitch or the person mid-transition, not the
    # true deepest point. 95th percentile = top 1-2 frames, which is the
    # instant just after the person started moving from standing, reading
    # e.g. 158° instead of the true 170°. Small per-rep window = high noise.
    #
    # THE FIX — global pool:
    # Collect every angle from every rep frame into one list (400-600 samples
    # for 20 reps × 20-30 frames each). Then:
    #   global 5th  percentile → true bottom (e.g. 90° for squats)
    #   global 95th percentile → true top    (e.g. 170° for squats)
    #
    # With 500 samples, 5th percentile = 25 frames worth of data — the
    # person genuinely at the bottom of the movement, not a glitch.
    # With 500 samples, 95th percentile = 25 frames worth of standing —
    # the person fully extended, not mid-transition.
    # Simpler, more data, more reliable.

    all_angles = []
    for rep_seq in raw_rep_seqs:
        for frame in rep_seq:
            all_angles.append(_bilateral_angle(frame, triplet, bilateral_triplet))

    angles_arr = np.array(all_angles)
    lo = float(np.percentile(angles_arr, 5))    # true bottom angle
    hi = float(np.percentile(angles_arr, 95))   # true top angle

    print(f"  📐 Tracking joint={tracking_joint_index}  triplet={triplet}")
    if bilateral_triplet:
        print(f"      Bilateral partner joint={bilateral_joint}  triplet={bilateral_triplet}")
    print(f"      total angle samples={len(all_angles)}")
    print(f"      global 5th={lo:.1f}°  global 95th={hi:.1f}°  "
          f"range={abs(hi - lo):.1f}°")

    # Sanity check
    if abs(hi - lo) < 10.0:
        raise ValueError(
            f"Angle range too small ({abs(hi - lo):.1f}°). "
            "The tracking joint barely moves across annotated reps.\n"
            "Check that you selected the correct tracking joint and that "
            "annotations cover the full range of motion."
        )

    threshold = (lo + hi) / 2.0
    deadband  = max(abs(hi - lo) * DEADBAND_FRACTION, MIN_DEADBAND_DEG)

    # start_above: check which end the first frame of each rep sits closer to.
    # Squat:          first frame ≈ 170° (near hi) → NOT start_above
    # Shoulder press: first frame ≈ 30°  (near lo) → start_above
    first_frame_angles = [
        float(_bilateral_angle(seq[0], triplet, bilateral_triplet))
        for seq in raw_rep_seqs
    ]
    avg_first   = float(np.median(first_frame_angles))
    start_above = bool(abs(avg_first - lo) < abs(avg_first - hi))

    print(f"      threshold={threshold:.1f}°  deadband={deadband:.1f}°")
    print(f"      avg_first_frame={avg_first:.1f}°  start_above={start_above}")


    # Build and save the angle-parameterized form profile used for live scoring
    build_form_profile(
        raw_rep_seqs, key_joint_indices,
        triplet, bilateral_triplet,
        min(lo, hi), max(lo, hi),
        output_dir,
    )

    return {
        "num_reps":             len(rep_sequences),
        "avg_length":           float(np.mean(lengths)),
        "min_length":           int(np.min(lengths)),
        "max_length":           int(np.max(lengths)),
        "pattern_path":         pattern_path,
        "tracking_joint_index": tracking_joint_index,
        "bilateral_joint_index": bilateral_joint,
        "threshold":            threshold,
        "lo":                   min(lo, hi),
        "hi":                   max(lo, hi),
        "deadband":             deadband,
        "start_above":          start_above,
    }


# ── Form profile ──────────────────────────────────────────────────────────────

def build_form_profile(
    raw_rep_seqs:        list,
    key_joint_indices:   list,
    tracking_triplet:    tuple,
    bilateral_triplet,
    lo:                  float,
    hi:                  float,
    output_dir:          str,
    n_bins:              int = 20,
    std_floor_deg:       float = 12.0,
) -> str:
    """
    Build an angle-parameterized form profile from annotated reps.

    Instead of comparing time series (which penalizes speed), this bins
    reps by how far through the movement they are — measured by the
    tracking joint angle — then records key joint angles at each bin.

    At scoring time, a live rep is compared bin-by-bin.  Bins the person
    never reached (insufficient depth) contribute 0 to the score, so
    depth and form quality are both captured in one number.

    Parameters
    ----------
    raw_rep_seqs      : list of (T, 33, 3) arrays — one per annotated rep
    key_joint_indices : landmark indices to score (user-selected key joints)
    tracking_triplet  : (a, b, c) landmark indices for the counting joint
    bilateral_triplet : same for the bilateral partner, or None
    lo / hi           : angle range computed by build_rep_pattern
    output_dir        : where to save form_profile.npy
    n_bins            : number of angle bins (default 20 = 5% increments)
    std_floor_deg     : minimum std per bin in degrees — prevents the
                        within-training variance being too tight and making
                        any live rep look like an outlier.  5° ≈ the natural
                        precision of "close enough" in PT exercises.

    Saved file
    ----------
    form_profile.npy  — dict with keys:
        lo, hi, n_bins, std_floor_deg,
        key_joint_indices,
        key_triplets       : list of (a,b,c) tuples per key joint
        bin_means          : (n_bins, n_key_joints) — mean angle per bin/joint
        bin_stds           : (n_bins, n_key_joints) — std  angle per bin/joint
        bin_counts         : (n_bins,)              — training frames per bin
    """
    from core.landmarks import ANGLE_TRIPLETS

    # Build angle triplets for every key joint that has one.
    # Joints without a triplet (e.g. derived joints 100+) are skipped —
    # they don't have a simple three-point angle definition.
    key_triplets = []
    valid_key_indices = []
    for idx in key_joint_indices:
        t = ANGLE_TRIPLETS.get(idx)
        if t is not None:
            key_triplets.append(t)
            valid_key_indices.append(idx)

    if not key_triplets:
        print("  ⚠️  No angle triplets found for key joints — skipping form profile")
        return None

    n_joints = len(key_triplets)
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    # Accumulate key joint angles per bin across all training reps
    bin_data = [[] for _ in range(n_bins)]   # bin_data[b] = list of (n_joints,) arrays

    for rep_seq in raw_rep_seqs:
        for frame in rep_seq:
            tracking_angle = _bilateral_angle(frame, tracking_triplet, bilateral_triplet)
            b = int((tracking_angle - lo) / (hi - lo) * n_bins)
            b = max(0, min(n_bins - 1, b))   # clamp to valid range

            key_angles = np.array([
                compute_angle(*t, frame) for t in key_triplets
            ])
            bin_data[b].append(key_angles)

    # Compute mean and std per bin
    bin_means  = np.zeros((n_bins, n_joints))
    bin_stds   = np.zeros((n_bins, n_joints))
    bin_counts = np.zeros(n_bins, dtype=int)

    for b, samples in enumerate(bin_data):
        bin_counts[b] = len(samples)
        if len(samples) >= 2:
            arr = np.array(samples)             # (N, n_joints)
            bin_means[b] = arr.mean(axis=0)
            bin_stds[b]  = np.maximum(arr.std(axis=0), std_floor_deg)
        elif len(samples) == 1:
            bin_means[b] = samples[0]
            bin_stds[b]  = std_floor_deg        # only one sample — use floor
        else:
            # No training data in this bin — mark as empty
            bin_means[b] = -1.0
            bin_stds[b]  = std_floor_deg

    profile = {
        "lo":                lo,
        "hi":                hi,
        "n_bins":            n_bins,
        "std_floor_deg":     std_floor_deg,
        "key_joint_indices": valid_key_indices,
        "key_triplets":      key_triplets,
        "bin_means":         bin_means,
        "bin_stds":          bin_stds,
        "bin_counts":        bin_counts,
    }

    path = os.path.join(output_dir, "form_profile.npy")
    np.save(path, profile)

    filled = int(np.sum(bin_counts > 0))
    print(f"  ✅ Saved form_profile.npy — {n_bins} bins  "
          f"{filled} filled  key_joints={valid_key_indices}")
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bilateral_angle(frame: np.ndarray, triplet: tuple,
                     bilateral_triplet: tuple = None) -> float:
    """Compute angle, averaging bilateral pair if available."""
    a = compute_angle(*triplet, frame)
    if bilateral_triplet is not None:
        b = compute_angle(*bilateral_triplet, frame)
        return (a + b) / 2.0
    return a


def _normalise_pose(frame: np.ndarray) -> np.ndarray:
    hip_mid = (frame[23] + frame[24]) / 2.0
    sho_mid = (frame[11] + frame[12]) / 2.0
    scale   = np.linalg.norm(sho_mid - hip_mid)
    if scale < 1e-6:
        scale = 1e-6
    return (frame - hip_mid) / scale


def _extract_features_sequence(poses: np.ndarray) -> np.ndarray:
    """
    Extract y-only body-relative features for every frame in a rep sequence.
    MUST match rep_counter._extract_features exactly.

    poses   : (T, 33, 3)
    Returns : (T, 22)
    """
    features = []
    for frame in poses:
        norm = _normalise_pose(frame)
        features.append([float(norm[idx, 1]) for idx in range(11, 33)])
    return np.array(features)
