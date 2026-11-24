"""
diagnose_scoring.py
--------------------
Run from your project root AFTER retraining:
    python diagnose_scoring.py

Diagnoses TWO things:

1. TRAINING PIPELINE
   - Angle curve of every annotated rep over time
     (should look like a clean U-shape for squats, V-shape for curls etc.)
   - Where the computed threshold + deadband sit on those curves
   - Whether annotations started/ended at the right angle
     (if a rep curve starts at 140° instead of 170°, annotation was mid-rep)
   - Which reps are outliers vs the group

2. SCORING APPROACH
   - Builds an angle-parameterized form profile from your training reps
   - Leave-one-out cross validation: score each rep against all others
   - Shows score distribution and per-rep scores
   - Shows per-bin mean ± std for each key joint
   - Shows which bins have sparse training data

Output: one PNG per exercise saved to current directory.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Standalone copies of angle helpers ───────────────────────────────────────
ANGLE_TRIPLETS = {
    11: (13, 11, 23), 12: (14, 12, 24),
    13: (11, 13, 15), 14: (12, 14, 16),
    23: (11, 23, 25), 24: (12, 24, 26),
    25: (23, 25, 27), 26: (24, 26, 28),
    27: (25, 27, 31), 28: (26, 28, 32),
}
BILATERAL_PAIRS = {
    11:12, 12:11, 13:14, 14:13,
    23:24, 24:23, 25:26, 26:25,
    27:28, 28:27,
}

def compute_angle(a_idx, b_idx, c_idx, frame):
    a = frame[a_idx, :2];  b = frame[b_idx, :2];  c = frame[c_idx, :2]
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def bilateral_angle(frame, triplet, b_triplet=None):
    a = compute_angle(*triplet, frame)
    if b_triplet:
        return (a + compute_angle(*b_triplet, frame)) / 2.0
    return a

# ── Profile builder ───────────────────────────────────────────────────────────
def build_profile(seqs, key_triplets, tracking_triplet, b_triplet,
                  lo, hi, n_bins=20, std_floor=5.0):
    n_joints = len(key_triplets)
    bin_data = [[] for _ in range(n_bins)]
    for seq in seqs:
        for frame in seq:
            ta = bilateral_angle(frame, tracking_triplet, b_triplet)
            b  = int((ta - lo) / (hi - lo) * n_bins)
            b  = max(0, min(n_bins - 1, b))
            bin_data[b].append([compute_angle(*t, frame) for t in key_triplets])
    bin_means  = np.zeros((n_bins, n_joints))
    bin_stds   = np.full((n_bins, n_joints), std_floor)
    bin_counts = np.zeros(n_bins, dtype=int)
    for b, samples in enumerate(bin_data):
        bin_counts[b] = len(samples)
        if len(samples) >= 2:
            arr = np.array(samples)
            bin_means[b] = arr.mean(axis=0)
            bin_stds[b]  = np.maximum(arr.std(axis=0), std_floor)
        elif len(samples) == 1:
            bin_means[b] = samples[0]
    return {"means": bin_means, "stds": bin_stds,
            "counts": bin_counts, "n_bins": n_bins, "lo": lo, "hi": hi}

def score_rep(seq, profile, key_triplets, tracking_triplet, b_triplet):
    n_bins = profile["n_bins"]
    lo, hi = profile["lo"], profile["hi"]
    means  = profile["means"]
    stds   = profile["stds"]
    counts = profile["counts"]
    hit       = np.zeros(n_bins, dtype=bool)
    bin_devs  = [[] for _ in range(n_bins)]
    for frame in seq:
        ta = bilateral_angle(frame, tracking_triplet, b_triplet)
        b  = int((ta - lo) / (hi - lo) * n_bins)
        b  = max(0, min(n_bins - 1, b))
        if counts[b] == 0:
            continue
        key_angles = np.array([compute_angle(*t, frame) for t in key_triplets])
        dev = np.mean(np.abs(key_angles - means[b]) / stds[b])
        bin_devs[b].append(dev)
        hit[b] = True
    bin_scores = np.zeros(n_bins)
    for b in range(n_bins):
        if hit[b] and bin_devs[b]:
            bin_scores[b] = max(0.0, 100.0 - np.mean(bin_devs[b]) * 20.0)
    valid = counts > 0
    if not np.any(valid):
        return 0, bin_scores
    return float(np.mean(bin_scores[valid])), bin_scores

# ── Load exercises ────────────────────────────────────────────────────────────
EXERCISES_DATA_DIR = "exercises_data"
if not os.path.exists(EXERCISES_DATA_DIR):
    print(f"❌  exercises_data/ not found in {os.getcwd()}")
    sys.exit(1)

for folder_name in sorted(os.listdir(EXERCISES_DATA_DIR)):
    folder_path  = os.path.join(EXERCISES_DATA_DIR, folder_name)
    config_path  = os.path.join(folder_path, "config.json")
    poses_path   = os.path.join(folder_path, "rep_poses.npy")
    pattern_path = os.path.join(folder_path, "rep_pattern.npy")

    if not os.path.isdir(folder_path):   continue
    if not os.path.exists(config_path):  continue
    if not os.path.exists(pattern_path): continue

    with open(config_path) as f:
        cfg = json.load(f)

    ex_name = cfg.get("name", folder_name)
    print(f"\n{'='*60}")
    print(f"Exercise: {ex_name}")

    # ── Check for rep_poses.npy ───────────────────────────────────────────────
    if not os.path.exists(poses_path):
        print(f"  ❌  rep_poses.npy not found.")
        print(f"      This means the exercise was trained with the OLD pipeline.")
        print(f"      → Swap in the updated rep_trainer.py and retrain.")
        continue

    raw_seqs = list(np.load(poses_path, allow_pickle=True))
    print(f"  Annotated reps  : {len(raw_seqs)}")

    lo  = cfg.get("lo");   hi  = cfg.get("hi")
    tj  = cfg.get("tracking_joint_index")
    thr = cfg.get("threshold_override", cfg.get("threshold_auto"))
    dbd = cfg.get("deadband_override",  cfg.get("deadband_auto"))
    sa  = cfg.get("start_above", False)
    key_joints = [j["landmark_index"] for j in cfg.get("key_joints", [])]

    print(f"  lo / hi         : {lo:.1f}° / {hi:.1f}°")
    print(f"  Threshold       : {thr:.1f}°   deadband: {dbd:.1f}°")
    print(f"  start_above     : {sa}")
    print(f"  Tracking joint  : {tj}")
    print(f"  Key joints      : {key_joints}")

    if lo is None or hi is None or tj is None:
        print("  ❌  Missing lo/hi/tracking_joint in config. Retrain.")
        continue

    tracking_triplet = ANGLE_TRIPLETS.get(tj)
    if not tracking_triplet:
        print(f"  ❌  No angle triplet for joint {tj}. Skipping.")
        continue

    b_joint   = BILATERAL_PAIRS.get(tj)
    b_triplet = ANGLE_TRIPLETS.get(b_joint) if b_joint else None

    key_triplets = []
    valid_key    = []
    for idx in key_joints:
        t = ANGLE_TRIPLETS.get(idx)
        if t:
            key_triplets.append(t)
            valid_key.append(idx)

    # ── Compute angle curve for every rep ─────────────────────────────────────
    rep_angle_curves = []
    rep_stats = []   # (start_angle, end_angle, min_angle, max_angle, n_frames)
    for seq in raw_seqs:
        curve = [bilateral_angle(f, tracking_triplet, b_triplet) for f in seq]
        rep_angle_curves.append(curve)
        rep_stats.append({
            "start":    curve[0],
            "end":      curve[-1],
            "min":      min(curve),
            "max":      max(curve),
            "n_frames": len(curve),
        })

    # ── Print per-rep pipeline diagnostics ───────────────────────────────────
    print(f"\n  {'Rep':>4}  {'Start°':>7}  {'End°':>7}  {'Min°':>7}  "
          f"{'Max°':>7}  {'Range°':>7}  {'Frames':>7}  {'Start OK':>9}  {'End OK':>7}")
    print(f"  {'-'*75}")
    start_ok_threshold = hi - 10   # start should be within 10° of hi
    end_ok_threshold   = hi - 10
    n_bad_start = 0;  n_bad_end = 0

    for i, s in enumerate(rep_stats):
        rng       = s["max"] - s["min"]
        start_ok  = s["start"] >= start_ok_threshold
        end_ok    = s["end"]   >= end_ok_threshold
        if not start_ok: n_bad_start += 1
        if not end_ok:   n_bad_end   += 1
        flag = ""
        if not start_ok: flag += "⚠START "
        if not end_ok:   flag += "⚠END"
        print(f"  {i+1:>4}  {s['start']:>7.1f}  {s['end']:>7.1f}  "
              f"{s['min']:>7.1f}  {s['max']:>7.1f}  {rng:>7.1f}  "
              f"{s['n_frames']:>7}  {'✓' if start_ok else '✗':>9}  "
              f"{'✓' if end_ok else '✗':>7}  {flag}")

    if n_bad_start > 0:
        print(f"\n  ⚠️  {n_bad_start} reps start below {start_ok_threshold:.0f}° "
              f"— annotation began mid-movement, not at rest.")
        print(f"     Threshold is likely computed too low because of this.")
    if n_bad_end > 0:
        print(f"\n  ⚠️  {n_bad_end} reps end below {end_ok_threshold:.0f}° "
              f"— annotation ended before person returned to rest.")

    # ── LOO scoring ───────────────────────────────────────────────────────────
    loo_scores = []
    if key_triplets:
        for i, test_seq in enumerate(raw_seqs):
            train_seqs = [s for j, s in enumerate(raw_seqs) if j != i]
            profile    = build_profile(train_seqs, key_triplets,
                                       tracking_triplet, b_triplet, lo, hi)
            score, _   = score_rep(test_seq, profile, key_triplets,
                                   tracking_triplet, b_triplet)
            loo_scores.append(score)

        print(f"\n  LOO scores: mean={np.mean(loo_scores):.1f}  "
              f"std={np.std(loo_scores):.1f}  "
              f"min={np.min(loo_scores):.1f}  "
              f"max={np.max(loo_scores):.1f}")
        if np.mean(loo_scores) >= 75:
            print("  ✅ Scoring: GOOD")
        elif np.mean(loo_scores) >= 55:
            print("  ⚠️  Scoring: MARGINAL")
        else:
            print("  ❌ Scoring: POOR — inconsistent training reps")
    else:
        print("  ⚠️  No key joints with angle triplets — skipping LOO scoring")
        loo_scores = [0] * len(raw_seqs)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{ex_name} — Training Pipeline + Scoring Diagnostic",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors_rep = plt.cm.tab20(np.linspace(0, 1, len(raw_seqs)))

    # ── Plot 1: Angle curves per rep (time) ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, curve in enumerate(rep_angle_curves):
        t = np.linspace(0, len(curve)/30.0, len(curve))   # seconds at 30fps
        ax1.plot(t, curve, color=colors_rep[i], alpha=0.6, linewidth=1.2)

    # Threshold and deadband bands
    if thr and dbd:
        commit = thr - dbd if not sa else thr + dbd
        ret    = thr + dbd if not sa else thr - dbd
        ax1.axhline(thr,    color='yellow',  linewidth=1.5, linestyle='-',
                    label=f"Threshold {thr:.0f}°")
        ax1.axhline(commit, color='orange',  linewidth=1,   linestyle='--',
                    label=f"Commit {commit:.0f}°")
        ax1.axhline(ret,    color='cyan',    linewidth=1,   linestyle='--',
                    label=f"Return {ret:.0f}°")
    ax1.axhline(lo, color='red',   linewidth=1, linestyle=':', alpha=0.7,
                label=f"lo={lo:.0f}°")
    ax1.axhline(hi, color='green', linewidth=1, linestyle=':', alpha=0.7,
                label=f"hi={hi:.0f}°")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Tracking joint angle (°)")
    ax1.set_title("Angle curves per rep\n(all reps overlaid)")
    ax1.legend(fontsize=7, loc='upper right')

    # ── Plot 2: Angle curves normalised 0→1 (removes speed difference) ───────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, curve in enumerate(rep_angle_curves):
        t_norm = np.linspace(0, 1, len(curve))
        ax2.plot(t_norm, curve, color=colors_rep[i], alpha=0.6, linewidth=1.2)
    if thr:
        ax2.axhline(thr, color='yellow', linewidth=1.5, linestyle='-',
                    label=f"Threshold {thr:.0f}°")
    ax2.set_xlabel("Rep progress (0=start, 1=end)")
    ax2.set_ylabel("Tracking joint angle (°)")
    ax2.set_title("Angle curves normalised by time\n(speed removed — shape only)")
    ax2.legend(fontsize=8)

    # ── Plot 3: Start and end angles per rep ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    starts = [s["start"] for s in rep_stats]
    ends   = [s["end"]   for s in rep_stats]
    mins   = [s["min"]   for s in rep_stats]
    reps   = list(range(1, len(rep_stats)+1))
    ax3.scatter(reps, starts, color='green',  s=50, zorder=3, label="Start angle")
    ax3.scatter(reps, ends,   color='cyan',   s=50, zorder=3, label="End angle")
    ax3.scatter(reps, mins,   color='red',    s=50, zorder=3, label="Deepest angle")
    ax3.axhline(hi - 10, color='green', linewidth=1, linestyle='--',
                alpha=0.6, label=f"Expected start ≥{hi-10:.0f}°")
    ax3.set_xlabel("Rep number")
    ax3.set_ylabel("Angle (°)")
    ax3.set_title("Start / end / deepest angle per rep\n(start+end should cluster near hi)")
    ax3.legend(fontsize=7)

    # ── Plot 4: Bin density ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if key_triplets:
        full_profile = build_profile(raw_seqs, key_triplets,
                                     tracking_triplet, b_triplet, lo, hi)
        bin_centers  = np.linspace(lo, hi, 20)
        ax4.bar(bin_centers, full_profile["counts"],
                width=(hi-lo)/20*0.8, color="#4a90d9", alpha=0.8)
        ax4.axhline(5, color='red', linestyle='--', alpha=0.6,
                    label="Min reliable (5 frames)")
        for b, c in enumerate(full_profile["counts"]):
            if c == 0:
                ax4.axvspan(bin_centers[b]-(hi-lo)/40,
                            bin_centers[b]+(hi-lo)/40,
                            color='red', alpha=0.3)
        ax4.set_xlabel("Tracking joint angle (°)")
        ax4.set_ylabel("Training frames in bin")
        ax4.set_title("Bin fill density\n(red = empty bin = blind spot in scoring)")
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No key joints with triplets",
                 ha='center', va='center', transform=ax4.transAxes)

    # ── Plot 5: Per-joint form profile ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if key_triplets:
        colors_j = plt.cm.tab10(np.linspace(0, 1, len(key_triplets)))
        valid_mask = full_profile["counts"] > 0
        for j, (triplet, cj) in enumerate(zip(key_triplets, colors_j)):
            m = full_profile["means"][:, j]
            s = full_profile["stds"][:, j]
            ax5.plot(bin_centers[valid_mask], m[valid_mask],
                     color=cj, linewidth=2, label=f"Joint {valid_key[j]}")
            ax5.fill_between(bin_centers[valid_mask],
                             (m-s)[valid_mask], (m+s)[valid_mask],
                             color=cj, alpha=0.15)
        ax5.set_xlabel("Tracking joint angle (°)")
        ax5.set_ylabel("Key joint angle (°)")
        ax5.set_title("Form profile\n(mean ± 1 std per bin — should be smooth)")
        ax5.legend(fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No key joints with triplets",
                 ha='center', va='center', transform=ax5.transAxes)

    # ── Plot 6: LOO scores per rep ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if loo_scores and any(s > 0 for s in loo_scores):
        bar_colors = ["#2d6a4f" if s >= 75 else "#f4a261" if s >= 55 else "#e63946"
                      for s in loo_scores]
        ax6.bar(range(1, len(loo_scores)+1), loo_scores,
                color=bar_colors, alpha=0.9)
        ax6.axhline(75, color='green', linestyle='--', linewidth=1.5,
                    label="Good (75)")
        ax6.axhline(np.mean(loo_scores), color='yellow', linewidth=1.5,
                    label=f"Mean ({np.mean(loo_scores):.1f})")
        ax6.set_ylim(0, 105)
        ax6.legend(fontsize=8)
        legend_patches = [
            Patch(color="#2d6a4f", label="Good ≥75"),
            Patch(color="#f4a261", label="Marginal 55-75"),
            Patch(color="#e63946", label="Poor <55"),
        ]
        ax6.legend(handles=legend_patches, fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No LOO scores\n(no key joints with triplets)",
                 ha='center', va='center', transform=ax6.transAxes)
    ax6.set_xlabel("Rep number")
    ax6.set_ylabel("LOO Score (0-100)")
    ax6.set_title("Leave-one-out scores per rep\n(each rep scored against all others)")

    out_path = f"diagnostic_{ex_name.replace(' ','_').lower()}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Saved: {out_path}")

print("\n" + "="*60)
print("WHAT TO LOOK FOR IN THE GRAPHS:")
print()
print("Plot 1 — Angle curves (time):")
print("  All curves should have a similar U-shape.")
print("  Jagged = MediaPipe noise in those reps.")
print("  Very flat bottom = reps didn't go to full depth.")
print()
print("Plot 2 — Angle curves (normalised):")
print("  When speed is removed, curves should overlap tightly.")
print("  Wide spread = people squatted to different depths.")
print()
print("Plot 3 — Start/end/deepest per rep:")
print("  Start + End dots should cluster near hi (e.g. 165-170° for squats).")
print("  If start dots are scattered low → annotation started mid-movement.")
print("  Red dots (deepest) should cluster near lo (e.g. 85-95° for squats).")
print("  If red dots are scattered high → reps didn't go to full depth.")
print()
print("Plot 4 — Bin density:")
print("  Red zones = no training data at that angle = scoring blind spot.")
print("  If the deepest bins are empty → training reps never reached lo.")
print()
print("Plot 5 — Form profile:")
print("  Should be smooth curves, not jagged.")
print("  Jagged = key joints selected don't move predictably with the exercise.")
print()
print("Plot 6 — LOO scores:")
print("  Mean ≥ 75 → scoring approach works with this data.")
print("  Red bars → those specific reps are outliers, consider removing them.")
