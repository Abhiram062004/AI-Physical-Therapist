"""
core/rep_counter.py
-------------------
Rep counting via joint angle threshold crossing with hysteresis.
Form scoring via body-relative normalised y-position path deviation.

FIXES in this version
---------------------
1. SCORE_SENSITIVITY reduced 8 → 3: previously a mean_z of just 12.5 would
   produce a score of 0. Now mean_z needs to reach 33 to hit 0. With only
   5 training reps, variance was tiny so even clean reps scored 0. Now even
   small templates produce meaningful scores.

2. SCORE_SCALE_FLOOR raised 0.05 → 0.15: the per-feature std floor prevents
   z-score explosion on features that barely vary between training reps.

3. Bilateral angle averaging: if the tracking joint has a bilateral partner
   (e.g. left knee 25 → right knee 26), the angle used for the state machine
   is the average of both sides. Reduces noise from minor asymmetries and
   makes exercises like squats more robust.

4. Visibility gating: if the landmarks forming the tracking joint's angle
   have low MediaPipe confidence, the frame is skipped. Prevents occluded
   limbs (hand behind body, foot out of frame) from triggering phantom reps.

5. Angle spike filtering: if the raw angle jumps more than MAX_ANGLE_JUMP
   degrees in one frame, the frame is skipped. Catches MediaPipe glitches.

Counting
--------
  Rep STARTS  when ema crosses threshold toward movement
  Rep COMMITS when ema crosses the commit line (went deep enough)
  Rep ENDS    when ema crosses the return line (fully back to rest)

Scoring
-------
  y-only hip-relative features, invariant to horizontal flip and depth.
"""

import numpy as np
from core.exercise_loader import LoadedExercise
from core.landmarks import ANGLE_TRIPLETS, BILATERAL_PAIRS, compute_angle

# ── Constants ──────────────────────────────────────────────────────────────────

POS_EMA_ALPHA = 0.6   # angle EMA — responsive with light smoothing

MIN_REP_FRAMES     = 8   # minimum frames to count as a rep (~0.27 s at 30 fps)
MIN_REP_GAP_FRAMES = 10  # minimum frames between two reps (~0.33 s at 30 fps)

# Low floor — prevents z-score explosion on features that genuinely never vary
SCORE_SCALE_FLOOR = 0.05

CROSS_SESSION_SCALE = 4.0

# ── Scoring tolerance (dead zone) ────────────────────────────────────────────
# Deviations below SCORE_DEAD_ZONE_STD are ignored completely — normal wobble,
# minor camera differences, and breathing all produce sub-1.5 std deviations
# and should not cost any points.
# Only deviations ABOVE the dead zone are penalised at SCORE_PENALTY_RATE
# points per std above the threshold.
#
# Examples:
#   0.0 std off -> 100  (perfect)
#   1.0 std off -> 100  (within dead zone, ignored)
#   1.5 std off -> 100  (edge of dead zone, still ignored)
#   2.0 std off ->  92  (0.5 std past zone x 15 = 7.5 penalty)
#   3.0 std off ->  77  (1.5 std past zone x 15 = 22.5 penalty)
#   5.0 std off ->  47  (3.5 std past zone x 15 = 52.5 penalty)
SCORE_DEAD_ZONE_STD = 1.5   # std deviations ignored completely
SCORE_PENALTY_RATE  = 15.0  # points lost per std ABOVE the dead zone

# Visibility threshold — skip frame if tracking landmarks below this confidence
MIN_LANDMARK_VISIBILITY = 0.5

# Angle jump threshold — skip frame if raw angle changes by more than this
MAX_ANGLE_JUMP = 40.0  # degrees


class RepCounter:
    """
    One instance per exercise session.
    Call .update(pose_frame, visibility) every frame a pose is detected.
    Call .notify_no_detection() every frame no pose is detected.

    Public attributes
    -----------------
    rep_count           : int
    current_form_score  : int | None
    last_rep_score      : int | None
    is_in_rep           : bool
    """

    def __init__(self, exercise: LoadedExercise):
        self.exercise           = exercise
        self.rep_count              = 0
        self.current_form_score     = None
        self.last_rep_score         = None
        self.last_rep_joint_scores  = {}   # {landmark_idx: score 0-100}
        self.last_rep_joint_depths  = {}   # {landmark_idx: max_depth_pct 0-100}
        self.is_in_rep              = False
        self._ready                 = False

        self._tracking_triplet   = None
        self._bilateral_triplet  = None   # partner joint triplet for averaging
        self._threshold          = None
        self._deadband           = None
        self._start_above        = None
        self._line_commit        = None
        self._line_return        = None

        self._committed        = False
        self._in_start_pos     = True
        self._rep_frame_count  = 0
        self._ema_val          = None
        self._last_raw_angle   = None     # for spike filtering
        self._frames_since_rep = MIN_REP_GAP_FRAMES + 1

        self._path_mean  = None
        self._path_std   = None
        self._avg_len    = 0
        self._rep_frames     = []
        self._rep_raw_frames = []

        # Feature vector covers landmarks 11-32 (indices 0-21).
        # Map key joint landmark indices to their column positions so scoring
        # only weighs the joints that actually matter for this exercise.
        # Non-key joints (e.g. wrists during squats) add noise and dilute scores.
        self._key_feature_indices = sorted([
            idx - 11 for idx in exercise.key_joint_indices
            if 11 <= idx <= 32
        ])

        self._setup()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup(self):
        pattern     = self.exercise.rep_pattern
        key_indices = self.exercise.key_joint_indices
        cfg         = self.exercise.config

        if pattern is None or len(pattern) == 0:
            print(f"[RepCounter] '{self.exercise.name}' — no rep pattern found")
            return
        if not key_indices:
            print(f"[RepCounter] '{self.exercise.name}' — no key joints defined")
            return

        tracking_joint = cfg.get("tracking_joint_index") or key_indices[0]
        triplet        = ANGLE_TRIPLETS.get(tracking_joint)
        if triplet is None:
            print(f"[RepCounter] '{self.exercise.name}' — no angle triplet for "
                  f"joint {tracking_joint}.")
            return

        self._tracking_triplet = triplet

        # Bilateral partner
        bilateral_joint = (
            cfg.get("bilateral_joint_index")
            or BILATERAL_PAIRS.get(tracking_joint)
        )
        self._bilateral_triplet = ANGLE_TRIPLETS.get(bilateral_joint) if bilateral_joint else None

        if "threshold_auto" not in cfg and "threshold_override" not in cfg:
            print(f"[RepCounter] '{self.exercise.name}' — no threshold in config. Retrain.")
            return

        self._threshold   = cfg.get("threshold_override", cfg["threshold_auto"])
        self._deadband    = cfg.get("deadband_override",  cfg["deadband_auto"])
        self._start_above = cfg.get("start_above", False)

        if self._start_above:
            self._line_commit = self._threshold + self._deadband
            self._line_return = self._threshold - self._deadband
        else:
            self._line_commit = self._threshold - self._deadband
            self._line_return = self._threshold + self._deadband

        lo       = cfg.get("lo", "?")
        hi       = cfg.get("hi", "?")
        override = "threshold_override" in cfg
        bilateral_str = f"  bilateral={bilateral_joint}" if bilateral_joint else ""
        print(
            f"[RepCounter] '{self.exercise.name}' ready  "
            f"joint={tracking_joint}{bilateral_str}  triplet={triplet}  "
            f"range=[{lo}°, {hi}°]  "
            f"threshold={self._threshold:.1f}°{'(override)' if override else ''}  "
            f"commit={'>' if self._start_above else '<'}{self._line_commit:.1f}°  "
            f"return={'<' if self._start_above else '>'}{self._line_return:.1f}°  "
            f"avg_rep_len={self._avg_len}  n_training_reps={len(pattern)}"
        )
        self._ready = True

    # ── Per-frame update ────────────────────────────────────────────────────────

    def update(self, pose_frame: np.ndarray,
               visibility: np.ndarray = None) -> bool:
        """
        Call once per frame where a pose is detected.

        Parameters
        ----------
        pose_frame : (33, 3) landmark positions
        visibility : (33,) MediaPipe landmark visibility scores, optional.
                     When provided, frames where tracking landmarks have low
                     confidence are skipped to prevent phantom reps from
                     occluded limbs.

        Returns True if a rep was just completed.
        """
        self._frames_since_rep += 1

        if not self._ready:
            return False

        # ── Visibility gate ───────────────────────────────────────────────────
        if visibility is not None:
            a_idx, b_idx, c_idx = self._tracking_triplet
            if (visibility[a_idx] < MIN_LANDMARK_VISIBILITY or
                    visibility[b_idx] < MIN_LANDMARK_VISIBILITY or
                    visibility[c_idx] < MIN_LANDMARK_VISIBILITY):
                # Reset the spike-filter reference so the first visible frame after
                # an occlusion isn't killed because the angle legitimately changed
                # while the landmark was hidden.
                self._last_raw_angle = None
                return False  # landmark not reliably visible, skip frame

        # ── Compute angle, optionally averaging bilateral partner ─────────────
        raw_angle = compute_angle(*self._tracking_triplet, pose_frame)
        if self._bilateral_triplet is not None:
            raw_angle = (raw_angle +
                         compute_angle(*self._bilateral_triplet, pose_frame)) / 2.0

        # ── Spike filter ──────────────────────────────────────────────────────
        if (self._last_raw_angle is not None and
                abs(raw_angle - self._last_raw_angle) > MAX_ANGLE_JUMP):
            # Always update the reference even when rejecting the frame.
            # Without this, every subsequent frame is compared against a stale
            # reference and the filter trips continuously during fast movements.
            self._last_raw_angle = raw_angle
            return False  # sudden jump — probably a MediaPipe glitch
        self._last_raw_angle = raw_angle

        # ── EMA smoothing ─────────────────────────────────────────────────────
        if self._ema_val is None:
            self._ema_val = raw_angle
        else:
            self._ema_val = (POS_EMA_ALPHA * raw_angle +
                             (1 - POS_EMA_ALPHA) * self._ema_val)

        # Periodic debug
        if self._frames_since_rep % 30 == 0:
            commit_dir = '>' if self._start_above else '<'
            return_dir = '<' if self._start_above else '>'
            print(f"[Angle] raw={raw_angle:.1f}°  ema={self._ema_val:.1f}°  "
                  f"thresh={self._threshold:.1f}°  "
                  f"commit{commit_dir}{self._line_commit:.1f}°  "
                  f"return{return_dir}{self._line_return:.1f}°  "
                  f"in_rep={self.is_in_rep}  committed={self._committed}")

        features    = _extract_features(pose_frame)

        # Collect raw pose while a rep is in progress so the scorer can
        # compute joint angles.  _rep_frames stores y-features (for legacy),
        # _rep_raw_frames stores raw (33,3) poses (for angle-profile scoring).
        if self.is_in_rep:
            self._rep_raw_frames.append(pose_frame.copy())

        rep_counted = self._state_machine(self._ema_val, features)

        # current_form_score always reflects the last COMPLETED rep score.
        # Live per-frame scoring is removed — it was comparing individual frames
        # against the template at a guessed time index, producing noisy numbers
        # that flickered and didn't reflect real form quality. The completed-rep
        # score resamples the full path to match the template length first,
        # making the time alignment exact and the score meaningful.

        return rep_counted

    # ── State machine ──────────────────────────────────────────────────────────

    def _state_machine(self, val: float, features: np.ndarray) -> bool:
        rep_counted = False

        if self._start_above:
            in_rest_zone  = val < self._line_return
            crossed_start = val > self._threshold
            past_commit   = val > self._line_commit
        else:
            in_rest_zone  = val > self._line_return
            crossed_start = val < self._threshold
            past_commit   = val < self._line_commit

        if self._in_start_pos:
            if crossed_start:
                self._in_start_pos    = False
                self._committed       = False
                self._rep_frame_count = 0
                self.is_in_rep        = True
                self._rep_frames      = []
                self._rep_raw_frames  = []
                print(f"[RepCounter] → Rep started  ema={val:.1f}°")
        else:
            self._rep_frames.append(features)
            self._rep_frame_count += 1

            if past_commit and not self._committed:
                self._committed = True
                print(f"[RepCounter] ✓ Committed  ema={val:.1f}°  "
                      f"frame={self._rep_frame_count}")

            if in_rest_zone:
                if self._committed and self._rep_frame_count >= MIN_REP_FRAMES:
                    final_score             = self._score_completed_rep()
                    self.last_rep_score     = final_score
                    self.current_form_score = final_score
                    if self._frames_since_rep > MIN_REP_GAP_FRAMES:
                        self.rep_count        += 1
                        self._frames_since_rep = 0
                        rep_counted            = True
                        print(f"[RepCounter] ✅ REP #{self.rep_count}  "
                              f"frames={self._rep_frame_count}  score={final_score}")
                else:
                    reason = ("not committed — go deeper"
                              if not self._committed
                              else f"too short ({self._rep_frame_count} < {MIN_REP_FRAMES})")
                    print(f"[RepCounter] ↩ Cancelled — {reason}  ema={val:.1f}°")

                self._in_start_pos    = True
                self._committed       = False
                self.is_in_rep        = False
                self._rep_frames      = []
                self._rep_raw_frames  = []
                self._rep_frame_count = 0

        return rep_counted

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_completed_rep(self) -> int:
        """
        Score a completed rep using the angle-parameterized form profile.

        For each frame of the live rep, determine which angle bin it falls in,
        then measure how far EACH key joint deviates from the training mean at
        that bin in std units.  Per-joint scores are stored in
        self.last_rep_joint_scores so the UI can show which joint hurt the score.

        Bins the person never reached score 0 — naturally penalises insufficient
        range of motion.  Falls back to 50 when no form_profile exists.
        """
        profile = self.exercise.form_profile
        if profile is None or len(self._rep_frames) < 2:
            self.last_rep_joint_scores = {}
            self.last_rep_joint_depths = {}
            return self.last_rep_score if self.last_rep_score is not None else 50

        lo               = profile["lo"]
        hi               = profile["hi"]
        n_bins           = profile["n_bins"]
        bin_means        = profile["bin_means"]        # (n_bins, n_joints)
        bin_stds         = profile["bin_stds"]         # (n_bins, n_joints)
        bin_counts       = profile["bin_counts"]       # (n_bins,)
        key_triplets     = profile["key_triplets"]     # list of (a,b,c)
        key_joint_indices = profile["key_joint_indices"]  # landmark indices

        if not key_triplets or not self._rep_raw_frames:
            self.last_rep_joint_scores = {}
            self.last_rep_joint_depths = {}
            return 50

        n_joints = len(key_triplets)

        # Per-joint, per-bin deviation accumulation.
        # joint_bin_devs[j][b] = list of per-frame deviations for joint j in bin b.
        # Only bins actually visited by this live rep are populated — we never
        # append 0.0 for unvisited bins (that was the main scoring bug: it dragged
        # every joint's average down even when form was perfect on reached bins).
        joint_bin_devs = [[[] for _ in range(n_bins)] for _ in range(n_joints)]
        bins_visited   = set()

        for pose_frame in self._rep_raw_frames:
            ta = compute_angle(*self._tracking_triplet, pose_frame)
            if self._bilateral_triplet:
                ta = (ta + compute_angle(*self._bilateral_triplet, pose_frame)) / 2.0
            b = int((ta - lo) / (hi - lo) * n_bins)
            b = max(0, min(n_bins - 1, b))
            if bin_counts[b] == 0:
                continue

            bins_visited.add(b)
            key_angles = np.array([compute_angle(*t, pose_frame) for t in key_triplets])

            for j in range(n_joints):
                dev = abs(float(key_angles[j]) - float(bin_means[b][j])) / float(bin_stds[b][j])
                joint_bin_devs[j][b].append(dev)

        valid_bins = np.where(bin_counts > 0)[0]
        if len(valid_bins) == 0 or not bins_visited:
            self.last_rep_joint_scores = {}
            self.last_rep_joint_depths = {}
            return 50

        # Depth penalty: fraction of training range the person actually covered.
        # Applied to the OVERALL score only — not to individual joint scores,
        # because a shallow rep isn't the fault of any single joint.
        depth_pct   = len(bins_visited) / len(valid_bins)   # 0.0–1.0
        visited_arr = sorted(bins_visited)

        # Detect "inactive" joints — joints whose angle barely changes across the
        # full training range.  These are legitimate key joints for posture checking
        # but they produce meaningless scores because:
        #   (a) their training std collapses to the 5° floor, and
        #   (b) any tiny camera-angle difference causes a huge z-score.
        # Threshold: if the peak-to-trough mean angle across all training bins is
        # < INACTIVE_RANGE_DEG, mark the joint inactive and show "N/A" in the UI.
        INACTIVE_RANGE_DEG = 15.0
        joint_active = []
        for j in range(n_joints):
            filled = [b for b in valid_bins if bin_counts[b] > 0]
            if filled:
                means_j = [bin_means[b][j] for b in filled]
                joint_active.append((max(means_j) - min(means_j)) >= INACTIVE_RANGE_DEG)
            else:
                joint_active.append(False)

        # ── Per-joint form scores ─────────────────────────────────────────────
        # Score = average over VISITED bins only.  Depth is separate.
        joint_scores = {}
        joint_depths = {}

        for j, joint_idx in enumerate(key_joint_indices):
            if not joint_active[j]:
                # Mark inactive joints explicitly so the UI can show "N/A"
                joint_scores[joint_idx] = None
                joint_depths[joint_idx] = int(depth_pct * 100)
                continue

            visited_devs = []
            for b in visited_arr:
                if joint_bin_devs[j][b]:
                    visited_devs.append(np.mean(joint_bin_devs[j][b]))

            if not visited_devs:
                joint_scores[joint_idx] = None
                joint_depths[joint_idx] = int(depth_pct * 100)
                continue

            # Form quality on REACHED bins, then scale by depth coverage
            mean_dev     = np.mean(visited_devs)
            form_quality = max(0.0, 100.0 - max(0.0, mean_dev - SCORE_DEAD_ZONE_STD) * SCORE_PENALTY_RATE)
            j_score      = int(form_quality * depth_pct)
            joint_scores[joint_idx] = j_score
            joint_depths[joint_idx] = int(depth_pct * 100)

        # ── Overall score ─────────────────────────────────────────────────────
        # Average combined deviation over VISITED bins only, then apply depth.
        visited_bin_scores = []
        for b in visited_arr:
            combined_devs = []
            for j in range(n_joints):
                if joint_active[j] and joint_bin_devs[j][b]:
                    combined_devs.append(np.mean(joint_bin_devs[j][b]))
            if combined_devs:
                mean_dev_b = np.mean(combined_devs)
                s = max(0.0, 100.0 - max(0.0, mean_dev_b - SCORE_DEAD_ZONE_STD) * SCORE_PENALTY_RATE)
                visited_bin_scores.append(s)

        if not visited_bin_scores:
            self.last_rep_joint_scores = joint_scores
            self.last_rep_joint_depths = joint_depths
            return 50

        form_score = np.mean(visited_bin_scores)

        
        depth_penalty = depth_pct ** 2

        coverage_penalty = len(visited_arr) / max(len(valid_bins), 1)

        score = int(form_score * depth_penalty * coverage_penalty)

        score = max(20, min(95, score))
        
        self.last_rep_joint_scores = joint_scores
        self.last_rep_joint_depths = joint_depths

        active_scores = {k: v for k, v in joint_scores.items() if v is not None}
        print(f"[Score] angle-profile  score={score}  depth={int(depth_pct*100)}%  "
              f"frames={len(self._rep_raw_frames)}  "
              f"per_joint={active_scores}")
        return score

    # ── No detection / Reset ───────────────────────────────────────────────────

    def notify_no_detection(self):
        self._ema_val        = None
        self._last_raw_angle = None
        self._in_start_pos   = True
        self._committed      = False
        self.is_in_rep       = False
        self._rep_frames     = []
        self._rep_raw_frames = []

    def reset(self):
        self.rep_count              = 0
        self.current_form_score     = None
        self.last_rep_score         = None
        self.last_rep_joint_scores  = {}
        self.last_rep_joint_depths  = {}
        self.is_in_rep              = False
        self._in_start_pos          = True
        self._committed             = False
        self._rep_frame_count       = 0
        self._ema_val               = None
        self._last_raw_angle        = None
        self._frames_since_rep      = MIN_REP_GAP_FRAMES + 1
        self._rep_frames            = []
        self._rep_raw_frames        = []

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_pose(frame: np.ndarray) -> np.ndarray:
    hip_mid = (frame[23] + frame[24]) / 2.0
    sho_mid = (frame[11] + frame[12]) / 2.0
    scale   = np.linalg.norm(sho_mid - hip_mid)
    if scale < 1e-6:
        scale = 1e-6
    return (frame - hip_mid) / scale


def _extract_features(pose_frame: np.ndarray) -> np.ndarray:
    """
    y-only body-relative features. MUST match rep_trainer._extract_features_sequence.
    Returns shape (22,) — one y value per body landmark (indices 11–32).
    """
    norm = _normalise_pose(pose_frame)
    return np.array([float(norm[idx, 1]) for idx in range(11, 33)], dtype=np.float64)


def _resample(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if len(sequence) == target_length:
        return sequence
    indices   = np.linspace(0, len(sequence) - 1, target_length)
    resampled = np.array([
        np.interp(indices, np.arange(len(sequence)), sequence[:, i])
        for i in range(sequence.shape[1])
    ]).T
    return resampled
