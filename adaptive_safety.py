"""
adaptive_safety.py
------------------
Injury-prevention and progressive overload engine.

WHAT IT DOES
------------
1. Reads performance_log.json (written by app.py every rep)
2. Computes per-exercise safety levels based on recent form scores
3. Adjusts THREE values in exercises_data/<ex>/config.json:
     adaptive_rep_target  — how many reps to do next session
     threshold_override   — how deep into the movement to go (ROM depth)
     deadband_override    — hysteresis band around the threshold
4. Provides live session monitoring — warns when form is degrading mid-set

SAFETY LEVELS
-------------
  LEVEL 0 — No data yet
    Rep target: beginner default (5-8 reps)
    ROM:        60% of full trained range (shallow = safe)

  LEVEL 1 — Learning (avg form < 60%)
    Rep target: reduce by 2, min 3
    ROM:        50% of range — force shallow reps to build motor pattern
    Warning:    "Focus on form before adding reps"

  LEVEL 2 — Building (avg form 60-74%)
    Rep target: hold current
    ROM:        70% of range — slightly deeper, still protected
    Warning:    None unless sudden drop

  LEVEL 3 — Consistent (avg form 75-84%)
    Rep target: +2 reps every 3 sessions at this level
    ROM:        85% of range — near full depth
    Warning:    None

  LEVEL 4 — Performing (avg form 85%+, 20+ reps)
    Rep target: +3 reps or suggest adding load
    ROM:        100% — full trained range
    Warning:    None

LIVE SESSION MONITORING
-----------------------
  Call check_session_safety() every rep.
  If the last 5 reps average < 55%, return a STOP warning.
  If the last 3 reps are each lower than the one before → fatigue detected.
"""

import json
import os
import math

# ── Constants ──────────────────────────────────────────────────────────────────

PERF_PATH    = "performance_log.json"
EXERCISES_DIR = "exercises_data"

# How many recent reps to use for level assessment
WINDOW_REPS = 20

# Rep target bounds
MIN_REPS     = 3
MAX_REPS     = 40
DEFAULT_REPS = {
    "beginner":     6,
    "intermediate": 10,
    "advanced":     15,
}

# ROM fraction per level (fraction of lo→hi range used as depth)
ROM_FRACTION = {
    0: 0.60,   # no data — shallow start
    1: 0.50,   # learning — force shallow
    2: 0.70,   # building
    3: 0.85,   # consistent
    4: 1.00,   # performing — full range
}

# Live safety: stop warning threshold
LIVE_STOP_THRESHOLD  = 55   # avg of last 5 reps below this → stop warning
LIVE_WARN_THRESHOLD  = 65   # avg of last 5 reps below this → slow down warning
FATIGUE_WINDOW       = 3    # consecutive reps each worse than last → fatigue


# ── Core assessment ────────────────────────────────────────────────────────────

def _load_perf():
    if os.path.exists(PERF_PATH):
        with open(PERF_PATH) as f:
            return json.load(f)
    return {}


def assess_exercise(exercise_name: str, fitness_level: str = "beginner") -> dict:
    """
    Assess current safety level and compute adaptive values for one exercise.

    Returns
    -------
    dict:
        level          (int)    0-4
        level_name     (str)    human label
        avg_form       (float)  recent average
        rep_target     (int)    recommended reps for next session
        rom_fraction   (float)  0-1 fraction of full ROM to use
        message        (str)    user-facing explanation
        can_progress   (bool)   True if ready to increase difficulty
        needs_focus    (bool)   True if form needs work
        n_reps         (int)    total reps logged for this exercise
    """
    perf  = _load_perf()
    reps  = perf.get(exercise_name, [])
    n     = len(reps)

    # ── No data ───────────────────────────────────────────────────────────────
    if n == 0:
        default_target = DEFAULT_REPS.get(fitness_level.lower(), 8)
        return {
            "level":        0,
            "level_name":   "New",
            "avg_form":     None,
            "rep_target":   default_target,
            "rom_fraction": ROM_FRACTION[0],
            "message":      f"Starting fresh — {default_target} reps at reduced depth to build your pattern safely.",
            "can_progress": False,
            "needs_focus":  False,
            "n_reps":       0,
        }

    # ── Compute recent average ────────────────────────────────────────────────
    recent       = reps[-WINDOW_REPS:]
    avg          = sum(r["score"] for r in recent) / len(recent)
    current_target = _get_current_target(exercise_name, fitness_level)

    # ── Determine level ───────────────────────────────────────────────────────
    if   avg < 60:   level = 1
    elif avg < 75:   level = 2
    elif avg < 85:   level = 3
    else:            level = 4 if n >= 20 else 3

    level_names = {1:"Learning", 2:"Building", 3:"Consistent", 4:"Performing"}

    # ── Compute rep target ────────────────────────────────────────────────────
    if   level == 1: new_target = max(MIN_REPS, current_target - 2)
    elif level == 2: new_target = current_target
    elif level == 3:
        # Progress every 3 sessions at this level
        sessions_at_level = _sessions_at_level(exercise_name, level=3, min_avg=75)
        new_target = current_target + (2 if sessions_at_level % 3 == 0 and sessions_at_level > 0 else 0)
    else:            # level 4
        new_target = min(current_target + 3, MAX_REPS)

    new_target = max(MIN_REPS, min(MAX_REPS, new_target))

    # ── Message ───────────────────────────────────────────────────────────────
    messages = {
        1: f"Form needs work ({avg:.0f}% avg). Reduced to {new_target} reps at shallow depth — focus on control.",
        2: f"Building ({avg:.0f}% avg). Holding at {new_target} reps, 70% depth — keep steady.",
        3: f"Consistent form ({avg:.0f}%). Progressing to {new_target} reps at near-full depth.",
        4: f"Excellent form ({avg:.0f}%)! Full range at {new_target} reps. Consider adding resistance.",
    }

    return {
        "level":        level,
        "level_name":   level_names[level],
        "avg_form":     round(avg, 1),
        "rep_target":   new_target,
        "rom_fraction": ROM_FRACTION[level],
        "message":      messages[level],
        "can_progress": level >= 3,
        "needs_focus":  level <= 1,
        "n_reps":       n,
    }


def _get_current_target(exercise_name: str, fitness_level: str) -> int:
    """Read last saved adaptive_rep_target from config, or use default."""
    safe   = exercise_name.replace(" ", "_").replace("/","_").lower()
    cfg_p  = os.path.join(EXERCISES_DIR, safe, "config.json")
    if os.path.exists(cfg_p):
        with open(cfg_p) as f:
            cfg = json.load(f)
        return cfg.get("adaptive_rep_target",
               cfg.get("target_reps",
               DEFAULT_REPS.get(fitness_level.lower(), 8)))
    return DEFAULT_REPS.get(fitness_level.lower(), 8)


def _sessions_at_level(exercise_name: str, level: int, min_avg: float) -> int:
    """Count how many past sessions had avg_form >= min_avg."""
    perf  = _load_perf()
    reps  = perf.get(exercise_name, [])
    if not reps:
        return 0
    # Group by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for r in reps:
        by_date[r.get("date","")].append(r["score"])
    count = 0
    for scores in by_date.values():
        if scores and (sum(scores)/len(scores)) >= min_avg:
            count += 1
    return count


# ── ROM threshold adjustment ───────────────────────────────────────────────────

def compute_threshold_override(config: dict, rom_fraction: float) -> float | None:
    """
    Compute a safe threshold_override based on the trained range and ROM fraction.

    The trained range is [lo, hi] stored in config.json.
    - hi  = top of movement (standing upright for squats ~170°)
    - lo  = bottom of movement (deep squat ~90°)
    - threshold = midpoint between hi and lo

    At rom_fraction=1.0 → threshold = trained threshold (full depth)
    At rom_fraction=0.5 → threshold pulled toward hi (shallower movement)

    For start_above exercises (e.g. shoulder press starts at bottom):
    Logic is flipped — threshold pulled toward lo.
    """
    lo           = config.get("lo")
    hi           = config.get("hi")
    trained_thr  = config.get("threshold_auto")

    if lo is None or hi is None or trained_thr is None:
        return None   # exercise not rep-trained, can't adjust

    full_range   = abs(hi - lo)
    start_above  = config.get("start_above", False)

    if start_above:
        # Movement starts near lo, goes toward hi
        # Full depth = threshold at trained value (midpoint)
        # Reduced depth = threshold closer to lo (less travel)
        safe_threshold = lo + (full_range * rom_fraction * 0.5)
    else:
        # Movement starts near hi (standing), goes toward lo (deep)
        # Full depth = threshold at trained value
        # Reduced depth = threshold closer to hi (shallower squat)
        safe_threshold = hi - (full_range * rom_fraction * 0.5)

    return round(safe_threshold, 1)


# ── Apply adaptive values to config.json ──────────────────────────────────────

def apply_to_config(exercise_name: str, assessment: dict) -> bool:
    """
    Write adaptive_rep_target and (if applicable) threshold_override
    to the exercise's config.json.

    Returns True if config was updated.
    """
    safe  = exercise_name.replace(" ", "_").replace("/","_").lower()
    cfg_p = os.path.join(EXERCISES_DIR, safe, "config.json")

    if not os.path.exists(cfg_p):
        return False

    with open(cfg_p) as f:
        cfg = json.load(f)

    cfg["adaptive_rep_target"] = assessment["rep_target"]
    cfg["adaptive_level"]      = assessment["level"]
    cfg["adaptive_level_name"] = assessment["level_name"]
    cfg["adaptive_rom_fraction"] = assessment["rom_fraction"]

    # Only adjust threshold if exercise has rep training data
    if "threshold_auto" in cfg:
        override = compute_threshold_override(cfg, assessment["rom_fraction"])
        if override is not None:
            cfg["threshold_override"] = override
            # Keep deadband proportional to ROM
            if "deadband_auto" in cfg:
                cfg["deadband_override"] = round(
                    cfg["deadband_auto"] * assessment["rom_fraction"], 1
                )

    with open(cfg_p, "w") as f:
        json.dump(cfg, f, indent=2)

    return True


# ── Full recalculation for all exercises ──────────────────────────────────────

def recalculate_all(fitness_level: str = "beginner") -> dict:
    """
    Recalculate and apply adaptive values for every exercise that has data.
    Call this at app startup or after a session ends.

    Returns {exercise_name: assessment_dict}
    """
    results = {}
    if not os.path.isdir(EXERCISES_DIR):
        return results

    for folder in sorted(os.listdir(EXERCISES_DIR)):
        cfg_p = os.path.join(EXERCISES_DIR, folder, "config.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        name = cfg.get("name", folder.replace("_"," "))
        a    = assess_exercise(name, fitness_level)
        apply_to_config(name, a)
        results[name] = a

    return results


# ── Live session safety monitor ───────────────────────────────────────────────

class SessionSafetyMonitor:
    """
    Call .check(form_score) after each rep.
    Returns a safety event dict or None.

    Safety events:
        {"level": "stop",  "message": "..."}  — stop immediately
        {"level": "warn",  "message": "..."}  — slow down / rest
        {"level": "fatigue","message": "..."} — fatigue pattern detected
        None                                   — all good
    """

    def __init__(self, exercise_name: str):
        self.exercise   = exercise_name
        self._scores    = []
        self._warned    = False
        self._stopped   = False

    def check(self, form_score: int) -> dict | None:
        self._scores.append(form_score)

        if len(self._scores) < 3:
            return None

        last5 = self._scores[-5:]
        avg5  = sum(last5) / len(last5)

        # ── STOP: form collapsed ─────────────────────────────────────────────
        if avg5 < LIVE_STOP_THRESHOLD and not self._stopped and len(self._scores) >= 5:
            self._stopped = True
            return {
                "level":   "stop",
                "message": f"⛔ Stop — form has dropped to {avg5:.0f}% over the last 5 reps. "
                           f"Rest before continuing to avoid injury.",
            }

        # ── WARN: form degrading ─────────────────────────────────────────────
        if avg5 < LIVE_WARN_THRESHOLD and not self._warned:
            self._warned = True
            return {
                "level":   "warn",
                "message": f"⚠️ Form dropping ({avg5:.0f}% avg). Slow down and focus on control.",
            }

        # ── FATIGUE: 3 consecutive drops ────────────────────────────────────
        if len(self._scores) >= FATIGUE_WINDOW + 1:
            tail = self._scores[-FATIGUE_WINDOW-1:]
            if all(tail[i] > tail[i+1] for i in range(FATIGUE_WINDOW)):
                return {
                    "level":   "fatigue",
                    "message": f"😓 Fatigue detected — each rep is getting worse. "
                               f"Take a 60-second rest.",
                }

        # ── Recovery: reset warnings if form improved ────────────────────────
        if avg5 >= LIVE_WARN_THRESHOLD + 10:
            self._warned  = False
            self._stopped = False

        return None

    @property
    def rolling_avg(self) -> float | None:
        if not self._scores:
            return None
        return round(sum(self._scores[-5:]) / len(self._scores[-5:]), 1)

    def reset(self):
        self._scores  = []
        self._warned  = False
        self._stopped = False


# ── CLI: inspect current adaptive state ───────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Show adaptive safety state")
    parser.add_argument("--level", default="beginner",
                        choices=["beginner","intermediate","advanced"])
    parser.add_argument("--apply", action="store_true",
                        help="Write adaptive values to config.json files")
    args = parser.parse_args()

    results = recalculate_all(args.level) if args.apply else None

    perf = _load_perf()
    if not perf:
        print("No performance data yet. Complete some workouts first.")
        exit()

    if not results:
        # Just show, don't apply
        results = {}
        if os.path.isdir(EXERCISES_DIR):
            for folder in sorted(os.listdir(EXERCISES_DIR)):
                cfg_p = os.path.join(EXERCISES_DIR, folder, "config.json")
                if not os.path.exists(cfg_p):
                    continue
                with open(cfg_p) as f:
                    cfg = json.load(f)
                name = cfg.get("name", folder)
                results[name] = assess_exercise(name, args.level)

    print(f"\n{'='*70}")
    print(f"{'Exercise':<30} {'Level':<12} {'Avg':<8} {'Target':<8} {'ROM'}")
    print(f"{'='*70}")
    for name, a in results.items():
        avg_str = f"{a['avg_form']:.0f}%" if a["avg_form"] else "—"
        print(f"{name:<30} {a['level_name']:<12} {avg_str:<8} "
              f"{a['rep_target']:<8} {int(a['rom_fraction']*100)}%")
        print(f"  → {a['message']}")
    print()

    if args.apply:
        print("✅ Adaptive values written to config.json files")
    else:
        print("Run with --apply to write values to config.json files")
