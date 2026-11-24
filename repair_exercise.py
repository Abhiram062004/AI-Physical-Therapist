"""
repair_exercise.py
------------------
Run this once to fix a trained exercise config without retraining.

Usage:
    python repair_exercise.py
    python repair_exercise.py --exercise "Bicep Curl"
    python repair_exercise.py --list
"""

import json
import os
import argparse
import glob

EXERCISES_DATA_DIR = "exercises_data"


def list_exercises():
    configs = glob.glob(os.path.join(EXERCISES_DATA_DIR, "*", "config.json"))
    if not configs:
        print("No exercises found in exercises_data/")
        return
    for c in configs:
        with open(c) as f:
            cfg = json.load(f)
        print(f"  {cfg.get('name', '?')}  —  {c}")
        print(f"    threshold_auto={cfg.get('threshold_auto','?')}  "
              f"threshold_override={cfg.get('threshold_override','(none)')}  "
              f"deadband_auto={cfg.get('deadband_auto','?')}  "
              f"deadband_override={cfg.get('deadband_override','(none)')}")


def repair(exercise_name: str):
    # Find config
    slug = exercise_name.lower().replace(" ", "_")
    config_path = os.path.join(EXERCISES_DATA_DIR, slug, "config.json")
    if not os.path.exists(config_path):
        # Try searching all folders
        matches = glob.glob(os.path.join(EXERCISES_DATA_DIR, "*", "config.json"))
        for m in matches:
            with open(m) as f:
                cfg = json.load(f)
            if cfg.get("name", "").lower() == exercise_name.lower():
                config_path = m
                break
        else:
            print(f"Could not find config for '{exercise_name}'")
            return

    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\n=== Repairing: {cfg.get('name')} ===")
    print(f"Config: {config_path}\n")

    changed = []

    # ── 1. Remove bad threshold_override / deadband_override ─────────────────
    # These keys were written by the builder sliders and override the auto values.
    # If the auto values are correct but overrides are wrong, remove the overrides.
    if "threshold_override" in cfg:
        auto = cfg.get("threshold_auto", None)
        override = cfg["threshold_override"]
        print(f"  threshold_override={override}  threshold_auto={auto}")
        del cfg["threshold_override"]
        changed.append(f"Removed threshold_override ({override}) → will use threshold_auto ({auto})")

    if "deadband_override" in cfg:
        auto = cfg.get("deadband_auto", None)
        override = cfg["deadband_override"]
        print(f"  deadband_override={override}  deadband_auto={auto}")
        del cfg["deadband_override"]
        changed.append(f"Removed deadband_override ({override}) → will use deadband_auto ({auto})")

    # ── 2. Fix deadband_auto if it's too small (< 8°) ────────────────────────
    deadband_auto = cfg.get("deadband_auto", None)
    if deadband_auto is not None and deadband_auto < 8.0:
        lo = cfg.get("lo", None)
        hi = cfg.get("hi", None)
        if lo is not None and hi is not None:
            new_deadband = max((hi - lo) * 0.15, 8.0)
        else:
            new_deadband = max(deadband_auto * 3, 8.0)
        cfg["deadband_auto"] = round(new_deadband, 2)
        changed.append(f"Fixed deadband_auto: {deadband_auto:.2f}° → {new_deadband:.2f}°")

    # ── 3. Fix threshold_auto if it doesn't bisect lo/hi correctly ───────────
    lo = cfg.get("lo", None)
    hi = cfg.get("hi", None)
    threshold_auto = cfg.get("threshold_auto", None)
    if lo is not None and hi is not None and threshold_auto is not None:
        correct_threshold = (lo + hi) / 2.0
        if abs(correct_threshold - threshold_auto) > 5:
            cfg["threshold_auto"] = round(correct_threshold, 2)
            changed.append(
                f"Fixed threshold_auto: {threshold_auto:.2f}° → {correct_threshold:.2f}° "
                f"(midpoint of lo={lo}° hi={hi}°)"
            )

    # ── 4. Add lo/hi if missing ───────────────────────────────────────────────
    if "lo" not in cfg or "hi" not in cfg:
        # Estimate from threshold
        t = cfg.get("threshold_auto", 90)
        cfg["lo"] = round(t - 60, 1)
        cfg["hi"] = round(t + 60, 1)
        changed.append(f"Added estimated lo={cfg['lo']}  hi={cfg['hi']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not changed:
        print("  Nothing to fix — config looks correct.")
        return

    print("\n  Changes:")
    for c in changed:
        print(f"    ✓ {c}")

    print(f"\n  Final values:")
    print(f"    threshold_auto = {cfg.get('threshold_auto')}")
    print(f"    deadband_auto  = {cfg.get('deadband_auto')}")
    print(f"    lo             = {cfg.get('lo')}")
    print(f"    hi             = {cfg.get('hi')}")
    print(f"    start_above    = {cfg.get('start_above')}")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  ✅ Saved to {config_path}")
    print("  Restart the app for changes to take effect.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", default=None, help="Exercise name to repair")
    parser.add_argument("--list", action="store_true", help="List all exercises and their config values")
    parser.add_argument("--all", action="store_true", help="Repair all exercises")
    args = parser.parse_args()

    if args.list:
        list_exercises()
    elif args.all:
        configs = glob.glob(os.path.join(EXERCISES_DATA_DIR, "*", "config.json"))
        for c in configs:
            with open(c) as f:
                cfg = json.load(f)
            repair(cfg.get("name", ""))
    elif args.exercise:
        repair(args.exercise)
    else:
        # Default: repair all
        configs = glob.glob(os.path.join(EXERCISES_DATA_DIR, "*", "config.json"))
        if not configs:
            print("No exercises found. Run with --list to check.")
        else:
            for c in configs:
                with open(c) as f:
                    cfg = json.load(f)
                repair(cfg.get("name", ""))
