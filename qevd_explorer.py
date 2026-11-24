"""
qevd_explorer.py
----------------
Run this FIRST to see every exercise in the QEVD dataset,
how many clips each has, and what mistake types exist.

Use the output to pick your 10 exercises for training.

Usage:
    python qevd_explorer.py --qevd_part4 D:/QEVD/QEVD-FIT-300k-Part-4
    python qevd_explorer.py --qevd_part4 D:/QEVD/QEVD-FIT-300k-Part-4 --split train
    python qevd_explorer.py --qevd_part4 D:/QEVD/QEVD-FIT-300k-Part-4 --show_mistakes squat
"""

import json
import argparse
from collections import defaultdict


def load_json(part4_path: str) -> list:
    json_path = f"{part4_path}/fine_grained_labels.json"
    with open(json_path) as f:
        return json.load(f)


def parse_label(label: str):
    """Split 'exercise - mistake' into (exercise, mistake)."""
    if " - " in label:
        parts = label.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return label.strip(), None


def main():
    parser = argparse.ArgumentParser(description="Explore QEVD dataset contents")
    parser.add_argument("--qevd_part4", required=True,
                        help="Path to QEVD-FIT-300k-Part-4 folder")
    parser.add_argument("--split", default=None,
                        help="Filter by split: train / val / test (default: all)")
    parser.add_argument("--show_mistakes", default=None,
                        help="Show all mistake types for a specific exercise")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N exercises by clip count")
    args = parser.parse_args()

    data = load_json(args.qevd_part4)

    # ── Filter by split ──────────────────────────────────────────────────────
    if args.split:
        data = [e for e in data if e.get("split") == args.split]
        print(f"Filtered to split='{args.split}': {len(data)} entries\n")
    else:
        print(f"Total entries (all splits): {len(data)}\n")

    # ── Aggregate by exercise ────────────────────────────────────────────────
    exercise_clips   = defaultdict(int)
    exercise_mistakes = defaultdict(set)
    exercise_splits  = defaultdict(set)

    for entry in data:
        if not entry.get("labels"):
            continue
        label = entry["labels"][0]
        exercise, mistake = parse_label(label)
        exercise_clips[exercise] += 1
        if mistake:
            exercise_mistakes[exercise].add(mistake)
        exercise_splits[exercise].add(entry.get("split", "?"))

    # ── Show all mistakes for a specific exercise ────────────────────────────
    if args.show_mistakes:
        target = args.show_mistakes.lower()
        matches = {k: v for k, v in exercise_mistakes.items()
                   if target in k.lower()}
        if not matches:
            print(f"❌ No exercise matching '{args.show_mistakes}' found.")
            print(f"   Try one of: {sorted(exercise_clips.keys())[:10]}")
            return

        for ex, mistakes in matches.items():
            print(f"\n{'='*60}")
            print(f"Exercise: {ex}  ({exercise_clips[ex]} clips)")
            print(f"{'='*60}")
            for i, m in enumerate(sorted(mistakes), 1):
                print(f"  {i:3}. {m}")
        return

    # ── Full exercise list ───────────────────────────────────────────────────
    sorted_exercises = sorted(exercise_clips.items(), key=lambda x: -x[1])

    if args.top:
        sorted_exercises = sorted_exercises[:args.top]

    print(f"{'#':>3}  {'Exercise':<45} {'Clips':>6}  {'Mistakes':>8}  Splits")
    print("-" * 80)

    for i, (exercise, count) in enumerate(sorted_exercises, 1):
        n_mistakes = len(exercise_mistakes[exercise])
        splits     = "/".join(sorted(exercise_splits[exercise]))
        print(f"{i:3}.  {exercise:<45} {count:6}  {n_mistakes:8}  {splits}")

    print(f"\n{'='*80}")
    print(f"Total exercises: {len(exercise_clips)}")
    print(f"Total clips:     {sum(exercise_clips.values())}")
    print(f"\nTo see mistakes for a specific exercise:")
    print(f"  python qevd_explorer.py --qevd_part4 <path> --show_mistakes \"squat\"")
    print(f"\nOnce you've picked your 10, run qevd_ingestor.py:")
    print(f"  python qevd_ingestor.py --qevd_root D:/QEVD --exercises \"squat\" \"push up\" ...")


if __name__ == "__main__":
    main()