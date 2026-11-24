"""
qevd_config_gen.py
------------------
Auto-generates config.json for every exercise folder that has
mistake_classifier.pkl but no config.json.

For exercises that already have config.json (e.g. squats recorded via builder),
it just adds the QEVD fields without touching anything else.

Usage:
    python qevd_config_gen.py
    python qevd_config_gen.py --data_dir exercises_data
"""

import os
import json
import argparse
import joblib

# ── Exercise metadata ─────────────────────────────────────────────────────────
# Display name, muscle groups, description for each QEVD exercise folder name.
# Add entries here if you add more exercises later.

EXERCISE_META = {
    "squats": {
        "name":          "Squats",
        "muscle_groups": ["Quadriceps", "Glutes", "Hamstrings"],
        "description":   "Stand with feet shoulder-width apart, lower hips until thighs are parallel to the floor, then return to standing.",
    },
    "pushups": {
        "name":          "Push-ups",
        "muscle_groups": ["Chest", "Triceps", "Shoulders"],
        "description":   "Start in a plank position, lower your chest to the floor keeping your body straight, then push back up.",
    },
    "glutes_bridge": {
        "name":          "Glute Bridge",
        "muscle_groups": ["Glutes", "Hamstrings", "Core"],
        "description":   "Lie on your back with knees bent, drive hips upward until your body forms a straight line from shoulders to knees.",
    },
    "high_knees": {
        "name":          "High Knees",
        "muscle_groups": ["Hip Flexors", "Quadriceps", "Core"],
        "description":   "Run in place driving knees up to hip height alternately, keeping a quick pace.",
    },
    "jumping_jacks": {
        "name":          "Jumping Jacks",
        "muscle_groups": ["Full Body", "Cardiovascular"],
        "description":   "Jump feet out wide while raising arms overhead, then jump feet back together while lowering arms.",
    },
    "alternating_forward_lunges": {
        "name":          "Alternating Forward Lunges",
        "muscle_groups": ["Quadriceps", "Glutes", "Hamstrings"],
        "description":   "Step forward alternating legs, lowering back knee toward the floor while keeping front knee over ankle.",
    },
    "kickback_right_leg": {
        "name":          "Glute Kickback (Right Leg)",
        "muscle_groups": ["Glutes", "Hamstrings"],
        "description":   "On hands and knees, extend right leg backward and upward squeezing the glute at the top.",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="exercises_data")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"❌ {args.data_dir} not found")
        return

    updated = 0
    created = 0

    for folder_name in sorted(os.listdir(args.data_dir)):
        folder_path = os.path.join(args.data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        classifier_path = os.path.join(folder_path, "mistake_classifier.pkl")
        if not os.path.exists(classifier_path):
            continue

        config_path = os.path.join(folder_path, "config.json")

        # ── Load mistake types from classifier ────────────────────────────────
        try:
            bundle      = joblib.load(classifier_path)
            label_names = bundle.get("label_names", [])
            n_mistakes  = len(label_names) - 1  # exclude "correct"
        except Exception as e:
            print(f"⚠️  {folder_name}: could not load classifier: {e}")
            continue

        # ── Get display metadata ──────────────────────────────────────────────
        meta = EXERCISE_META.get(folder_name)
        if meta is None:
            # Fallback: humanise the folder name
            display_name = folder_name.replace("_", " ").title()
            meta = {
                "name":          display_name,
                "muscle_groups": [],
                "description":   "",
            }
            print(f"⚠️  {folder_name}: no metadata defined — using '{display_name}'")

        # ── Load or create config ─────────────────────────────────────────────
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            action = "updated"
            updated += 1
        else:
            cfg    = {}
            action = "created"
            created += 1

        # ── Set / update fields ───────────────────────────────────────────────
        cfg.setdefault("name",          meta["name"])
        cfg.setdefault("muscle_groups", meta["muscle_groups"])
        cfg.setdefault("description",   meta["description"])
        cfg.setdefault("key_joints",    [])
        cfg.setdefault("trained",       False)   # rep-counting not trained yet

        # QEVD fields — always overwrite with latest
        cfg["mistake_classifier_trained"] = True
        cfg["mistake_types"]              = label_names
        cfg["n_mistake_types"]            = n_mistakes

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        print(f"✅ {action}: {folder_name}  →  '{cfg['name']}'  "
              f"({n_mistakes} mistake types)")

    print(f"\n{'='*50}")
    print(f"Created: {created}  |  Updated: {updated}")
    print(f"\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()