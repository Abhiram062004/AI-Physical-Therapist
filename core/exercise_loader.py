"""
core/exercise_loader.py
-----------------------
Scans the exercises_data/ folder and loads every trained exercise.
No hardcoding — adding a new exercise just means running the builder.

QEVD integration: if mistake_classifier.pkl exists in the exercise folder
(written by either the builder's optional Step 6 or the standalone
qevd_trainer.py), it is loaded automatically and exposed via has_qevd,
mistake_classifier, and n_mistake_types.
"""

import os
import json
import numpy as np
import joblib

EXERCISES_DATA_DIR = "exercises_data"


class LoadedExercise:
    """
    Represents a fully loaded exercise with its config,
    trained form model, rep pattern, and optional QEVD mistake classifier.
    """

    def __init__(self, folder_path: str):
        self.folder              = folder_path
        self.config              = {}
        self.form_stats          = None
        self.svm_model           = None
        self.rep_pattern         = None
        self.form_profile        = None   # angle-parameterized scoring profile
        self.mistake_classifier  = None   # QEVD multi-class SVC bundle (optional)

        self._load(folder_path)

    def _load(self, folder_path: str):
        config_path     = os.path.join(folder_path, "config.json")
        stats_path      = os.path.join(folder_path, "form_stats.npy")
        svm_path        = os.path.join(folder_path, "svm_model.pkl")
        rep_path        = os.path.join(folder_path, "rep_pattern.npy")
        profile_path    = os.path.join(folder_path, "form_profile.npy")
        classifier_path = os.path.join(folder_path, "mistake_classifier.pkl")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {folder_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        if os.path.exists(stats_path):
            self.form_stats = np.load(stats_path, allow_pickle=True).item()

        if os.path.exists(svm_path):
            self.svm_model = joblib.load(svm_path)

        if os.path.exists(rep_path):
            self.rep_pattern = np.load(rep_path, allow_pickle=True)

        if os.path.exists(profile_path):
            self.form_profile = np.load(profile_path, allow_pickle=True).item()

        if os.path.exists(classifier_path):
            try:
                self.mistake_classifier = joblib.load(classifier_path)
            except Exception as e:
                print(f"⚠️  Could not load mistake_classifier.pkl in {folder_path}: {e}")

    # ── Core properties ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.config.get("name", "Unknown")

    @property
    def is_trained(self) -> bool:
        return self.config.get("trained", False)

    @property
    def key_joints(self) -> list:
        return self.config.get("key_joints", [])

    @property
    def key_joint_indices(self) -> list:
        return [j["landmark_index"] for j in self.key_joints]

    @property
    def muscle_groups(self) -> list:
        return self.config.get("muscle_groups", [])

    @property
    def description(self) -> str:
        return self.config.get("description", "")

    # ── QEVD / mistake-classifier properties ──────────────────────────────────

    @property
    def has_qevd(self) -> bool:
        """True if a QEVD mistake classifier has been trained for this exercise."""
        return self.mistake_classifier is not None

    @property
    def n_mistake_types(self) -> int:
        """Number of distinct mistake classes (excluding 'correct')."""
        if self.mistake_classifier is None:
            return 0
        # Prefer the value in config (written at train time); fall back to
        # counting label_names minus the 'correct' class.
        if "n_mistake_types" in self.config:
            return self.config["n_mistake_types"]
        labels = self.mistake_classifier.get("label_names", [])
        return max(0, len(labels) - 1)

    @property
    def mistake_types(self) -> list:
        """List of all label names including 'correct' (index 0)."""
        if self.mistake_classifier is None:
            return []
        return self.mistake_classifier.get("label_names", [])

    # ── Feedback helpers ──────────────────────────────────────────────────────

    def feedback_for_joint(self, landmark_index: int) -> list:
        """Returns list of feedback messages for this joint (may be empty)."""
        for j in self.key_joints:
            if j["landmark_index"] == landmark_index:
                if "feedback_messages" in j:
                    return j["feedback_messages"]
                elif "feedback_message" in j and j["feedback_message"]:
                    return [j["feedback_message"]]
        return []

    def __repr__(self):
        qevd_str = f", {self.n_mistake_types} mistakes" if self.has_qevd else ""
        status   = "trained" if self.is_trained else "untrained"
        return f"LoadedExercise({self.name}, {status}{qevd_str})"


class ExerciseLoader:
    """
    Scans exercises_data/ and returns all loadable exercises.
    Call .load() once at app startup.
    """

    def __init__(self, data_dir: str = EXERCISES_DATA_DIR):
        self.data_dir  = data_dir
        self._exercises: dict[str, LoadedExercise] = {}

    def load(self) -> dict:
        """Load all exercises. Returns {name: LoadedExercise}."""
        self._exercises = {}

        if not os.path.exists(self.data_dir):
            print(f"⚠️  exercises_data/ folder not found at {self.data_dir}")
            return {}

        for folder_name in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            config_path = os.path.join(folder_path, "config.json")
            if not os.path.exists(config_path):
                continue

            try:
                ex = LoadedExercise(folder_path)
                qevd_str = f" + QEVD({ex.n_mistake_types})" if ex.has_qevd else ""
                print(f"✅ Loaded: {ex.name} "
                      f"({'trained' if ex.is_trained else 'untrained'}{qevd_str})")
                self._exercises[ex.name] = ex
            except Exception as e:
                print(f"⚠️  Could not load {folder_name}: {e}")

        print(f"\n📦 Total exercises loaded: {len(self._exercises)}")
        return self._exercises

    def get(self, name: str) -> LoadedExercise | None:
        return self._exercises.get(name)

    def names(self) -> list:
        return list(self._exercises.keys())

    def trained_only(self) -> dict:
        return {k: v for k, v in self._exercises.items() if v.is_trained}

    def __len__(self):
        return len(self._exercises)


# ── Singleton for use across the app ─────────────────────────────────────────
loader = ExerciseLoader()
