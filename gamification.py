"""
gamification.py
---------------
Tracks session stats and awards badges.

Changes:
  - Removed record_form() — it was being called every frame and diluting
    avg_form with thousands of per-frame readings instead of per-rep scores.
  - avg_form now purely reflects completed rep scores, which is meaningful.
  - rep_score_sum / rep_score_count are separate from any live display.
"""

import json
import os
from datetime import date

HISTORY_FILE = "session_history.json"

BADGES = {
    "First Rep":   {"desc": "Complete your first rep",              "icon": "🎯"},
    "Warming Up":  {"desc": "Complete 10 reps in a session",        "icon": "🔥"},
    "On a Roll":   {"desc": "Complete 25 reps in a session",        "icon": "💪"},
    "Century Club":{"desc": "Complete 100 total reps lifetime",     "icon": "💯"},
    "Consistency": {"desc": "Work out 3 days in a row",             "icon": "📅"},
    "Form Master": {"desc": "Maintain ≥80 form score for 20 reps",  "icon": "⭐"},
    "Perfect Set": {"desc": "10 consecutive reps with 100% form",   "icon": "🏆"},
    "Iron Will":   {"desc": "Complete 500 total reps lifetime",     "icon": "🦾"},
}


class SessionTracker:
    def __init__(self):
        self.session_reps      = 0
        self.rep_score_sum     = 0.0   # sum of completed rep scores only
        self.rep_score_count   = 0     # number of completed reps with a score
        self.high_form_streak  = 0
        self.perfect_streak    = 0
        self.new_badges        = []
        self.history           = self._load_history()

    # ── Per-rep update ────────────────────────────────────────────────────────
    def record_rep(self, form_score: int):
        """Call once per completed rep with its form score (0-100)."""
        self.session_reps    += 1
        self.rep_score_sum   += form_score
        self.rep_score_count += 1

        if form_score >= 80:
            self.high_form_streak += 1
        else:
            self.high_form_streak = 0

        if form_score == 100:
            self.perfect_streak += 1
        else:
            self.perfect_streak = 0

        self._check_badges()

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def avg_form(self) -> float:
        """Average form score across all completed reps this session."""
        if self.rep_score_count == 0:
            return 0.0
        return round(self.rep_score_sum / self.rep_score_count, 1)

    @property
    def lifetime_reps(self) -> int:
        return sum(s.get("reps", 0) for s in self.history) + self.session_reps

    @property
    def earned_badges(self) -> list:
        all_badges = set()
        for s in self.history:
            all_badges.update(s.get("badges", []))
        all_badges.update(self.new_badges)
        return list(all_badges)

    # ── Gamification helpers ──────────────────────────────────────────────────
    def progress_to_next_milestone(self) -> dict:
        milestones = [10, 25, 50, 100, 200, 500]
        for m in milestones:
            if self.session_reps < m:
                return {
                    "current": self.session_reps,
                    "target":  m,
                    "pct":     int(self.session_reps / m * 100),
                    "label":   f"{self.session_reps}/{m} reps",
                }
        return {
            "current": self.session_reps,
            "target":  self.session_reps,
            "pct":     100,
            "label":   "🏆 All milestones reached!",
        }

    def accuracy_stars(self) -> int:
        score = self.avg_form
        if score >= 95: return 5
        if score >= 80: return 4
        if score >= 65: return 3
        if score >= 50: return 2
        if score >= 30: return 1
        return 0

    # ── Badge engine ──────────────────────────────────────────────────────────
    def _check_badges(self):
        def award(name):
            if name not in self.new_badges and name not in self.earned_badges:
                self.new_badges.append(name)

        if self.session_reps >= 1:          award("First Rep")
        if self.session_reps >= 10:         award("Warming Up")
        if self.session_reps >= 25:         award("On a Roll")
        if self.lifetime_reps >= 100:       award("Century Club")
        if self.lifetime_reps >= 500:       award("Iron Will")
        if self.high_form_streak >= 20:     award("Form Master")
        if self.perfect_streak >= 10:       award("Perfect Set")

        if len(self.history) >= 2:
            dates = sorted([s.get("date", "") for s in self.history])
            if dates[-1] == str(date.today()):
                award("Consistency")

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_session(self, exercise: str):
        entry = {
            "date":     str(date.today()),
            "exercise": exercise,
            "reps":     self.session_reps,
            "avg_form": self.avg_form,
            "badges":   self.new_badges,
        }
        self.history.append(entry)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def _load_history(self) -> list:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []
