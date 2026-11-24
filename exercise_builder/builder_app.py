"""
exercise_builder/builder_app.py
--------------------------------
CustomTkinter multi-step wizard for adding new exercises.

Steps:
  1. Basic Info
  2. Joint Selection + feedback messages
  3. Select training video folder + extract poses
  4. Rep annotation (with joint tracking selector)
  5. Train & Save (with threshold tuning sliders)
  6. Mistake Detection — OPTIONAL
       Point to your QEVD root, pick which QEVD exercise name maps to this
       exercise, set quality/quantity parameters, then ingest + train.
       Can be skipped entirely — the exercise works fine without it.
       Can also be run later by re-opening the builder and going straight
       to Step 6 for an already-trained exercise.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.landmarks import (
    get_name, get_description, is_derived,
    get_all_selectable, EXERCISE_JOINT_SUGGESTIONS, DERIVED_JOINTS,
    get_suggestions, get_tracking_suggestion,
)
from exercise_builder.pose_extractor import extract_poses_from_folder
from exercise_builder.form_trainer   import train_form_model
from exercise_builder.rep_trainer    import build_rep_pattern
from exercise_builder.rep_annotator  import RepAnnotator

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

EXERCISES_DATA_DIR = "exercises_data"
MIN_ANNOTATED_REPS = 20


class BuilderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Exercise Builder — AI Physical Therapist")
        self.geometry("960x720")
        self.resizable(True, True)

        self._init_state()
        self._build_header()
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self._build_footer()
        self._show_step(0)

    def _init_state(self):
        self.exercise_name   = tk.StringVar()
        self.description     = tk.StringVar()
        self.muscle_groups   = tk.StringVar()
        self.video_folder    = tk.StringVar()

        self.added_joints       = []
        self.extracted_poses    = []
        self.annotations        = []
        self.video_poses        = []
        self.tracking_joint_idx = None

        self._computed_threshold = None
        self._computed_deadband  = None
        self._computed_lo        = None
        self._computed_hi        = None

        self._threshold_var = None
        self._deadband_var  = None

        # QEVD state
        self._qevd_root        = tk.StringVar()
        self._qevd_ex_name     = tk.StringVar()  # which QEVD exercise name to use
        self._qevd_max_clips   = tk.IntVar(value=50)
        self._qevd_min_det     = tk.DoubleVar(value=0.70)
        self._qevd_min_vis     = tk.DoubleVar(value=0.60)
        self._qevd_trained     = False

        self.current_step = 0
        self.steps = [
            self._build_step1,
            self._build_step2,
            self._build_step3,
            self._build_step4,
            self._build_step5,
            self._build_step6,
        ]
        self.step_titles = [
            "Step 1 — Basic Info",
            "Step 2 — Key Joints & Feedback",
            "Step 3 — Training Videos",
            "Step 4 — Rep Annotation",
            "Step 5 — Train & Save",
            "Step 6 — Mistake Detection (optional)",
        ]

    # ── Header / Footer / Nav ──────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, height=60)
        header.pack(fill="x", padx=20, pady=(15, 0))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🏥 Exercise Builder",
            font=("Helvetica", 20, "bold")
        ).pack(side="left", padx=15)

        self.step_indicator = ctk.CTkLabel(
            header, text="Step 1 of 6",
            font=("Helvetica", 13), text_color="gray"
        )
        self.step_indicator.pack(side="right", padx=15)

    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=60)
        footer.pack(fill="x", padx=20, pady=(0, 15))
        footer.pack_propagate(False)

        self.back_btn = ctk.CTkButton(
            footer, text="◀ Back",
            command=self._prev_step,
            fg_color="#4a4e69", hover_color="#2b2d42",
            width=120
        )
        self.back_btn.pack(side="left", padx=15, pady=10)

        self.next_btn = ctk.CTkButton(
            footer, text="Next ▶",
            command=self._next_step,
            width=120
        )
        self.next_btn.pack(side="right", padx=15, pady=10)

    def _show_step(self, step: int):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        self.current_step = step
        self.step_indicator.configure(
            text=f"{self.step_titles[step]}  ({step+1} of {len(self.steps)})"
        )
        self.back_btn.configure(state="normal" if step > 0 else "disabled")

        # Step 6 is skippable — show "Finish" instead of "Next"
        if step == len(self.steps) - 1:
            self.next_btn.configure(
                text="Finish ✅",
                fg_color="#2d6a4f", hover_color="#1b4332"
            )
        else:
            # Reset to the default CTk blue used when the button was created
            self.next_btn.configure(
                text="Next ▶",
                fg_color="#1f538d", hover_color="#144272"
            )

        self.steps[step]()

    def _next_step(self):
        if not self._validate_step(self.current_step):
            return
        if self.current_step < len(self.steps) - 1:
            self._show_step(self.current_step + 1)
        else:
            self.destroy()

    def _prev_step(self):
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _validate_step(self, step: int) -> bool:
        if step == 0:
            if not self.exercise_name.get().strip():
                messagebox.showwarning("Missing Info", "Please enter an exercise name.")
                return False
        if step == 1:
            if not self.added_joints:
                messagebox.showwarning("No Joints",
                    "Please add at least one key joint.")
                return False
        if step == 2:
            if not self.video_folder.get() or not os.path.isdir(self.video_folder.get()):
                messagebox.showwarning("No Folder",
                    "Please select a valid video folder.")
                return False
        if step == 3:
            if len(self.annotations) < MIN_ANNOTATED_REPS:
                messagebox.showwarning(
                    "Not Enough Reps",
                    f"Please annotate at least {MIN_ANNOTATED_REPS} reps.\n"
                    f"Current: {len(self.annotations)}\n\n"
                    f"More annotated reps = better scoring accuracy.\n"
                    f"Aim for 30+ reps across different videos."
                )
                return False
        # Step 6 (QEVD) is always valid to skip — no mandatory fields
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Basic Info
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step1(self):
        frame = ctk.CTkScrollableFrame(self.content_frame,
                                        label_text="Exercise Details")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        def row(label, var, placeholder=""):
            r = ctk.CTkFrame(frame, fg_color="transparent")
            r.pack(fill="x", pady=10)
            ctk.CTkLabel(r, text=label, width=160,
                         font=("Helvetica", 13)).pack(side="left")
            ctk.CTkEntry(r, textvariable=var,
                         placeholder_text=placeholder,
                         width=520, height=40).pack(side="left")

        row("Exercise Name *", self.exercise_name, "e.g. Squats")
        row("Description",     self.description,   "e.g. Lower body compound movement")
        row("Muscle Groups",   self.muscle_groups,  "e.g. Quads, Glutes, Hamstrings")

        ctk.CTkLabel(
            frame,
            text=(
                "Tip: joint suggestions and tracking joint recommendations "
                "will appear in Step 2.\n"
                "You can optionally add QEVD mistake detection in Step 6."
            ),
            font=("Helvetica", 11), text_color="gray"
        ).pack(pady=(20, 0))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Joint Selection
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step2(self):
        self._step2_frame = ctk.CTkFrame(self.content_frame)
        self._step2_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._build_step2_top(self._step2_frame)
        self._build_step2_cards(self._step2_frame)

    def _build_step2_top(self, parent):
        top = ctk.CTkFrame(parent)
        top.pack(fill="x", padx=10, pady=(10, 5))

        selectable         = get_all_selectable()
        self._dropdown_map = {}
        options            = []

        for group, entries in selectable.items():
            for idx, name, desc in entries:
                if any(j["idx"] == idx for j in self.added_joints):
                    continue
                label = f"{group}  —  {name}"
                if desc:
                    label += (f"  ({desc[:40]}...)"
                              if len(desc) > 40 else f"  ({desc})")
                options.append(label)
                self._dropdown_map[label] = idx

        self._dropdown_var     = tk.StringVar(value=options[0] if options else "")
        self._dropdown_options = options

        ctk.CTkLabel(top, text="Add a joint:",
                     font=("Helvetica", 13)).pack(side="left", padx=(10, 5))

        self._dropdown = ctk.CTkOptionMenu(
            top, variable=self._dropdown_var,
            values=options if options else ["No more joints available"],
            width=480, height=36, font=("Helvetica", 12)
        )
        self._dropdown.pack(side="left", padx=5)

        ctk.CTkButton(
            top, text="+ Add Joint",
            command=self._add_joint_from_dropdown,
            width=110, height=36,
            fg_color="#2d6a4f", hover_color="#1b4332"
        ).pack(side="left", padx=8)

        ex_name       = self.exercise_name.get().strip()
        suggestions   = get_suggestions(ex_name)
        tracking_hint = get_tracking_suggestion(ex_name)

        if suggestions:
            banner = ctk.CTkFrame(parent, fg_color="#1d3557", corner_radius=8)
            banner.pack(fill="x", padx=10, pady=(5, 0))
            names = ", ".join(get_name(i) for i in suggestions)
            ctk.CTkLabel(
                banner,
                text=f"💡 Suggested joints for {ex_name}: {names}",
                font=("Helvetica", 12), wraplength=700, text_color="#90e0ef"
            ).pack(side="left", padx=12, pady=(8, 2))
            ctk.CTkButton(
                banner, text="Add All",
                command=lambda s=suggestions: self._add_suggested(s),
                width=90, height=30,
                fg_color="#457b9d", hover_color="#1d3557"
            ).pack(side="right", padx=10)

        if tracking_hint:
            t_idx, t_desc = tracking_hint
            track_banner = ctk.CTkFrame(parent, fg_color="#1b3a1b", corner_radius=8)
            track_banner.pack(fill="x", padx=10, pady=(4, 0))
            ctk.CTkLabel(
                track_banner,
                text=f"🎯 Recommended tracking joint:  {t_desc}",
                font=("Helvetica", 12), text_color="#95d5b2", wraplength=750
            ).pack(anchor="w", padx=12, pady=6)

    def _build_step2_cards(self, parent):
        self._cards_frame = ctk.CTkScrollableFrame(parent,
                                                    label_text="Key Joints",
                                                    height=380)
        self._cards_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._refresh_cards()

    def _refresh_cards(self):
        for w in self._cards_frame.winfo_children():
            w.destroy()
        if not self.added_joints:
            ctk.CTkLabel(
                self._cards_frame,
                text="No joints added yet — use the dropdown above.",
                font=("Helvetica", 12), text_color="gray"
            ).pack(pady=30)
            return
        for joint in self.added_joints:
            self._build_joint_card(self._cards_frame, joint)

    def _build_joint_card(self, parent, joint: dict):
        idx  = joint["idx"]
        name = joint["name"]
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.pack(fill="x", padx=5, pady=6)

        header = ctk.CTkFrame(card, fg_color="#2b2d42", corner_radius=8)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(header, text=f"  {name}",
                     font=("Helvetica", 14, "bold")).pack(side="left", padx=8, pady=6)
        desc = get_description(idx)
        if desc:
            ctk.CTkLabel(header, text=desc,
                         font=("Helvetica", 11),
                         text_color="gray").pack(side="left", padx=4)
        ctk.CTkButton(
            header, text="Remove Joint",
            command=lambda j=joint: self._remove_joint(j),
            width=120, height=28,
            fg_color="#6b0f1a", hover_color="#c9184a"
        ).pack(side="right", padx=8)

        msg_frame = ctk.CTkFrame(card, fg_color="transparent")
        msg_frame.pack(fill="x", padx=12, pady=4)
        if not joint["messages"]:
            ctk.CTkLabel(msg_frame, text="No feedback messages yet.",
                         font=("Helvetica", 11),
                         text_color="gray").pack(anchor="w", pady=2)
        else:
            for i, msg in enumerate(joint["messages"]):
                row = ctk.CTkFrame(msg_frame, fg_color="#1a1a2e", corner_radius=6)
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=f"  {msg}", font=("Helvetica", 12),
                             anchor="w", wraplength=680).pack(
                    side="left", padx=8, pady=5, fill="x", expand=True)
                ctk.CTkButton(
                    row, text="x",
                    command=lambda j=joint, mi=i: self._remove_message(j, mi),
                    width=30, height=28,
                    fg_color="#3d405b", hover_color="#6b0f1a"
                ).pack(side="right", padx=6)

        add_row = ctk.CTkFrame(card, fg_color="transparent")
        add_row.pack(fill="x", padx=12, pady=(4, 10))
        msg_var = tk.StringVar()
        ctk.CTkEntry(
            add_row, textvariable=msg_var,
            placeholder_text="Type a feedback message for this joint...",
            width=580, height=36
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            add_row, text="+ Add",
            command=lambda j=joint, v=msg_var: self._add_message(j, v),
            width=80, height=36,
            fg_color="#2d6a4f", hover_color="#1b4332"
        ).pack(side="left")

    def _add_joint_from_dropdown(self):
        label = self._dropdown_var.get()
        if not label or label not in self._dropdown_map:
            return
        idx = self._dropdown_map[label]
        if any(j["idx"] == idx for j in self.added_joints):
            return
        self.added_joints.append(
            {"idx": idx, "name": get_name(idx), "messages": []})
        self._rebuild_step2()

    def _add_suggested(self, suggestions: list):
        for idx in suggestions:
            if not any(j["idx"] == idx for j in self.added_joints):
                self.added_joints.append(
                    {"idx": idx, "name": get_name(idx), "messages": []})
        self._rebuild_step2()

    def _remove_joint(self, joint: dict):
        self.added_joints = [j for j in self.added_joints
                             if j["idx"] != joint["idx"]]
        self._rebuild_step2()

    def _add_message(self, joint: dict, var: tk.StringVar):
        msg = var.get().strip()
        if not msg:
            return
        joint["messages"].append(msg)
        var.set("")
        self._refresh_cards()

    def _remove_message(self, joint: dict, msg_index: int):
        joint["messages"].pop(msg_index)
        self._refresh_cards()

    def _rebuild_step2(self):
        for w in self._step2_frame.winfo_children():
            w.destroy()
        self._build_step2_top(self._step2_frame)
        self._build_step2_cards(self._step2_frame)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Training Videos
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step3(self):
        frame = ctk.CTkFrame(self.content_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            frame,
            text=(
                "Select the folder containing your correct-form training videos.\n"
                "Aim for 40+ videos, 5–10 reps each, good lighting, "
                "clear full-body view."
            ),
            font=("Helvetica", 13), wraplength=750
        ).pack(pady=20)

        folder_row = ctk.CTkFrame(frame, fg_color="transparent")
        folder_row.pack(fill="x", padx=20)

        ctk.CTkEntry(
            folder_row, textvariable=self.video_folder,
            width=560, height=38,
            placeholder_text="No folder selected..."
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            folder_row, text="📂 Browse",
            command=self._browse_folder,
            width=120
        ).pack(side="left")

        ctk.CTkLabel(frame, text="Pose Extraction Progress",
                     font=("Helvetica", 13, "bold")).pack(pady=(30, 5))

        self.extract_progress = ctk.CTkProgressBar(frame, width=600)
        self.extract_progress.pack(pady=5)
        self.extract_progress.set(0)

        self.extract_status = ctk.CTkLabel(
            frame, text="Press 'Extract Poses' to begin.",
            font=("Helvetica", 12), text_color="gray"
        )
        self.extract_status.pack(pady=5)

        ctk.CTkButton(
            frame, text="⚙️ Extract Poses",
            command=self._run_extraction,
            height=42, font=("Helvetica", 13, "bold")
        ).pack(pady=15)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select video folder")
        if folder:
            self.video_folder.set(folder)

    def _run_extraction(self):
        folder = self.video_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Folder",
                                   "Please select a valid folder first.")
            return

        ex_name  = self.exercise_name.get().strip().lower().replace(" ", "_")
        out_path = os.path.join(EXERCISES_DATA_DIR, ex_name, "poses.npy")

        self.extract_status.configure(
            text="Extracting poses...", text_color="white")
        self.extract_progress.set(0)
        self.next_btn.configure(state="disabled")

        def progress_cb(current, total, filename):
            pct = current / max(total, 1)
            self.after(0, lambda: self.extract_progress.set(pct))
            self.after(0, lambda: self.extract_status.configure(
                text=f"Processing: {filename} ({current}/{total})"
            ))

        def run():
            try:
                result = extract_poses_from_folder(folder, out_path, progress_cb)
                self.extracted_poses = result["poses"]
                self.after(0, lambda: self.extract_status.configure(
                    text=(f"✅ Done — {result['total_videos']} videos, "
                          f"{result['total_frames']} frames extracted."),
                    text_color="#95d5b2"
                ))
                self.after(0, lambda: self.extract_progress.set(1.0))
                self.after(0, lambda: self.next_btn.configure(state="normal"))
            except Exception as e:
                self.after(0, lambda: self.extract_status.configure(
                    text=f"❌ Error: {str(e)}", text_color="#e63946"
                ))
                self.after(0, lambda: self.next_btn.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Rep Annotation
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step4(self):
        frame = ctk.CTkFrame(self.content_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            frame,
            text=(
                f"Annotate reps across as many videos as possible.\n"
                f"Minimum: {MIN_ANNOTATED_REPS} reps. "
                f"Recommended: 30+ reps across different videos."
            ),
            font=("Helvetica", 13), wraplength=750
        ).pack(pady=(15, 5))

        instructions = ctk.CTkFrame(frame, fg_color="#1a1a2e", corner_radius=8)
        instructions.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(
            instructions,
            text="⚠️  ANNOTATION RULES — read before annotating:",
            font=("Helvetica", 13, "bold"), text_color="#f4a261"
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            instructions,
            text=(
                "• Press Mark Start at the VERY BEGINNING of movement "
                "(arm fully extended, body fully upright, etc.)\n"
                "• Press Mark End when the person has FULLY RETURNED "
                "to that same rest position\n"
                "• If you mark start mid-movement or end before returning "
                "to rest, the threshold will be wrong\n"
                "• Annotate reps from as many DIFFERENT videos as possible"
            ),
            font=("Helvetica", 12), text_color="#e0e0e0",
            wraplength=750, justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 12))

        self.annot_status = ctk.CTkLabel(
            frame,
            text=(f"Annotations so far: {len(self.annotations)} reps  "
                  f"(need {MIN_ANNOTATED_REPS}+, want 30+)"),
            font=("Helvetica", 14, "bold"), text_color="#90e0ef"
        )
        self.annot_status.pack(pady=10)

        if self.tracking_joint_idx is not None:
            ctk.CTkLabel(
                frame,
                text=f"Tracking joint: {get_name(self.tracking_joint_idx)}",
                font=("Helvetica", 12), text_color="#95d5b2"
            ).pack(pady=2)

        ctk.CTkButton(
            frame, text="🎬 Open Annotation Tool",
            command=self._open_annotator,
            height=50, font=("Helvetica", 14, "bold"),
            width=300
        ).pack(pady=15)

        if self.annotations:
            n_videos = len(set(a['video_index'] for a in self.annotations))
            ctk.CTkLabel(
                frame,
                text=(f"✅ {len(self.annotations)} reps annotated "
                      f"across {n_videos} videos."),
                font=("Helvetica", 12), text_color="#95d5b2"
            ).pack()

    def _open_annotator(self):
        folder = self.video_folder.get()
        if not folder:
            messagebox.showwarning("No Folder",
                                   "Please complete Step 3 first.")
            return

        import glob
        video_files = sorted(
            glob.glob(os.path.join(folder, "*.mp4")) +
            glob.glob(os.path.join(folder, "*.avi")) +
            glob.glob(os.path.join(folder, "*.mov"))
        )

        if not video_files:
            messagebox.showwarning("No Videos",
                "No video files found in the selected folder.")
            return

        key_indices = [j["idx"] for j in self.added_joints]

        def on_complete(annotations, video_poses, tracking_joint_idx):
            self.annotations        = annotations
            self.video_poses        = video_poses
            self.tracking_joint_idx = tracking_joint_idx
            n_videos = len(set(a['video_index'] for a in annotations))
            self.annot_status.configure(
                text=(f"Annotations: {len(annotations)} reps  |  "
                      f"{n_videos} videos  |  "
                      f"Tracking: {get_name(tracking_joint_idx)}")
            )

        RepAnnotator(self, video_files, key_indices, on_complete)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Train & Save
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step5(self):
        self._step5_frame = ctk.CTkScrollableFrame(self.content_frame)
        self._step5_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ex_name     = self.exercise_name.get().strip()
        key_indices = [j["idx"] for j in self.added_joints]

        ctk.CTkLabel(
            self._step5_frame, text=f"Ready to train: {ex_name}",
            font=("Helvetica", 17, "bold")
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            self._step5_frame,
            text=(f"Training videos: {len(self.extracted_poses)}\n"
                  f"Annotated reps:  {len(self.annotations)}\n"
                  f"Key joints:      {len(key_indices)}\n"
                  f"Tracking joint:  "
                  f"{get_name(self.tracking_joint_idx) if self.tracking_joint_idx else 'auto'}"),
            font=("Helvetica", 13), text_color="gray", justify="left"
        ).pack(pady=10)

        self.train_status = ctk.CTkLabel(
            self._step5_frame, text="Press Train to begin.",
            font=("Helvetica", 12), text_color="gray"
        )
        self.train_status.pack(pady=5)

        self.train_progress = ctk.CTkProgressBar(self._step5_frame, width=500)
        self.train_progress.pack(pady=10)
        self.train_progress.set(0)

        ctk.CTkButton(
            self._step5_frame, text="🚀 Train & Save Exercise",
            command=self._run_training,
            height=50, font=("Helvetica", 14, "bold"),
            width=300
        ).pack(pady=15)

        self._threshold_section = ctk.CTkFrame(
            self._step5_frame, fg_color="transparent")
        self._threshold_section.pack(fill="x", padx=10, pady=5)

        self._finish_section = ctk.CTkFrame(
            self._step5_frame, fg_color="transparent")
        self._finish_section.pack(fill="x", padx=10, pady=10)

        self.next_btn.configure(state="disabled", text="Next ▶")

    def _build_threshold_sliders(self):
        for w in self._threshold_section.winfo_children():
            w.destroy()

        f = self._threshold_section

        ctk.CTkLabel(
            f, text="── Rep Detection Tuning ──",
            font=("Helvetica", 14, "bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            f,
            text=(
                "Adjust these if reps are not being counted correctly.\n"
                "Threshold = the midpoint the tracked joint must cross.\n"
                "Deadband  = how far past threshold before committing "
                "(prevents false counts)."
            ),
            font=("Helvetica", 11), text_color="gray", wraplength=700
        ).pack(pady=(0, 10))

        lo = self._computed_lo or 0.0
        hi = self._computed_hi or 180.0

        thresh_row = ctk.CTkFrame(f, fg_color="transparent")
        thresh_row.pack(fill="x", pady=5)
        ctk.CTkLabel(thresh_row, text="Threshold:", width=110,
                     font=("Helvetica", 12)).pack(side="left")
        self._threshold_var = tk.DoubleVar(
            value=round(self._computed_threshold, 4))
        thresh_val_label = ctk.CTkLabel(
            thresh_row, text=f"{self._computed_threshold:.1f}°",
            font=("Helvetica", 12), width=60)
        thresh_val_label.pack(side="right", padx=10)

        def on_thresh(val):
            thresh_val_label.configure(text=f"{float(val):.1f}°")
            self._threshold_var.set(float(val))

        ctk.CTkSlider(
            thresh_row, from_=lo, to=hi,
            variable=self._threshold_var,
            command=on_thresh, width=450
        ).pack(side="left", padx=10)

        dead_row = ctk.CTkFrame(f, fg_color="transparent")
        dead_row.pack(fill="x", pady=5)
        ctk.CTkLabel(dead_row, text="Deadband:", width=110,
                     font=("Helvetica", 12)).pack(side="left")
        self._deadband_var = tk.DoubleVar(
            value=round(self._computed_deadband, 4))
        dead_val_label = ctk.CTkLabel(
            dead_row, text=f"{self._computed_deadband:.1f}°",
            font=("Helvetica", 12), width=60)
        dead_val_label.pack(side="right", padx=10)

        def on_dead(val):
            dead_val_label.configure(text=f"{float(val):.1f}°")
            self._deadband_var.set(float(val))

        ctk.CTkSlider(
            dead_row,
            from_=5, to=max(40, self._computed_deadband * 3),
            variable=self._deadband_var,
            command=on_dead, width=450
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            f, text="💾 Save Adjusted Values",
            command=self._save_adjusted_thresholds,
            fg_color="#457b9d", hover_color="#1d3557",
            height=36
        ).pack(pady=10)

    def _build_finish_buttons_step5(self):
        for w in self._finish_section.winfo_children():
            w.destroy()

        f = self._finish_section

        ctk.CTkLabel(f, text="── What next? ──",
                     font=("Helvetica", 13, "bold")).pack(pady=(10, 8))

        ctk.CTkLabel(
            f,
            text="Continue to Step 6 to optionally add QEVD mistake detection.\n"
                 "Or close / add another exercise without it.",
            font=("Helvetica", 12), text_color="gray", wraplength=700
        ).pack(pady=(0, 10))

        btn_row = ctk.CTkFrame(f, fg_color="transparent")
        btn_row.pack()

        ctk.CTkButton(
            btn_row, text="➕ Add Another Exercise",
            command=self._add_another,
            fg_color="#2d6a4f", hover_color="#1b4332",
            height=45, width=220, font=("Helvetica", 13, "bold")
        ).pack(side="left", padx=15)

        ctk.CTkButton(
            btn_row, text="✅ Close Builder",
            command=self.destroy,
            fg_color="#4a4e69", hover_color="#2b2d42",
            height=45, width=220, font=("Helvetica", 13, "bold")
        ).pack(side="left", padx=15)

    def _save_adjusted_thresholds(self):
        ex_name = self.exercise_name.get().strip()
        slug    = ex_name.lower().replace(" ", "_")
        out_dir = os.path.join(EXERCISES_DATA_DIR, slug)
        config_path = os.path.join(out_dir, "config.json")

        if not os.path.exists(config_path):
            messagebox.showerror("Error", "Config file not found. Train first.")
            return

        with open(config_path) as fh:
            cfg = json.load(fh)

        cfg["threshold_override"] = round(self._threshold_var.get(), 3)
        cfg["deadband_override"]  = round(self._deadband_var.get(), 3)

        with open(config_path, "w") as fh:
            json.dump(cfg, fh, indent=2)

        messagebox.showinfo(
            "Saved",
            f"Threshold: {cfg['threshold_override']:.1f}°\n"
            f"Deadband:  {cfg['deadband_override']:.1f}°\n\n"
            "Values saved. Restart the patient app to apply."
        )

    def _add_another(self):
        self._init_state()
        self._show_step(0)

    # ── Training ───────────────────────────────────────────────────────────────
    def _run_training(self):
        ex_name     = self.exercise_name.get().strip()
        slug        = ex_name.lower().replace(" ", "_")
        out_dir     = os.path.join(EXERCISES_DATA_DIR, slug)
        key_indices = [j["idx"] for j in self.added_joints]

        tracking_joint = self.tracking_joint_idx or key_indices[0]

        steps_done = [0]

        def progress_cb(msg):
            steps_done[0] += 1
            pct = min(steps_done[0] / 6, 1.0)
            self.after(0, lambda: self.train_progress.set(pct))
            self.after(0, lambda: self.train_status.configure(text=msg))

        def run():
            try:
                progress_cb("Building rep pattern from annotations...")
                rep_result = build_rep_pattern(
                    self.annotations,
                    self.video_poses,
                    key_indices,
                    out_dir,
                    tracking_joint_index=tracking_joint,
                )

                self._computed_threshold = rep_result["threshold"]
                self._computed_deadband  = rep_result["deadband"]
                self._computed_lo        = rep_result["lo"]
                self._computed_hi        = rep_result["hi"]

                progress_cb("Building form profile...")
                from exercise_builder.rep_trainer import build_form_profile
                from core.landmarks import ANGLE_TRIPLETS, BILATERAL_PAIRS
                import numpy as np

                t_triplet = ANGLE_TRIPLETS.get(tracking_joint)
                b_joint   = BILATERAL_PAIRS.get(tracking_joint)
                b_triplet = ANGLE_TRIPLETS.get(b_joint) if b_joint else None

                raw_rep_seqs = []
                for ann in self.annotations:
                    vid_idx = ann["video_index"]
                    sf, ef  = ann["start_frame"], ann["end_frame"]
                    if vid_idx >= len(self.video_poses): continue
                    raw = self.video_poses[vid_idx]
                    if not raw or ef > len(raw) or sf >= ef: continue
                    valid = [p for p in raw[sf:ef] if p is not None]
                    if len(valid) >= 5:
                        raw_rep_seqs.append(
                            np.array(valid).reshape(-1, 33, 3))

                if raw_rep_seqs and t_triplet:
                    build_form_profile(
                        raw_rep_seqs, key_indices,
                        t_triplet, b_triplet,
                        rep_result["lo"], rep_result["hi"],
                        out_dir,
                    )

                progress_cb("Training form model (rest-position baseline)...")
                train_form_model(
                    self.extracted_poses,
                    out_dir,
                    key_indices,
                    annotations=self.annotations,
                    video_poses=self.video_poses,
                    progress_callback=progress_cb
                )

                progress_cb("Saving config...")
                self._save_config(ex_name, slug, out_dir, key_indices,
                                  tracking_joint, rep_result)

                self.after(0, lambda: self.train_status.configure(
                    text=f"✅ {ex_name} saved to {out_dir}",
                    text_color="#95d5b2"
                ))
                self.after(0, lambda: self.train_progress.set(1.0))
                self.after(0, self._build_threshold_sliders)
                self.after(0, self._build_finish_buttons_step5)
                # Unlock Next so user can proceed to Step 6
                self.after(0, lambda: self.next_btn.configure(state="normal"))

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self.train_status.configure(
                    text=f"❌ Error: {str(e)}", text_color="#e63946"
                ))
                print(tb)
                self.after(0, lambda: self.next_btn.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _save_config(self, ex_name, slug, out_dir, key_indices,
                     tracking_joint, rep_result):
        muscles = [m.strip()
                   for m in self.muscle_groups.get().split(",")
                   if m.strip()]

        key_joints_config = []
        for joint in self.added_joints:
            key_joints_config.append({
                "name":              joint["name"],
                "landmark_index":    joint["idx"],
                "feedback_messages": joint["messages"],
            })

        config = {
            "name":                  ex_name,
            "description":           self.description.get().strip(),
            "muscle_groups":         muscles,
            "key_joints":            key_joints_config,
            "all_landmark_indices":  list(range(11, 33)),
            "trained":               True,
            "training_video_folder": self.video_folder.get(),
            "num_training_videos":   len(self.extracted_poses),
            "num_annotated_reps":    len(self.annotations),
            "tracking_joint_index":  tracking_joint,
            "bilateral_joint_index": rep_result.get("bilateral_joint_index"),
            "threshold_auto":        rep_result["threshold"],
            "deadband_auto":         rep_result["deadband"],
            "start_above":           rep_result["start_above"],
            "lo":                    rep_result["lo"],
            "hi":                    rep_result["hi"],
        }

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Mistake Detection (optional QEVD)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step6(self):
        """
        Optional QEVD mistake-classifier training, embedded directly in the
        builder flow.  The step can be skipped by pressing Finish without
        doing anything — the exercise is already fully trained from Step 5.

        Layout:
          ┌──────────────────────────────────────────────────────────┐
          │  What is this?  (explanation banner)                      │
          ├──────────────────────────────────────────────────────────┤
          │  QEVD root folder  [Browse]                               │
          │  Exercise name in QEVD  [auto-filled / editable]          │
          │  Quality settings (collapsible)                           │
          │  [Run Ingestion + Training]                               │
          │  Progress log                                             │
          ├──────────────────────────────────────────────────────────┤
          │  Status / result summary                                   │
          └──────────────────────────────────────────────────────────┘
        """
        self._step6_frame = ctk.CTkScrollableFrame(self.content_frame)
        self._step6_frame.pack(fill="both", expand=True, padx=10, pady=10)
        f = self._step6_frame

        # ── Already trained banner ─────────────────────────────────────────────
        ex_name = self.exercise_name.get().strip()
        slug    = ex_name.lower().replace(" ", "_")
        out_dir = os.path.join(EXERCISES_DATA_DIR, slug)
        already = os.path.exists(os.path.join(out_dir, "mistake_classifier.pkl"))

        if already or self._qevd_trained:
            ctk.CTkLabel(
                f,
                text="✅  Mistake classifier already trained for this exercise.",
                font=("Helvetica", 14, "bold"), text_color="#95d5b2"
            ).pack(pady=(15, 5))

        # ── Header ─────────────────────────────────────────────────────────────
        ctk.CTkLabel(
            f,
            text="Mistake Detection  —  Powered by QEVD",
            font=("Helvetica", 17, "bold")
        ).pack(pady=(16, 4))

        info = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        info.pack(fill="x", padx=10, pady=(0, 14))
        ctk.CTkLabel(
            info,
            text=(
                "This optional step trains a multi-class mistake classifier using the "
                "QEVD-FIT-300k dataset.\n\n"
                "• Point to your local QEVD root folder (contains "
                "QEVD-FIT-300k-Part-1 … Part-4)\n"
                "• Enter the exercise name exactly as it appears in QEVD "
                "(e.g. 'squats', 'push up')\n"
                "• The builder will ingest the matching clips, extract poses, "
                "and train an SVC in one step\n\n"
                "Skip this step if you don't have the dataset or if your exercise "
                "isn't in QEVD — everything else still works."
            ),
            font=("Helvetica", 12), text_color="#b0b0c8",
            wraplength=780, justify="left"
        ).pack(anchor="w", padx=14, pady=12)

        # ── Folder + exercise name ─────────────────────────────────────────────
        ctk.CTkLabel(f, text="QEVD Configuration",
                     font=("Helvetica", 14, "bold")).pack(anchor="w",
                                                          padx=10, pady=(6, 4))

        root_row = ctk.CTkFrame(f, fg_color="transparent")
        root_row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(root_row, text="QEVD root:", width=160,
                     font=("Helvetica", 13)).pack(side="left")
        ctk.CTkEntry(root_row, textvariable=self._qevd_root,
                     width=440, height=38,
                     placeholder_text=(
                         "e.g. D:/QEVD  (must contain QEVD-FIT-300k-Part-4/)")).pack(
            side="left", padx=(0, 8))
        ctk.CTkButton(
            root_row, text="📂 Browse",
            command=self._browse_qevd_root,
            width=100, height=38
        ).pack(side="left")

        # Auto-fill QEVD exercise name from the exercise name in Step 1
        if not self._qevd_ex_name.get():
            # Best-effort mapping: lowercase, strip trailing 's', common variants
            candidate = ex_name.lower().rstrip("s")
            self._qevd_ex_name.set(candidate)

        name_row = ctk.CTkFrame(f, fg_color="transparent")
        name_row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(name_row, text="QEVD exercise name:", width=160,
                     font=("Helvetica", 13)).pack(side="left")
        ctk.CTkEntry(name_row, textvariable=self._qevd_ex_name,
                     width=440, height=38,
                     placeholder_text=(
                         "e.g. squats, push up, jumping jacks")).pack(
            side="left", padx=(0, 8))

        # Small hint
        ctk.CTkLabel(
            f,
            text=(
                "Tip: run  python qevd_explorer.py --qevd_part4 <path>  "
                "in a terminal to browse available exercise names."
            ),
            font=("Helvetica", 11), text_color="gray"
        ).pack(anchor="w", padx=10, pady=(0, 6))

        # ── Quality / quantity settings ────────────────────────────────────────
        adv = ctk.CTkFrame(f, fg_color="#131318", corner_radius=8)
        adv.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(adv, text="Quality & Quantity Settings",
                     font=("Helvetica", 13, "bold")).pack(
            anchor="w", padx=12, pady=(10, 4))

        def param_row(parent, label, var, lo, hi, fmt=".0f"):
            r = ctk.CTkFrame(parent, fg_color="transparent")
            r.pack(fill="x", padx=12, pady=4)
            ctk.CTkLabel(r, text=label, width=200,
                         font=("Helvetica", 12)).pack(side="left")
            val_lbl = ctk.CTkLabel(r, text=f"{var.get():{fmt}}",
                                   width=60, font=("Helvetica", 12))
            val_lbl.pack(side="right")

            def on_change(v):
                try:
                    cast_val = type(var.get())(float(v))
                    var.set(cast_val)
                    val_lbl.configure(text=f"{cast_val:{fmt}}")
                except Exception:
                    pass

            ctk.CTkSlider(r, from_=lo, to=hi,
                          variable=var, command=on_change,
                          width=360).pack(side="left", padx=8)

        param_row(adv, "Max clips per mistake",
                  self._qevd_max_clips, 10, 100, fmt="d")
        param_row(adv, "Min pose detection rate",
                  self._qevd_min_det, 0.3, 1.0, fmt=".2f")
        param_row(adv, "Min landmark visibility",
                  self._qevd_min_vis, 0.3, 1.0, fmt=".2f")

        ctk.CTkLabel(
            adv,
            text=(
                "Lower detection/visibility thresholds accept more clips "
                "but may include noisier data.\n"
                "Higher values keep only clean clips — fewer samples but "
                "more reliable."
            ),
            font=("Helvetica", 11), text_color="gray", wraplength=720
        ).pack(anchor="w", padx=12, pady=(4, 10))

        # ── Run button ─────────────────────────────────────────────────────────
        self._qevd_run_btn = ctk.CTkButton(
            f, text="⚡ Ingest QEVD Data + Train Classifier",
            command=self._run_qevd_pipeline,
            height=50, font=("Helvetica", 14, "bold"),
            fg_color="#457b9d", hover_color="#1d3557",
            width=400
        )
        self._qevd_run_btn.pack(pady=14)

        # ── Progress ───────────────────────────────────────────────────────────
        self._qevd_progress = ctk.CTkProgressBar(f, width=600)
        self._qevd_progress.pack(pady=4)
        self._qevd_progress.set(0)

        self._qevd_status = ctk.CTkLabel(
            f, text="Ready — press the button above to begin.",
            font=("Helvetica", 12), text_color="gray"
        )
        self._qevd_status.pack(pady=4)

        self._qevd_log = ctk.CTkTextbox(f, height=180, font=("Courier", 11))
        self._qevd_log.pack(fill="x", padx=10, pady=6)
        self._qevd_log.configure(state="disabled")

        # If already trained, add a re-train note
        if already or self._qevd_trained:
            ctk.CTkLabel(
                f,
                text="You can re-run the pipeline above to update the classifier.",
                font=("Helvetica", 11), text_color="gray"
            ).pack(pady=(0, 8))

        # ── Bottom navigation ──────────────────────────────────────────────────
        bottom = ctk.CTkFrame(f, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=(12, 6))

        ctk.CTkButton(
            bottom, text="➕ Add Another Exercise",
            command=self._add_another,
            fg_color="#2d6a4f", hover_color="#1b4332",
            height=44, width=210, font=("Helvetica", 13, "bold")
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            bottom, text="✅ Close Builder",
            command=self.destroy,
            fg_color="#4a4e69", hover_color="#2b2d42",
            height=44, width=210, font=("Helvetica", 13, "bold")
        ).pack(side="left")

    def _browse_qevd_root(self):
        folder = filedialog.askdirectory(title="Select QEVD root folder")
        if folder:
            self._qevd_root.set(folder)

    def _qevd_log_append(self, text: str):
        """Append a line to the QEVD log textbox (thread-safe via after())."""
        def _do():
            self._qevd_log.configure(state="normal")
            self._qevd_log.insert("end", text + "\n")
            self._qevd_log.see("end")
            self._qevd_log.configure(state="disabled")
        self.after(0, _do)

    def _run_qevd_pipeline(self):
        """
        Combined ingest + train, run in a background thread.
        Reuses the logic from qevd_ingestor and qevd_trainer directly
        rather than shelling out, so progress can be reported inline.
        """
        qevd_root   = self._qevd_root.get().strip()
        qevd_ex_raw = self._qevd_ex_name.get().strip().lower()
        ex_name     = self.exercise_name.get().strip()
        slug        = ex_name.lower().replace(" ", "_")
        out_dir     = os.path.join(EXERCISES_DATA_DIR, slug)

        # ── Validation ─────────────────────────────────────────────────────────
        if not qevd_root or not os.path.isdir(qevd_root):
            messagebox.showwarning(
                "No QEVD Root",
                "Please select a valid QEVD root folder.\n"
                "It should contain QEVD-FIT-300k-Part-4/."
            )
            return

        json_path = os.path.join(
            qevd_root, "QEVD-FIT-300k-Part-4", "fine_grained_labels.json")
        if not os.path.exists(json_path):
            messagebox.showwarning(
                "Labels Not Found",
                f"fine_grained_labels.json not found at:\n{json_path}\n\n"
                "Make sure you selected the correct QEVD root folder."
            )
            return

        if not qevd_ex_raw:
            messagebox.showwarning(
                "No Exercise Name",
                "Please enter the exercise name as it appears in QEVD."
            )
            return

        # Disable button while running
        self._qevd_run_btn.configure(state="disabled",
                                      text="⏳ Running pipeline…")
        self._qevd_progress.set(0)

        max_clips = self._qevd_max_clips.get()
        min_det   = self._qevd_min_det.get()
        min_vis   = self._qevd_min_vis.get()

        def run():
            import json as _json
            import cv2
            import numpy as np
            import mediapipe as mp
            from collections import defaultdict

            # ── Local imports from the project modules ──────────────────────
            # We import here (inside the thread) to avoid circular issues.
            try:
                from qevd_ingestor import (
                    parse_label, safe_name, find_video,
                    extract_and_validate, save_checkpoint, load_checkpoint,
                    merge_checkpoints,
                )
                from qevd_trainer import (
                    train_exercise as qevd_train_exercise,
                    is_junk_class,
                )
            except ImportError:
                # Fall back if running from project root without package install
                sys.path.insert(0, os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                from qevd_ingestor import (
                    parse_label, safe_name, find_video,
                    extract_and_validate, save_checkpoint, load_checkpoint,
                    merge_checkpoints,
                )
                from qevd_trainer import (
                    train_exercise as qevd_train_exercise,
                    is_junk_class,
                )

            # ── Find QEVD part directories ──────────────────────────────────
            part_dirs = []
            for i in range(1, 5):
                d = os.path.join(qevd_root, f"QEVD-FIT-300k-Part-{i}")
                if os.path.isdir(d):
                    part_dirs.append(d)

            if not part_dirs:
                self.after(0, lambda: messagebox.showerror(
                    "Parts Not Found",
                    "No QEVD-FIT-300k-Part-N folders found inside the root.\n"
                    "Check the path and try again."
                ))
                self.after(0, self._qevd_reset_button)
                return

            self._qevd_log_append(
                f"Found {len(part_dirs)} QEVD part folder(s).")

            # ── Load annotations JSON ───────────────────────────────────────
            with open(json_path) as fh:
                data = _json.load(fh)

            # Filter to train split + target exercise
            exercise_clips  = defaultdict(list)
            exercise_descs  = {}
            n_matched       = 0

            for entry in data:
                if entry.get("split") != "train":
                    continue
                if not entry.get("labels"):
                    continue
                label       = entry["labels"][0]
                descriptive = (entry["labels_descriptive"][0]
                               if entry.get("labels_descriptive") else label)
                ex_raw, mistake = parse_label(label)

                if qevd_ex_raw not in ex_raw and ex_raw not in qevd_ex_raw:
                    continue
                if mistake is None:
                    continue

                desc_mistake = (descriptive.split(" - ", 1)[1].strip()
                                if " - " in descriptive
                                else descriptive.strip())

                exercise_clips[mistake].append({
                    "video_path":  entry["video_path"],
                    "descriptive": desc_mistake,
                })
                if mistake not in exercise_descs:
                    exercise_descs[mistake] = desc_mistake
                n_matched += 1

            if not exercise_clips:
                self._qevd_log_append(
                    f"❌  No clips found for '{qevd_ex_raw}' in the train split.\n"
                    "    Check the exercise name — use qevd_explorer.py to verify."
                )
                self.after(0, self._qevd_reset_button)
                return

            # Filter junk classes
            junk = [k for k in list(exercise_clips.keys()) if is_junk_class(k)]
            for k in junk:
                del exercise_clips[k]
            if junk:
                self._qevd_log_append(
                    f"Filtered {len(junk)} junk class(es): {junk}")

            self._qevd_log_append(
                f"Matched {n_matched} clips across "
                f"{len(exercise_clips)} mistake type(s) for '{qevd_ex_raw}'."
            )

            # ── Ingest ─────────────────────────────────────────────────────
            ckpt_dir = os.path.join(out_dir, "checkpoints")
            os.makedirs(out_dir,  exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            mp_pose = mp.solutions.pose
            pose_estimator = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            mistake_poses  = {}
            mistake_labels = {}
            split_counts   = {}
            quality_report = {}

            # Resume from checkpoints
            skipped = []
            for mk in sorted(exercise_clips.keys()):
                result = load_checkpoint(ckpt_dir, mk)
                if result is not None:
                    seqs, meta = result
                    mistake_poses[mk]  = seqs
                    mistake_labels[mk] = meta.get("label", mk)
                    split_counts[mk]   = meta.get("split_counts", {})
                    quality_report[mk] = meta.get("quality", {})
                    skipped.append(mk)

            if skipped:
                self._qevd_log_append(
                    f"Resuming — {len(skipped)} mistake(s) already checkpointed.")

            total_mistakes  = len(exercise_clips)
            done_count      = len(skipped)

            for mistake_key, clips in sorted(exercise_clips.items()):
                if mistake_key in skipped:
                    continue

                clips_to_use = clips[:max_clips]
                self._qevd_log_append(
                    f"\n[{mistake_key}]  "
                    f"({len(clips_to_use)} clips)")

                pose_seqs    = []
                fail_reasons = defaultdict(int)
                not_found    = 0

                for clip in clips_to_use:
                    video_path = find_video(clip["video_path"], part_dirs)
                    if video_path is None:
                        not_found += 1
                        fail_reasons["video_not_found"] += 1
                        continue

                    frames, quality = extract_and_validate(
                        video_path, pose_estimator,
                        min_detection_rate=min_det,
                        min_visibility=min_vis,
                    )

                    if frames:
                        pose_seqs.append(frames)
                    else:
                        reason = quality.get("fail_reason", "unknown")
                        bucket = (reason.split("(")[0].strip()
                                  if "(" in reason else reason)
                        fail_reasons[bucket] += 1

                processed = len(clips_to_use) - not_found
                pct       = int(len(pose_seqs) / max(processed, 1) * 100)
                self._qevd_log_append(
                    f"  ✅ {len(pose_seqs)}/{processed} passed  "
                    f"({pct}%)"
                    + (f"  ❌ {dict(fail_reasons)}" if fail_reasons else "")
                )

                q_info = {
                    "attempted":     len(clips_to_use),
                    "not_found":     not_found,
                    "passed":        len(pose_seqs),
                    "failed":        sum(fail_reasons.values()),
                    "pass_rate_pct": pct,
                    "fail_reasons":  dict(fail_reasons),
                }
                s_info = {
                    "attempted": len(clips_to_use),
                    "success":   len(pose_seqs),
                    "failed":    sum(fail_reasons.values()),
                }
                label_text = exercise_descs.get(mistake_key, mistake_key)

                if pose_seqs:
                    save_checkpoint(ckpt_dir, mistake_key,
                                    pose_seqs, label_text, s_info, q_info)
                    mistake_poses[mistake_key]  = pose_seqs
                    mistake_labels[mistake_key] = label_text
                    split_counts[mistake_key]   = s_info
                    quality_report[mistake_key] = q_info
                else:
                    self._qevd_log_append(
                        f"  ⚠️  No valid clips for '{mistake_key}' — skipping.")
                    quality_report[mistake_key] = q_info

                done_count += 1
                pct_overall = done_count / max(total_mistakes, 1)
                self.after(0,
                    lambda p=pct_overall * 0.7:   # ingest = 70 % of bar
                        self._qevd_progress.set(p))

            pose_estimator.close()

            if not mistake_poses:
                self._qevd_log_append(
                    "\n❌  Ingestion produced no valid clips at all.\n"
                    "Try lower quality thresholds or check the exercise name."
                )
                self.after(0, self._qevd_reset_button)
                return

            # Merge checkpoints into final npy/json files
            merge_checkpoints(
                out_dir, ckpt_dir,
                mistake_poses, mistake_labels,
                split_counts, quality_report)

            total_clips = sum(
                v.get("success", 0) for v in split_counts.values())
            self._qevd_log_append(
                f"\n✅  Ingestion complete — {len(mistake_poses)} mistake types, "
                f"{total_clips} total clips."
            )

            # ── Train ───────────────────────────────────────────────────────
            self._qevd_log_append("\nTraining mistake classifier (SVC)…")
            self.after(0, lambda: self._qevd_progress.set(0.75))

            ok = qevd_train_exercise(out_dir)

            self.after(0, lambda: self._qevd_progress.set(1.0))

            if ok:
                self._qevd_trained = True
                self._qevd_log_append(
                    f"\n🎉  Done!  mistake_classifier.pkl saved to {out_dir}.\n"
                    f"    Restart the app to see mistake feedback live."
                )
                self.after(0, lambda: self._qevd_status.configure(
                    text=f"✅ Trained — {len(mistake_poses)} mistake types",
                    text_color="#95d5b2"
                ))
            else:
                self._qevd_log_append(
                    "\n❌  Classifier training failed — check the log above."
                )
                self.after(0, lambda: self._qevd_status.configure(
                    text="❌ Training failed", text_color="#e63946"
                ))

            self.after(0, self._qevd_reset_button)

        threading.Thread(target=run, daemon=True).start()

    def _qevd_reset_button(self):
        self._qevd_run_btn.configure(
            state="normal",
            text="⚡ Ingest QEVD Data + Train Classifier"
        )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BuilderApp()
    app.mainloop()
