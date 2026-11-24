"""
exercise_builder/rep_annotator.py
----------------------------------
CustomTkinter video annotation tool.

Changes:
  - Bug fix: video_poses now stores ALL frames (including None) so frame
    indices from annotations match exactly when rep_trainer slices them.
  - New: joint tracking selector dropdown at the top.
  - on_complete(annotations, video_poses, tracking_joint_index) — 3 args now.
  - After annotation: Close or Add Another Exercise buttons.
"""

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import threading
import time
import os
from tkinter import messagebox

from core.landmarks import get_name

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


class RepAnnotator(ctk.CTkToplevel):

    def __init__(self, parent, video_paths: list, key_joint_indices: list, on_complete):
        super().__init__(parent)

        self.title("Rep Annotator")
        self.geometry("1100x780")
        self.resizable(True, True)

        self.video_paths           = video_paths
        self.key_joint_indices     = key_joint_indices
        self.on_complete           = on_complete

        # Default tracking joint = first key joint
        self.tracking_joint_idx    = key_joint_indices[0] if key_joint_indices else 11

        self.current_video_idx     = 0
        self.annotations           = []
        self.video_poses           = []   # list of lists; includes None entries — DO NOT STRIP
        self.all_frames            = []
        self.pose_frames           = []   # parallel to all_frames (may contain None)
        self.current_frame_idx     = 0
        self.is_playing            = False
        self.pending_start         = None
        self.fps                   = 25
        self._stop_event           = threading.Event()
        self._loading              = False

        self._build_ui()
        self._load_video_async(0)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, height=90)
        top.pack(fill="x", padx=10, pady=(10, 0))
        top.pack_propagate(False)

        self.video_label = ctk.CTkLabel(top, text="Loading...",
                                         font=("Helvetica", 14, "bold"))
        self.video_label.pack(side="left", padx=15)

        # Joint tracking selector
        joint_frame = ctk.CTkFrame(top, fg_color="transparent")
        joint_frame.pack(side="left", padx=20)

        ctk.CTkLabel(joint_frame, text="Tracking joint:",
                     font=("Helvetica", 12)).pack(side="left", padx=(0, 8))

        joint_options = {get_name(idx): idx for idx in self.key_joint_indices}
        self._joint_var = ctk.StringVar(
            value=get_name(self.tracking_joint_idx)
        )
        self._joint_menu = ctk.CTkOptionMenu(
            joint_frame,
            variable=self._joint_var,
            values=list(joint_options.keys()),
            command=self._on_joint_change,
            width=200, height=32, font=("Helvetica", 12)
        )
        self._joint_menu.pack(side="left")
        self._joint_options_map = joint_options

        ctk.CTkLabel(
            joint_frame,
            text="  ← pick the joint whose movement best represents one rep",
            font=("Helvetica", 11), text_color="gray"
        ).pack(side="left", padx=5)

        self.rep_count_label = ctk.CTkLabel(top, text="Reps marked: 0",
                                             font=("Helvetica", 13))
        self.rep_count_label.pack(side="right", padx=15)

        # ── Main area ─────────────────────────────────────────────────────────
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = ctk.CTkLabel(main, text="Loading video...",
                                    width=720, height=480)
        self.canvas.pack(side="left", padx=(0, 10))

        sidebar = ctk.CTkFrame(main, width=300)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        # Playback
        ctk.CTkLabel(sidebar, text="Playback",
                     font=("Helvetica", 14, "bold")).pack(pady=(15, 5))

        pb = ctk.CTkFrame(sidebar)
        pb.pack(fill="x", padx=10, pady=5)

        self.play_btn = ctk.CTkButton(
            pb, text="Play", command=self._toggle_play,
            fg_color="#2d6a4f", hover_color="#1b4332"
        )
        self.play_btn.pack(side="left", expand=True, padx=5, pady=8)

        ctk.CTkButton(
            pb, text="Restart", command=self._restart,
            fg_color="#4a4e69", hover_color="#2b2d42"
        ).pack(side="right", expand=True, padx=5, pady=8)

        ctk.CTkLabel(sidebar, text="Seek", font=("Helvetica", 12)).pack()
        self.seek_slider = ctk.CTkSlider(sidebar, from_=0, to=100,
                                          command=self._seek)
        self.seek_slider.pack(fill="x", padx=10, pady=5)

        self.frame_label = ctk.CTkLabel(sidebar, text="Frame: 0 / 0",
                                         font=("Helvetica", 11))
        self.frame_label.pack()

        # Annotation
        ctk.CTkLabel(sidebar, text="Annotation",
                     font=("Helvetica", 14, "bold")).pack(pady=(20, 5))

        self.start_btn = ctk.CTkButton(
            sidebar, text="Mark Rep Start",
            command=self._mark_start,
            fg_color="#1d3557", hover_color="#457b9d",
            height=45, font=("Helvetica", 13, "bold"),
            state="disabled"
        )
        self.start_btn.pack(fill="x", padx=10, pady=5)

        self.end_btn = ctk.CTkButton(
            sidebar, text="Mark Rep End",
            command=self._mark_end,
            fg_color="#6b0f1a", hover_color="#c9184a",
            height=45, font=("Helvetica", 13, "bold"),
            state="disabled"
        )
        self.end_btn.pack(fill="x", padx=10, pady=5)

        self.status_label = ctk.CTkLabel(
            sidebar, text="Waiting for video to load...",
            font=("Helvetica", 11), text_color="gray"
        )
        self.status_label.pack(pady=5)

        ctk.CTkButton(
            sidebar, text="Delete Last Rep",
            command=self._delete_last,
            fg_color="#3d405b", hover_color="#2b2d42"
        ).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sidebar, text="Marked Reps",
                     font=("Helvetica", 13, "bold")).pack(pady=(15, 3))

        self.rep_listbox = ctk.CTkTextbox(sidebar, height=100,
                                           font=("Courier", 11))
        self.rep_listbox.pack(fill="x", padx=10)
        self.rep_listbox.configure(state="disabled")

        # Video navigation
        nav = ctk.CTkFrame(sidebar)
        nav.pack(fill="x", padx=10, pady=(15, 5))

        self.prev_btn = ctk.CTkButton(
            nav, text="Prev Video",
            command=lambda: self._change_video(-1),
            fg_color="#4a4e69", state="disabled"
        )
        self.prev_btn.pack(side="left", expand=True, padx=3)

        self.next_btn = ctk.CTkButton(
            nav, text="Next Video",
            command=lambda: self._change_video(1),
            fg_color="#4a4e69",
            state="normal" if len(self.video_paths) > 1 else "disabled"
        )
        self.next_btn.pack(side="right", expand=True, padx=3)

        ctk.CTkButton(
            sidebar, text="✅ Done Annotating",
            command=self._finish,
            fg_color="#2d6a4f", hover_color="#1b4332",
            height=45, font=("Helvetica", 13, "bold")
        ).pack(fill="x", padx=10, pady=(10, 5))

        self.loading_label = ctk.CTkLabel(
            sidebar, text="", font=("Helvetica", 11), text_color="#90e0ef"
        )
        self.loading_label.pack(pady=3)

    def _on_joint_change(self, selected_name: str):
        self.tracking_joint_idx = self._joint_options_map[selected_name]

    # ── Video loading ─────────────────────────────────────────────────────────
    def _load_video_async(self, idx: int):
        self._loading = True
        self.play_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.end_btn.configure(state="disabled")
        self.canvas.configure(text="Loading video...", image=None)
        self.loading_label.configure(text="Extracting poses — please wait...")

        path = self.video_paths[idx]
        self.video_label.configure(
            text=f"Video {idx+1}/{len(self.video_paths)}: {os.path.basename(path)}"
        )

        def run():
            try:
                frames, poses = _extract_video(path)
                self.after(0, lambda: self._on_video_loaded(idx, frames, poses))
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_load_error(err))

        threading.Thread(target=run, daemon=True).start()

    def _on_video_loaded(self, idx, frames, poses):
        self.all_frames        = frames
        self.pose_frames       = poses
        self.current_frame_idx = 0
        self._loading          = False

        # ── CRITICAL FIX: store ALL poses including None so that annotation
        # frame indices (which point to the full frame list) stay aligned. ──
        while len(self.video_poses) <= idx:
            self.video_poses.append([])
        self.video_poses[idx] = poses   # keep Nones — rep_trainer filters them

        valid_count = sum(1 for p in poses if p is not None)
        total       = len(frames)
        self.seek_slider.configure(to=max(1, total - 1))
        self.seek_slider.set(0)
        self.loading_label.configure(
            text=f"{total} frames  |  {valid_count} with pose detected"
        )
        self.play_btn.configure(state="normal")
        self.start_btn.configure(state="normal")
        self.status_label.configure(
            text="Ready — press Mark Rep Start\nat the beginning of a rep",
            text_color="gray"
        )
        self._show_frame(0)

    def _on_load_error(self, error: str):
        self._loading = False
        self.canvas.configure(text=f"Failed to load video:\n{error}", image=None)
        self.loading_label.configure(text="Load failed", text_color="#e63946")

    # ── Playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self._loading or not self.all_frames:
            return
        self.is_playing = True
        self.play_btn.configure(text="Pause")
        self._stop_event.clear()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _stop_playback(self):
        self.is_playing = False
        self._stop_event.set()
        try:
            self.play_btn.configure(text="Play")
        except Exception:
            pass

    def _play_loop(self):
        delay = 1.0 / max(self.fps, 1)
        while not self._stop_event.is_set():
            if self.current_frame_idx >= len(self.all_frames) - 1:
                self.after(0, self._stop_playback)
                break
            self.current_frame_idx += 1
            idx = self.current_frame_idx
            self.after(0, lambda i=idx: self._show_frame(i))
            time.sleep(delay)

    def _show_frame(self, idx: int):
        if not self.all_frames:
            return
        idx = max(0, min(idx, len(self.all_frames) - 1))
        self.current_frame_idx = idx

        frame = self.all_frames[idx].copy()

        if self.pending_start is not None:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 6), (0, 255, 0), -1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (720, 480))
        img       = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        self.canvas.configure(image=img, text="")
        self.canvas.image = img

        try:
            self.seek_slider.set(idx)
            self.frame_label.configure(
                text=f"Frame: {idx} / {len(self.all_frames) - 1}"
            )
        except Exception:
            pass

    def _restart(self):
        self._stop_playback()
        self._show_frame(0)

    def _seek(self, value):
        self._stop_playback()
        self._show_frame(int(float(value)))

    # ── Annotation ────────────────────────────────────────────────────────────
    def _mark_start(self):
        self._stop_playback()
        self.pending_start = self.current_frame_idx
        self.status_label.configure(
            text=f"Start marked at frame {self.pending_start}\n"
                 f"Seek to end of rep and press Mark End.",
            text_color="#90e0ef"
        )
        self.end_btn.configure(state="normal")
        self.start_btn.configure(state="disabled")

    def _mark_end(self):
        if self.pending_start is None:
            return
        end = self.current_frame_idx
        if end <= self.pending_start:
            self.status_label.configure(
                text="End frame must be after start frame!",
                text_color="#e63946"
            )
            return

        self.annotations.append({
            "video_index": self.current_video_idx,
            "start_frame": self.pending_start,
            "end_frame":   end,
        })
        self.pending_start = None
        self.end_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.status_label.configure(
            text=f"Rep {len(self.annotations)} saved! Mark next rep.",
            text_color="#95d5b2"
        )
        self._update_rep_list()

    def _delete_last(self):
        if not self.annotations:
            return
        self.annotations.pop()
        self.pending_start = None
        self.end_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.status_label.configure(text="Last rep deleted.", text_color="gray")
        self._update_rep_list()

    def _update_rep_list(self):
        self.rep_count_label.configure(
            text=f"Reps marked: {len(self.annotations)}"
        )
        self.rep_listbox.configure(state="normal")
        self.rep_listbox.delete("1.0", "end")
        for i, ann in enumerate(self.annotations):
            self.rep_listbox.insert(
                "end",
                f"Rep {i+1}: V{ann['video_index']+1} "
                f"[{ann['start_frame']} → {ann['end_frame']}]\n"
            )
        self.rep_listbox.configure(state="disabled")

    # ── Navigation ────────────────────────────────────────────────────────────
    def _change_video(self, direction: int):
        if self._loading:
            return
        self._stop_playback()
        new_idx = self.current_video_idx + direction
        if 0 <= new_idx < len(self.video_paths):
            self.current_video_idx = new_idx
            self._load_video_async(new_idx)

        self.prev_btn.configure(state="normal" if new_idx > 0 else "disabled")
        self.next_btn.configure(
            state="normal" if new_idx < len(self.video_paths) - 1 else "disabled"
        )

    # ── Finish ────────────────────────────────────────────────────────────────
    def _finish(self):
        self._stop_playback()
        if len(self.annotations) < 5:
            messagebox.showwarning(
                "Not enough reps",
                f"Please annotate at least 5 reps.\n"
                f"Currently annotated: {len(self.annotations)}"
            )
            return
        # Pass tracking joint index as 3rd argument
        self.on_complete(self.annotations, self.video_poses, self.tracking_joint_idx)
        self.destroy()


# ── Video extraction ──────────────────────────────────────────────────────────
def _extract_video(path: str):
    """
    Extract all frames and MediaPipe pose landmarks from a video.
    Returns (frames, poses) where poses is parallel to frames and may
    contain None where MediaPipe detected no landmarks.
    """
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap    = cv2.VideoCapture(path)
    frames = []
    poses  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_estimator.process(rgb)

        if res.pose_landmarks:
            row = []
            for lm in res.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            poses.append(np.array(row).reshape(33, 3))

            mp_draw.draw_landmarks(
                frame, res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2),
            )
        else:
            poses.append(None)   # keep None — preserves frame index alignment

        frames.append(frame)

    cap.release()
    pose_estimator.close()
    return frames, poses
