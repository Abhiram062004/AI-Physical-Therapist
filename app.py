"""
app.py — AI Physical Therapist (Patient App)
=============================================
Run: streamlit run app.py

FIXES in this version
---------------------
1. Visibility array is now passed to RepCounter.update() so landmark
   confidence gating works — occluded limbs no longer trigger phantom reps.

2. Form feedback (check_form) is only shown BETWEEN reps (is_in_rep=False).
   During a rep the person is mid-movement, not at rest position, so comparing
   to the rest-position baseline always produced false positives. Between reps
   the form checker correctly flags posture issues.

3. Live score display: during a rep shows the live path score. Between reps
   shows the score of the last completed rep. This is clearer for the user.
"""

import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import streamlit as st
import time
import csv
import io
import mediapipe as mp

from core.exercise_loader import ExerciseLoader
from core.form_checker     import check_form
from core.rep_counter      import RepCounter
from calibration           import Calibrator
from gamification          import SessionTracker

# ── Optional voice ─────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 160)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

def speak(text):
    if TTS_AVAILABLE and st.session_state.get("voice_on"):
        try:
            _tts.say(text)
            _tts.runAndWait()
        except Exception:
            pass

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Physical Therapist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load exercises ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_exercises():
    loader = ExerciseLoader()
    return loader.load()

@st.cache_resource
def get_pose():
    return mp.solutions.pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.65
    )

exercises = load_exercises()
mp_draw   = mp.solutions.drawing_utils

# ── Session state ───────────────────────────────────────────────────────────────
def _init():
    ex_names = list(exercises.keys())
    defaults = {
        "exercise_name":   ex_names[0] if ex_names else None,
        "rep_counter":     None,
        "tracker":         SessionTracker(),
        "calibration":     Calibrator.load(),
        "running":         False,
        "session_start":   None,
        "voice_on":        False,
        "show_angles":     True,
        "last_form":       {"score": None, "feedback": [], "color": (0, 255, 0)},
        "rep_log":         [],
        "last_spoken":     "",
        "last_speak_time": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def maybe_speak(text):
    now = time.time()
    if (text != st.session_state["last_spoken"]
            or now - st.session_state["last_speak_time"] > 5):
        speak(text)
        st.session_state["last_spoken"]     = text
        st.session_state["last_speak_time"] = now

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏥 AI PT Assistant")
    st.markdown("---")

    ex_names = list(exercises.keys())

    if not ex_names:
        st.error("No exercises found.\nRun the Exercise Builder first.")
    else:
        chosen = st.selectbox(
            "Exercise", ex_names,
            index=ex_names.index(st.session_state["exercise_name"])
                  if st.session_state["exercise_name"] in ex_names else 0
        )
        if chosen != st.session_state["exercise_name"]:
            st.session_state["exercise_name"] = chosen
            st.session_state["rep_counter"]   = None

        target = st.number_input("Target reps", 1, 200, 10)

        st.session_state["voice_on"] = st.toggle(
            "🔊 Voice feedback",
            value=st.session_state["voice_on"],
            disabled=not TTS_AVAILABLE,
            help="pip install pyttsx3" if not TTS_AVAILABLE else "",
        )
        st.session_state["show_angles"] = st.toggle(
            "📐 Angle overlay", value=st.session_state["show_angles"]
        )

        st.markdown("---")
        st.subheader("📊 Session Stats")
        tracker = st.session_state["tracker"]
        sidebar_reps_ph   = st.empty()
        sidebar_form_ph   = st.empty()
        sidebar_rating_ph = st.empty()
        sidebar_reps_ph.metric("Reps",     tracker.session_reps)
        sidebar_form_ph.metric("Avg Form", f"{tracker.avg_form}%")
        sidebar_rating_ph.markdown(f"**Rating:** {'⭐' * tracker.accuracy_stars() or '–'}")

        if st.session_state["rep_log"]:
            buf = io.StringIO()
            w   = csv.DictWriter(buf, fieldnames=["time", "exercise", "form_score"])
            w.writeheader()
            w.writerows(st.session_state["rep_log"])
            st.download_button(
                "💾 Export CSV", buf.getvalue(),
                file_name="session.csv", mime="text/csv"
            )

        if st.button("💾 Save & End Session"):
            tracker.save_session(st.session_state["exercise_name"] or "Unknown")
            st.session_state["running"] = False
            st.success("Session saved!")

        st.markdown("---")
        st.subheader("🏆 Badges")
        from gamification import BADGES
        for b in tracker.earned_badges:
            icon = BADGES.get(b, {}).get("icon", "🎖️")
            st.markdown(f"{icon} **{b}**")
        if not tracker.earned_badges:
            st.caption("Complete reps to earn badges!")

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════
st.title("🏋️ AI-Powered Virtual Physical Therapist")

if not ex_names:
    st.info("No exercises loaded. Run `python exercise_builder/builder_app.py` to add exercises.")
    st.stop()

ex = exercises.get(st.session_state["exercise_name"])
if ex:
    c1, c2, c3 = st.columns(3)
    c1.metric("Exercise", ex.name)
    c2.metric("Muscles",  ", ".join(ex.muscle_groups[:2]) or "–")
    c3.metric("Status",   "✅ Trained" if ex.is_trained else "⚠️ Not trained")

col_vid, col_side = st.columns([3, 1])

with col_side:
    st.subheader("📋 Feedback")
    ex_label       = st.empty()
    reps_label     = st.empty()
    feedback_box   = st.empty()
    score_bar      = st.empty()
    st.subheader("🦴 Joint Breakdown")
    joint_breakdown = st.empty()
    st.subheader("📈 Progress")
    prog_bar     = st.empty()
    st.subheader("🏅 Badge")
    badge_box    = st.empty()

with col_vid:
    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("▶️ Start", type="primary", width='stretch')
    stop_btn  = c2.button("⏹ Stop",  width='stretch')
    reset_btn = c3.button("🔄 Reset", width='stretch')
    frame_ph  = st.empty()

if start_btn and ex:
    counter = RepCounter(ex)
    if not counter.is_ready:
        st.warning("⚠️ Rep counter not ready — exercise may not be fully trained.")
    st.session_state.update({
        "running":          True,
        "session_start":    time.time(),
        "tracker":          SessionTracker(),
        "rep_counter":      counter,
        "rep_log":          [],
        "_goal_announced":  False,   # reset so next goal triggers balloons again
    })

if stop_btn:
    st.session_state["running"] = False

if reset_btn and st.session_state.get("rep_counter"):
    st.session_state["rep_counter"].reset()

# ════════════════════════════════════════════════════════════════════════════════
#  VIDEO LOOP
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state["running"] and ex:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        st.session_state["running"] = False
    else:
        pose_estimator = get_pose()
        counter        = st.session_state["rep_counter"]
        rep_count      = 0

        no_detect_streak = 0          # consecutive frames with no valid detection
        NO_DETECT_RESET_FRAMES = 15   # frames of silence before wiping rep state

        while st.session_state["running"]:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = pose_estimator.process(rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                # Lowered from 0.6 → 0.5: brief arm/hand occlusion during a rep
                # used to trigger a full reset, killing in-progress reps.
                core_visibility = np.mean([lms[i].visibility for i in [11, 12, 23, 24]])
                person_detected = core_visibility >= 0.5
            else:
                person_detected = False

            if person_detected:
                no_detect_streak = 0   # reset the miss counter on any good frame
                lms        = res.pose_landmarks.landmark
                pose_frame = np.array([[lm.x, lm.y, lm.z] for lm in lms])
                # Extract visibility for all 33 landmarks
                visibility = np.array([lm.visibility for lm in lms])

                # ── Rep counting with visibility gating ───────────────────────
                rep_counted = False
                if counter and counter.is_ready:
                    rep_counted = counter.update(pose_frame, visibility)
                    if rep_counted:
                        rep_count += 1
                        score = counter.last_rep_score if counter.last_rep_score is not None else 0
                        st.session_state["tracker"].record_rep(score)
                        st.session_state["rep_log"].append({
                            "time":       round(time.time() - st.session_state["session_start"], 1),
                            "exercise":   ex.name,
                            "form_score": score,
                        })

                # ── Live score ────────────────────────────────────────────────
                # Score is per-rep only — shown after each completed rep.
                # During a rep: show "In Rep..." with a pulsing indicator.
                # Between reps: show the last completed rep score.
                if counter and counter.is_ready:
                    live_score = counter.current_form_score  # last rep score
                    is_in_rep  = counter.is_in_rep
                else:
                    live_score = None
                    is_in_rep  = False

                # ── Form feedback — only between reps ─────────────────────────
                # During a rep: person is mid-movement, skip form check
                # Between reps: compare against rest-position baseline
                form = check_form(ex, pose_frame, is_in_rep=is_in_rep)

                if not is_in_rep and live_score is None:
                    # Between reps — speak posture feedback if any
                    if form["feedback"] and form["feedback"][0] != "Good posture — ready to go!":
                        maybe_speak(form["feedback"][0])

                # ── Skeleton color ─────────────────────────────────────────────
                from core.form_checker import get_color
                if live_score is not None:
                    color = get_color(live_score)
                elif is_in_rep:
                    color = (200, 200, 0)   # yellow during unscored rep
                else:
                    color = (200, 200, 200)  # grey at rest

                custom = mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4)
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    custom, custom
                )

                st.session_state["last_form"] = {
                    "score":    live_score,
                    "feedback": form["feedback"],
                    "color":    color,
                    "is_in_rep": is_in_rep,
                }

            else:
                no_detect_streak += 1
                live_score = None
                is_in_rep  = False
                form       = {"feedback": ["No person detected — step into frame"]}
                st.session_state["last_form"] = {
                    "score": None, "feedback": form["feedback"],
                    "color": (128, 128, 128), "is_in_rep": False
                }
                # Only wipe rep state after N consecutive missed frames.
                # A single dropped frame (lighting glitch, fast movement) used to
                # immediately cancel a mid-rep, making reps impossible to count.
                if no_detect_streak >= NO_DETECT_RESET_FRAMES:
                    if counter and counter.is_ready:
                        counter.notify_no_detection()

            # ── HUD overlay ───────────────────────────────────────────────────
            h, w = frame.shape[:2]
            ov   = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 55), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

            elapsed = int(time.time() - (st.session_state["session_start"] or time.time()))
            mm, ss  = divmod(elapsed, 60)

            cv2.putText(frame, ex.name,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if not person_detected:
                cv2.putText(frame, "No Detection",
                            (w-230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            elif is_in_rep:
                cv2.putText(frame, "In Rep...",
                            (w-170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            elif counter and counter.last_rep_score is not None:
                sc     = counter.last_rep_score
                sc_col = (0,255,0) if sc >= 80 else (0,200,255) if sc >= 50 else (0,0,255)
                cv2.putText(frame, f"Last:{sc}%",
                            (w-190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc_col, 2)
            else:
                cv2.putText(frame, "Ready",
                            (w-110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.putText(frame, f"{mm:02d}:{ss:02d}",
                        (w//2-40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

            pct   = min(rep_count / max(target, 1), 1.0)
            bar_w = int((w - 20) * pct)
            cv2.rectangle(frame, (10, h-20), (w-10, h-8), (60, 60, 60), -1)
            if bar_w > 0:
                cv2.rectangle(frame, (10, h-20), (10+bar_w, h-8),
                              (0, 215, 255) if pct >= 1 else (0, 255, 100), -1)
            cv2.putText(frame, f"{rep_count}/{target}",
                        (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            frame_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB", width='stretch'
            )

            # ── Side panel ────────────────────────────────────────────────────
            ex_label.markdown(f"### 🏋️ `{ex.name}`")

            # Show goal-reached banner inline alongside rep count
            if rep_count >= target:
                reps_label.markdown(f"## 🎉 {rep_count} — Goal reached!")
            else:
                reps_label.markdown(f"## 🔁 {rep_count} / {target}")

            if not person_detected:
                feedback_box.warning("👤 No person detected — step into frame")
                score_bar.progress(0, text="Waiting for detection...")

            elif is_in_rep:
                feedback_box.markdown("🟡 Moving — score will appear when rep completes")
                score_bar.progress(
                    (counter.last_rep_score or 0) / 100,
                    text="Last rep: {}%".format(counter.last_rep_score or "—")
                )

            else:
                # Between reps — show posture feedback and last rep score
                if form["feedback"]:
                    icon = "🟢" if form["feedback"][0] == "Good posture — ready to go!" else "🟡"
                    fb   = "\n\n".join(f"{icon} {l}" for l in form["feedback"])
                    feedback_box.markdown(fb)
                else:
                    feedback_box.markdown("🟢 Ready")

                if counter and counter.last_rep_score is not None:
                    sc = counter.last_rep_score
                    score_bar.progress(sc / 100, text=f"Last rep: {sc}%")
                else:
                    score_bar.progress(0, text="Complete a rep to see your score")

            # ── Per-joint breakdown ───────────────────────────────────────────
            if counter and counter.last_rep_joint_scores:
                from core.landmarks import get_name as _get_name
                active_lines   = []
                inactive_names = []

                sorted_joints = sorted(
                    counter.last_rep_joint_scores.items(),
                    key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 999)
                )
                for joint_idx, jscore in sorted_joints:
                    name = _get_name(joint_idx)
                    if jscore is None:
                        # Joint doesn't move enough during this exercise to score
                        inactive_names.append(name)
                        continue
                    if   jscore >= 80: icon = "🟢"
                    elif jscore >= 60: icon = "🟡"
                    elif jscore >= 40: icon = "🟠"
                    else:              icon = "🔴"
                    bar = "█" * (jscore // 10) + "░" * (10 - jscore // 10)
                    active_lines.append(f"{icon} **{name}** {bar} {jscore}%")

                md = "\n\n".join(active_lines)
                if inactive_names:
                    md += f"\n\n*N/A (static): {', '.join(inactive_names)}*"
                joint_breakdown.markdown(md)
            elif not (counter and counter.last_rep_joint_scores):
                joint_breakdown.caption("Complete a rep to see joint scores")

            prog = st.session_state["tracker"].progress_to_next_milestone()
            prog_bar.progress(prog["pct"] / 100, text=prog["label"])

            if st.session_state["tracker"].new_badges:
                from gamification import BADGES
                b    = st.session_state["tracker"].new_badges[-1]
                icon = BADGES.get(b, {}).get("icon", "🎖️")
                badge_box.success(f"{icon} **{b}** unlocked!")

            _t = st.session_state["tracker"]
            sidebar_reps_ph.metric("Reps",     _t.session_reps)
            sidebar_form_ph.metric("Avg Form", f"{_t.avg_form}%")
            sidebar_rating_ph.markdown(f"**Rating:** {'⭐' * _t.accuracy_stars() or '–'}")

            if rep_count >= target:
                speak("Goal reached! Great work!")
                # Mark goal reached for balloons — but DON'T stop the loop.
                # Setting running=False here would kill the camera permanently
                # until the user presses Start again, which felt like a crash.
                # Instead we show balloons on the next Streamlit rerun by
                # storing a flag, then keep running so the user can continue.
                if not st.session_state.get("_goal_announced"):
                    st.session_state["_goal_announced"] = True
                    st.session_state["goal_reached"]    = True

        cap.release()
        # st.balloons() MUST be called outside the while loop.
        # Calling it inside causes Streamlit to try to rerun the script
        # while the loop is still blocking the main thread — instant crash.
        if st.session_state.pop("goal_reached", False):
            st.balloons()

else:
    frame_ph.markdown("""
        <div style="background:#1a1a2e;border-radius:12px;padding:80px;text-align:center;">
            <h2 style="color:#e0e0e0;">📷 Ready</h2>
            <p style="color:#aaa;">Select an exercise and press <b>▶️ Start</b></p>
            <p style="font-size:72px">🏋️</p>
        </div>
    """, unsafe_allow_html=True)

# ── History charts ─────────────────────────────────────────────────────────────
st.markdown("---")
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.subheader("📅 Session History")
    history = st.session_state["tracker"].history
    if history:
        import pandas as pd
        df = pd.DataFrame(history).sort_values("date")
        t1, t2 = st.tabs(["Reps", "Form"])
        with t1: st.bar_chart(df.set_index("date")["reps"])
        with t2: st.line_chart(df.set_index("date")["avg_form"])
    else:
        st.caption("No history yet.")

with col_h2:
    st.subheader("📊 Rep Quality This Session")
    if st.session_state["rep_log"]:
        import pandas as pd
        df2 = pd.DataFrame(st.session_state["rep_log"])
        st.line_chart(df2.set_index("time")["form_score"])
    else:
        st.caption("Form score per rep will appear here.")

st.markdown("---")
st.caption("🤖 AI Physical Therapist · MediaPipe + Scikit-learn · Streamlit")
