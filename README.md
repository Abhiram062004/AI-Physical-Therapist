# AI-Powered Virtual Physical Therapist

An AI-powered exercise monitoring and physical therapy assistant that uses computer vision and machine learning to analyze body movements, monitor exercise repetitions, evaluate form, and provide real-time feedback.

## Overview

The AI Physical Therapist is designed to help users monitor their exercise performance through a webcam-based system.

Using MediaPipe Pose and machine learning techniques, the application tracks body landmarks, monitors exercise movements, evaluates repetition quality, and provides visual feedback.

The application also includes personalized calibration, workout progress tracking, and a gamification system to encourage consistent exercise.

## Key Features

- **Real-Time Pose Detection:** Detects and tracks body landmarks using MediaPipe Pose.
- **Exercise Repetition Counting:** Monitors exercise movements and counts completed repetitions.
- **Form Assessment:** Evaluates exercise form and provides posture-related feedback.
- **Form Scoring:** Displays exercise performance scores for completed repetitions.
- **Joint Visualization:** Displays a pose skeleton and joint-related information.
- **Personalized Calibration:** Calculates movement thresholds based on individual body movements.
- **Workout Tracking:** Tracks repetitions, average form scores, and workout progress.
- **Gamification:** Awards badges for achieving exercise milestones.
- **Voice Feedback:** Supports optional voice-based feedback.
- **Exercise Builder:** Provides tools for creating and training exercise configurations.
- **QEVD Dataset Integration:** Includes scripts for extracting pose data and training exercise mistake classifiers.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| OpenCV | Webcam and video processing |
| MediaPipe | Human pose estimation |
| NumPy | Numerical computations and pose data |
| Scikit-learn | Machine learning and SVM classification |
| Pandas | Workout history and data visualization |
| Joblib | Machine learning model serialization |

## Project Structure

```text
AI-Physical-Therapist/
│
├── app.py
├── adaptive_safety.py
├── calibration.py
├── diagnose_scoring.py
├── gamification.py
├── qevd_config_gen.py
├── qevd_explorer.py
├── qevd_ingestor.py
├── qevd_trainer.py
├── repair_exercise.py
│
├── core/
│   ├── exercise_loader.py
│   ├── form_checker.py
│   ├── landmarks.py
│   └── rep_counter.py
│
├── exercise_builder/
│   ├── builder_app.py
│   ├── form_trainer.py
│   ├── pose_extractor.py
│   ├── rep_annotator.py
│   ├── rep_trainer.py
│   └── exercises_data/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abhiram062004/AI-Physical-Therapist.git
```

### 2. Navigate to the Project Folder

```bash
cd AI-Physical-Therapist
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

If no exercises are loaded, use the Exercise Builder to create and train exercise configurations before running the patient application.

## Exercise Builder

The project includes an exercise-building and training workflow.

Run:

```bash
streamlit run exercise_builder/builder_app.py
```

The supporting scripts provide functionality for pose extraction, repetition annotation, exercise training, and configuration generation.

## Machine Learning Workflow

The project uses a computer vision and machine learning workflow:

1. Capture video through a webcam.
2. Detect human body landmarks using MediaPipe Pose.
3. Extract pose information from detected landmarks.
4. Analyze body movements and exercise repetitions.
5. Evaluate exercise form using scoring and classification logic.
6. Display feedback and track workout performance.

The QEVD integration scripts can also extract pose data from an external dataset and train mistake-classification models.

## Future Improvements

- Support for additional exercises.
- Improved exercise mistake classification.
- More personalized rehabilitation recommendations.
- Advanced workout analytics and reporting.
- Improved adaptive safety monitoring.
- Cloud-based workout history and progress tracking.

## Important Disclaimer

This project is intended for exercise monitoring and educational purposes. It is not a replacement for professional medical diagnosis, physiotherapy, or advice from a qualified healthcare professional.

Users with injuries or medical conditions should consult a qualified professional before performing exercises.

## Author

**Abhiram P S**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: [Abhiram062004](https://github.com/Abhiram062004)

Project Repository: [AI Physical Therapist](https://github.com/Abhiram062004/AI-Physical-Therapist)

---

If you find this project interesting, consider giving the repository a star!