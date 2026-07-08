# Pose-Based Fall and Pre-Fall Prediction System

This project explores a pose-based fall and pre-fall prediction system using computer vision and machine learning.

The current prototype extracts human pose features from video input and predicts three states: **Normal**, **Pre-fall**, and **Fall**. The goal is to build an early-warning prototype rather than only detecting falls after they happen.

## Dataset

The current prototype uses data from:

- **UR Fall Detection Dataset**
- **UP-Fall Detection Dataset**

The merged dataset contains **117 videos** and **273 labeled intervals**, which are converted into **4,714 sliding-window samples** for Normal / Pre-fall / Fall prediction.

Pre-fall samples are relatively limited and remain the most challenging part of the task.

## Current Progress

- Built a pose-based fall prediction pipeline
- Used YOLO-pose features for human pose extraction
- Converted video sequences into 15-frame sliding windows
- Trained a machine learning model for Normal / Pre-fall / Fall prediction
- Added motion features such as velocity and acceleration
- Explored temporal modeling to reduce unstable predictions
- Designed multiple sensitivity thresholds for different warning needs
- Developed a macOS prototype application using Python

## Current Focus

The current work focuses on improving model robustness, especially reducing false positives caused by lying-posture cases.

I am also exploring temporal modeling and threshold-based sensitivity control to make the system more practical for different use scenarios.

## Pipeline

1. Video / image input
2. Pose estimation
3. Feature extraction
4. Sliding-window construction
5. State prediction
6. Temporal smoothing
7. Warning output and visualization

## Tools

- Python
- YOLO-pose
- OpenCV
- pandas / NumPy
- scikit-learn
- HMM / Viterbi smoothing
- macOS prototype application

## Notes

This repository is under active development. 
