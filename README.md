# FallGuard Reproducible Fall-Prediction Pipeline

This repository contains the final reproducible pipeline used by FallGuard:

1. YOLO-pose extracts 17 COCO body keypoints from each video frame.
2. A HistGradientBoosting classifier produces the confirmed Normal, Pre-fall, and Fall decisions.
3. An ST-GCN + TCN skeleton-fusion model provides supporting evidence and earlier warnings.
4. HMM smoothing, dual-model decision logic, and a static-lying rule stabilize temporal predictions and reduce ADL false positives.

## Final Data and Models

- Raw videos: `data/videos`
- Final three-class annotations: `data/ur_up_train_drop60f_15pct_annotations.csv`
- Extracted features: `outputs/features`
- Upper-body landmarks: `outputs/landmarks_upperbody`
- Pose model: `models/yolo26n-pose.pt`
- Confirmed tree model: `models/yolo_tail60_prefall_accel_robust_classifier.joblib`
- Skeleton-fusion model: `models/skeleton_feature_fusion_tuned.pt`
- Five-fold fusion models: `models/cross_validation_tuned/fold_*_selected_full_outer.pt`
- Final five-fold report: `reports/dual_model_tuned_static_lying_postprocess_5fold_cv.json`

The final evaluation set contains 4,714 windows of 15 frames, 117 camera sequences, and 93 independent trial groups.

## Environment Setup

Python 3.11 is recommended. `requirements-lock.txt` records the exact package versions used to produce the final results.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
```

## Reproduce the Final Results

```bash
./scripts/reproduce_final_result.sh
```

The script:

- verifies SHA-256 checksums for the final annotations, models, fold models, and reports;
- retrains the tree classifier for each retained fold and regenerates complete out-of-fold predictions; and
- compares the regenerated report with the final reference report byte for byte.

To include the automated test suite:

```bash
./scripts/reproduce_final_result.sh --with-tests
```

Expected five-fold means:

| Output | Accuracy | Macro F1 | Pre-fall Precision | Pre-fall Recall | Fall Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confirmed decision + lying post-processing | 91.83% | 85.71% | 77.19% | 65.96% | 95.06% |
| Early warning + lying post-processing | 89.60% | 83.73% | 57.42% | 81.40% | 95.06% |

At the event level, the system detects 76 of 78 Fall camera sequences. The confirmed channel produces at least one false Fall in 5 of 39 non-Fall sequences.

## Retrain the Confirmed Tree Model

```bash
.venv/bin/python -m fall_prediction.train_model \
  --input-dir outputs/features \
  --output models/yolo_tail60_prefall_accel_robust_classifier.joblib \
  --metrics-output reports/yolo_tail60_prefall_accel_robust_metrics.json \
  --label-mode annotations \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --window-size 15 \
  --stride 3 \
  --classifier hist_gradient_boosting \
  --test-size 0 \
  --prefall-weight 8.0 \
  --prefall-alert-threshold 0.06 \
  --use-accel \
  --use-standing-calibration \
  --partial-pose-augmentation
```

## Repeat the Five-Fold Fusion Tuning

This process regenerates temporary candidate models and takes substantially longer than verifying the final report.

```bash
.venv/bin/python scripts/cross_validate_fusion.py \
  --input-dir outputs/features \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --output reports/fusion_grouped_5fold_cv.json

.venv/bin/python scripts/retrain_cv_full_outer.py \
  --source-report reports/fusion_grouped_5fold_cv.json \
  --input-dir outputs/features \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --output reports/fusion_grouped_5fold_cv_full_outer.json

.venv/bin/python scripts/tune_fusion_weight_calibration_cv.py \
  --source-report reports/fusion_grouped_5fold_cv_full_outer.json \
  --input-dir outputs/features \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --no-resume

./scripts/reproduce_final_result.sh
```

## Real-Time Inference

```bash
.venv/bin/python -m fall_prediction \
  --source 0 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor dual \
  --classifier-model models/yolo_tail60_prefall_accel_robust_classifier.joblib \
  --fusion-model models/skeleton_feature_fusion_tuned.pt \
  --use-accel \
  --show
```

The static-lying Normal correction remains a runtime post-processing rule and does not modify the training labels.
