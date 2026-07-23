# Fall Prediction 最终可复现版本

本目录只保留当前 FallGuard 使用的最终链路：

1. YOLO-pose 从视频帧提取 17 个 COCO 人体关键点。
2. HistGradientBoosting 树模型负责正式确认 Normal / Pre-fall / Fall。
3. ST-GCN + TCN 骨架融合模型负责辅助与提前预警。
4. HMM、双模型决策和静态躺姿规则负责时序稳定与 ADL 后处理。

## 最终数据与模型

- 原始数据：data/videos
- 最终三分类标注：data/ur_up_train_drop60f_15pct_annotations.csv
- 最终特征：outputs/features
- 最终骨架关键点：outputs/landmarks_upperbody
- 姿态模型：models/yolo26n-pose.pt
- 正式树模型：models/yolo_tail60_prefall_accel_robust_classifier.joblib
- 辅助融合模型：models/skeleton_feature_fusion_tuned.pt
- 五折融合模型：models/cross_validation_tuned/fold_*_selected_full_outer.pt
- 最终五折报告：reports/dual_model_tuned_static_lying_postprocess_5fold_cv.json

数据规模为 4,714 个 15 帧窗口、117 条摄像机序列、93 个独立试验组。

## 建立环境

推荐使用 Python 3.11。requirements-lock.txt 保存了生成最终结果时的完整版本。

~~~
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
~~~

## 一键验证最终结果

~~~
./scripts/reproduce_final_result.sh
~~~

脚本会执行以下检查：

- 验证最终模型、折模型、标注和报告的 SHA-256。
- 使用保留的 5 个折模型重新训练每折树模型并生成完整折外预测。
- 将新报告与最终参考报告逐字节比较。

加入自动测试：

~~~
./scripts/reproduce_final_result.sh --with-tests
~~~

预期最终五折均值：

| 输出层 | Accuracy | Macro F1 | Pre-fall Precision | Pre-fall Recall | Fall Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 正式确认 + 躺姿后处理 | 91.83% | 85.71% | 77.19% | 65.96% | 95.06% |
| 提前预警 + 躺姿后处理 | 89.60% | 83.73% | 57.42% | 81.40% | 95.06% |

事件级结果：78 条 Fall 摄像机序列中检出 76 条；39 条非 Fall 序列中，正式确认通道有 5 条出现过错误 Fall。

## 重新训练正式树模型

~~~
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
~~~

## 从头重新完成融合模型五折调优

此过程会重新生成临时候选模型，耗时明显长于验证最终报告。

~~~
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
~~~

## 实时运行

~~~
.venv/bin/python -m fall_prediction \
  --source 0 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor dual \
  --classifier-model models/yolo_tail60_prefall_accel_robust_classifier.joblib \
  --fusion-model models/skeleton_feature_fusion_tuned.pt \
  --use-accel \
  --show
~~~

静态躺姿的 Normal 修正仍然属于运行时后处理，不会改写模型训练标签。

