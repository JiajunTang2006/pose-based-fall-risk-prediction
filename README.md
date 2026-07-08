# Fall Prediction 跌倒预测原型

这是一个基于 **姿态估计 + OpenCV 视频处理 + 规则风险评分 / 机器学习窗口分类** 的跌倒检测 / 跌倒前预警原型项目。默认使用 MediaPipe 提取人体关键点，也可以切换到 Ultralytics YOLO-pose。它会从摄像头或视频中提取人体关键点，把关键点转换成一些可解释的身体运动特征，然后按时间平滑后输出每一帧的状态：

- `Normal`：正常状态
- `Pre-fall`：疑似跌倒前状态
- `Fall`：疑似已经跌倒
- `Unknown`：没有检测到可靠人体姿态

默认运行时仍然可以使用可解释的规则系统；当前推荐的机器学习路径使用 YOLO-pose 特征、UR+UP 合并三分类标注、Fall tail 60 帧裁剪、加速度增强特征，以及已经训练好的滑动窗口分类器进行实时/离线预测。

## 项目结构

```text
fall_prediction/
  __main__.py        # python -m fall_prediction 的入口
  video_app.py       # 打开摄像头/视频、逐帧预测、写 CSV、保存标注视频
  pose.py            # MediaPipe / YOLO-pose 姿态估计封装，以及 OpenCV 骨架绘制
  landmarks.py       # MediaPipe 33 个人体关键点的索引和辅助函数
  features.py        # 从姿态关键点提取可解释特征
  risk.py            # 按阈值和权重计算风险分数
  predictor.py       # 组合特征、风险评分、基线、时间平滑，输出最终状态
  config.py          # 从 JSON 加载运行配置
  ml_features.py     # 机器学习特征列和窗口展开工具
  window_dataset.py  # 把特征 CSV 切成训练用时间窗口
  train_model.py     # 训练机器学习分类器并保存 joblib 模型
  ml_predictor.py    # 加载训练模型，按滑动窗口进行预测
  export_dataset_features.py # 批量把数据集视频/图片序列导出为特征 CSV
  plot_features.py   # 根据 CSV 画特征曲线

configs/
  default.json       # 默认阈值、权重、基线和平滑窗口配置，可用 --config 加载

models/
  pose_landmarker_full.task  # 新版 MediaPipe Tasks API 需要的姿态模型文件
  yolo26n-pose.pt            # 可选：YOLO-pose 轻量姿态模型

outputs/
  webcam.csv         # 示例/运行输出 CSV
  webcam.mp4         # 示例/运行输出标注视频

tests/
  test_features.py   # 测试特征提取逻辑
  test_predictor.py  # 测试合成跌倒序列能否触发预警/跌倒状态
```

## FallGuard macOS 摄像头监测初版

当前已经新增一个面向普通用户的轻量桌面项目：

```text
apps/macos/
  launch.command  # macOS 双击/命令行启动入口
  build_app.sh    # 后续用 PyInstaller 打包 .app 的初版脚本
  src/fall_prediction_desktop/
```

这个版本使用 Web 前端界面和本机 Python 后端，默认使用摄像头 `0`、YOLO 姿态识别和机器学习跌倒预测模型。打开 FallGuard 后只需要点击 `Start Monitoring`。

启动方式：

```bash
apps/macos/launch.command
```

详细说明见 `apps/macos/README.md`。

## 整体运行流程

执行：

```bash
python -m fall_prediction
```

实际入口是 `fall_prediction/__main__.py`，它会调用 `fall_prediction/video_app.py` 里的 `main()`。

程序的主要流程如下：

1. `video_app.py` 用 OpenCV 打开摄像头或视频文件。
2. 每读取一帧，就把画面交给 `pose.py` 里的姿态估计器。
3. MediaPipe 输出 33 个人体关键点；YOLO-pose 输出 COCO 17 点后会自动映射成项目内部的 33 点格式。
4. `features.py` 把关键点转换成身体姿态和运动特征。
5. `risk.py` 根据这些特征计算 0 到 1 之间的风险分数。
6. `predictor.py` 对风险分数做时间平滑，并要求连续多帧达到阈值后才改变状态。
7. `video_app.py` 可以把状态、风险值和关键特征写入 CSV，也可以把骨架和文字叠加到视频中。

## 核心特征含义

`FeatureExtractor` 会从每一帧的人体关键点中提取这些特征：

| 特征 | 含义 |
| --- | --- |
| `torso_angle_deg` | 躯干相对竖直方向的角度。接近 0 表示站得比较直，越大表示身体越倾斜/接近横向。 |
| `torso_angular_velocity` | 躯干角度变化速度。突然快速倾斜时会变大。 |
| `body_center_y` | 肩膀中点和髋部中点的平均 y 坐标，用来近似身体中心高度。 |
| `body_center_delta` | 当前身体中心 y 坐标相对上一帧的变化量。 |
| `vertical_velocity` | 身体中心的竖直速度。图像坐标里 y 越大越靠下，所以正值通常表示身体在往下掉。 |
| `aspect_ratio` | 可见人体关键点包围盒的宽高比。人倒下后，身体往往变“宽”，这个值会升高。 |
| `visibility_mean` | 肩、髋、膝、踝等关键点的平均可见度，用来判断姿态检测是否可靠。 |

这些特征都是归一化图像坐标下计算的，因此它们会受到摄像头角度、人物距离、遮挡和光照影响。

## 风险分数怎么计算

`risk.py` 里定义了 `RiskConfig` 和 `RiskScorer`。它不是直接判断“摔倒/没摔倒”，而是先给每个危险信号打分：

- 身体是否明显倾斜：`torso_score`
- 躯干是否快速旋转：`angular_velocity_score`
- 身体中心是否快速下降：`vertical_velocity_score`
- 身体中心是否比初始站立位置下降很多：`center_drop_score`
- 人体包围盒是否从竖向变得更横向：`aspect_ratio_score`

每个子分数都会通过 `ramp(value, low, high)` 转成 0 到 1：

- 低于 `low`：风险接近 0
- 高于 `high`：风险接近 1
- 中间：线性增长

然后代码按权重合成总风险：

```text
risk_score =
  0.22 * torso_score
+ 0.12 * angular_velocity_score
+ 0.34 * vertical_velocity_score
+ 0.16 * center_drop_score
+ 0.16 * aspect_ratio_score
```

最后还会乘上一个和 `visibility_mean` 相关的因子。也就是说，如果 MediaPipe 对关键点不够确定，风险分数会被压低，避免在姿态检测很差时乱报警。

默认阈值：

| 阈值 | 默认值 | 含义 |
| --- | --- | --- |
| `prefall_threshold` | `0.45` | 风险超过它，瞬时状态可能是 `Pre-fall`。 |
| `fall_threshold` | `0.72` | 风险超过它，瞬时状态可能是 `Fall`。 |

## 时间平滑和最终状态

`FallPredictor` 在 `predictor.py` 里负责把单帧风险变成稳定的状态。

它做了三件重要的事：

1. **建立初始基线**

   默认用前 `15` 帧的 `body_center_y` 求平均，作为人物站立时的身体中心高度。后续如果身体中心明显下降，就会增加 `center_drop_score`。

2. **平滑风险分数**

   默认保留最近 `5` 帧风险分数，计算平均值作为 `smoothed_risk_score`。这样可以减少单帧误检造成的状态闪烁。

3. **要求连续多帧触发**

   默认连续 `3` 帧超过 `prefall_threshold` 才输出 `Pre-fall`，连续 `3` 帧超过 `fall_threshold` 才输出 `Fall`。

如果当前没有检测到人体，或者关键点可见度太低，最终状态会变成 `Unknown`。

## 运行配置

默认参数写在代码里，也可以通过 JSON 配置文件覆盖。项目自带：

```text
configs/default.json
```

运行时传入：

```bash
python -m fall_prediction \
  --source data/videos/test.mp4 \
  --config configs/default.json \
  --output-csv outputs/test.csv
```

配置文件可以调整：

- `state_thresholds`：`prefall_threshold`、`fall_threshold`、`min_visibility`
- `temporal_smoothing`：`baseline_frames`、`smoothing_window`、连续触发帧数
- `risk_scoring`：各个特征的阈值和权重

## CSV 输出字段

运行后会生成类似 `outputs/webcam.csv` 的文件。每一行对应一帧：

| 字段 | 含义 |
| --- | --- |
| `frame` | 帧编号 |
| `time` | 当前帧时间，单位秒 |
| `state` | 时间平滑后的最终状态 |
| `alert_state` | 实时报警状态；ML 模式下可能比 `state` 更敏感 |
| `instant_state` | 只根据当前风险分数得到的瞬时状态 |
| `risk_score` | 当前帧原始风险分数 |
| `smoothed_risk_score` | 多帧平均后的风险分数 |
| `has_pose` | 是否检测到可靠人体姿态，1 表示检测到，0 表示未检测到 |
| `torso_angle` | 躯干倾斜角度 |
| `torso_angular_velocity` | 躯干角速度 |
| `body_center_y` | 身体中心 y 坐标 |
| `body_center_delta` | 身体中心相对上一帧的 y 坐标变化 |
| `vertical_velocity` | 身体中心竖直速度 |
| `aspect_ratio` | 人体包围盒宽高比 |
| `body_width` | 人体关键点包围盒宽度 |
| `body_height` | 人体关键点包围盒高度 |
| `visibility_mean` | 关键点平均可见度 |
| `center_drop` | 身体中心相对初始基线下降的幅度 |

`risk_score` 适合观察某一帧的危险程度，`smoothed_risk_score` 更接近程序最终判断依据。

## 安装环境

`pyproject.toml` 要求 Python 版本为：

```text
>=3.10,<3.13
```

MediaPipe 对 Python 3.10 和 3.11 通常更稳定。如果你的系统默认 Python 太新，建议单独建一个 Python 3.11 环境。

当前项目推荐使用 `mediapipe==0.10.14`。你刚才装到的 `mediapipe 0.10.35` 只暴露新版 `tasks` API，没有旧版 `mp.solutions.pose`，所以会出现 `module 'mediapipe' has no attribute 'solutions'` 这个报错。

```bash
cd "/Users/tangjiajun/Desktop/SURF2026/Fall prediction"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 会从当前项目的 `pyproject.toml` 安装运行依赖，避免依赖版本在两个文件里重复维护。如果要同时安装测试、lint 和类型检查工具：

```bash
python -m pip install -e ".[dev]"
```

如果你已经装过新版 MediaPipe，先卸载再按 `requirements.txt` 重装：

```bash
python -m pip uninstall mediapipe -y
python -m pip install -r requirements.txt
```

依赖包括：

- `opencv-python`：读取摄像头/视频、写视频、绘制文字和骨架
- `mediapipe==0.10.14`：人体姿态估计
- `matplotlib`：绘制 CSV 特征曲线
- `numpy` / `scikit-learn` / `joblib`：训练和加载机器学习分类器

## MediaPipe 模型文件

推荐的 `mediapipe==0.10.14` 会使用 `mp.solutions.pose`，不需要额外模型文件。新版 MediaPipe 可能使用 Tasks API，需要额外的姿态模型文件。项目默认会找：

```text
models/pose_landmarker_full.task
```

下载地址：

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

如果你使用的是旧版 `mp.solutions.pose`，代码会走旧接口，不需要这个 `.task` 文件。

## 可选：使用 YOLO-pose

项目默认仍然走 MediaPipe。如果你想试 YOLO-pose，可以安装可选依赖并使用轻量模型：

```bash
python -m pip install -U ultralytics
```

项目默认会找：

```text
models/yolo26n-pose.pt
```

单个视频或图片序列切换到 YOLO：

```bash
python -m fall_prediction \
  --source data/videos/fall-01-cam0-rgb \
  --image-fps 30 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --output-csv outputs/yolo_fall-01-cam0-rgb.csv \
  --output-video outputs/yolo_fall-01-cam0-rgb_annotated.mp4
```

如果要用 YOLO 特征重新训练模型，建议导出到单独目录，避免和 MediaPipe 特征混在一起：

```bash
python -m fall_prediction.export_dataset_features \
  --input-dir data/videos/urfall \
  --output-dir outputs/features/urfall_yolo \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --image-fps 30
```

然后把训练脚本的 `--input-dir` 改成 `outputs/features/urfall_yolo` 重新训练。轻量模型推荐先用 `yolo26n-pose.pt`；如果后面更看重精度、能接受更慢速度，可以再试更大的 pose 模型。

## 直接训练当前三分类模型

在项目根目录运行下面这条命令，会使用 `outputs/features` 下的 YOLO 特征 CSV 和三分类标注 `data/ur_up_train_drop60f_15pct_annotations.csv` 重新训练模型：

```bash
.venv/bin/python -m fall_prediction.train_model \
  --input-dir outputs/features \
  --output models/yolo_tail60_prefall_accel_classifier.joblib \
  --metrics-output reports/yolo_tail60_prefall_accel_metrics.json \
  --label-mode annotations \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --window-size 15 \
  --stride 3 \
  --classifier hist_gradient_boosting \
  --test-size 0 \
  --prefall-weight 8.0 \
  --prefall-alert-threshold 0.06 \
  --use-accel
```

`--test-size 0` 表示全部窗口都参与最终训练，不额外留验证集；适合生成实际推理用模型。如果要重新报告验证指标，需要把它改回一个验证比例。

## 使用摄像头运行

在项目根目录运行下面这条命令，可以用训练好的三分类模型打开默认摄像头：

```bash
.venv/bin/python -m fall_prediction \
  --source 0 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor ml \
  --classifier-model models/yolo_tail60_prefall_accel_classifier.joblib \
  --use-accel \
  --use-hmm \
  --sensitivity medium \
  --show
```

这条命令会使用 YOLO 姿态提取和当前训练好的 ML 模型。默认不会导出 CSV，窗口中按 `q` 可以退出。

如果你想保存逐帧结果或标注视频，可以使用下面这种带输出文件的运行方式：

```bash
.venv/bin/python -m fall_prediction \
  --source 0 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor ml \
  --classifier-model models/yolo_tail60_prefall_accel_classifier.joblib \
  --use-accel \
  --use-hmm \
  --sensitivity medium \
  --output-csv outputs/webcam.csv \
  --output-video outputs/webcam.mp4 \
  --show
```

参数说明：

- `--source 0`：使用默认摄像头
- `--output-csv outputs/webcam.csv`：保存逐帧预测结果；不写这个参数就不导出 CSV
- `--output-video outputs/webcam.mp4`：保存带骨架和文字标注的视频
- `--show`：显示实时预览窗口

预览窗口中按 `q` 可以退出。

## 使用视频文件运行

把视频放到项目里，例如：

```text
data/videos/test.mp4
```

然后运行：

```bash
python -m fall_prediction --source data/videos/test.mp4 --output-csv outputs/test.csv --output-video outputs/test_annotated.mp4
```

如果不需要输出视频，可以省略 `--output-video`。

## 使用图片序列运行

UR Fall 有些下载包不是 `.mp4`，而是一帧一张图片。只要一个目录里直接放着按帧编号命名的图片，也可以直接作为 `--source`：

```text
data/videos/fall-01-cam0-rgb/
  fall-01-cam0-rgb-001.png
  fall-01-cam0-rgb-002.png
  fall-01-cam0-rgb-003.png
```

运行方式：

```bash
python -m fall_prediction \
  --source data/videos/fall-01-cam0-rgb \
  --image-fps 30 \
  --output-csv outputs/fall-01-cam0-rgb.csv \
  --output-video outputs/fall-01-cam0-rgb_annotated.mp4
```

说明：

- `--source` 可以是视频文件，也可以是图片目录。
- `--image-fps 30` 表示把图片序列按 30fps 处理；如果你的数据集说明写的是其他帧率，可以改成对应数值。
- 对 UP Fall 这类文件名带时间戳的图片序列，可以传 `--image-fps 0`，程序会从图片文件名自动推断帧率。
- 输出的 `--output-video` 会把图片序列重新写成一个带骨架和状态文字的 mp4，方便检查效果。

## 当前机器学习模型和报告

### Stage 2 三分类模型

当前推荐模型使用：

- YOLO-pose 特征：`outputs/features/urfall_yolo` 和 `outputs/features/upfall_yolo`
- 三分类标注：`data/ur_up_train_drop60f_15pct_annotations.csv`
- 标签：`Normal`, `Pre-fall`, `Fall`
- 加速度增强特征：`torso_angular_accel`, `vertical_accel`
- 窗口设置：15 帧窗口，stride 为 3
- 分类器：`hist_gradient_boosting`
- Pre-fall 样本权重：8.0
- Pre-fall artifact 报警阈值：0.06（实时最终报警还会经过 `--sensitivity` 时序门控）
- HMM Viterbi 时序平滑（实时推理时可选）
- 三档时序门控（实时推理默认开启）：`Pre-fall` 必须从近期 `Normal` 启动并在短窗口内维持；`Fall` 必须来自近期 `Normal` 或已确认 `Pre-fall`；静态躺姿/低姿态开局会被压回 `Normal`

重新训练当前模型：

```bash
.venv/bin/python -m fall_prediction.train_model \
  --input-dir outputs/features \
  --output models/yolo_tail60_prefall_accel_classifier.joblib \
  --metrics-output reports/yolo_tail60_prefall_accel_metrics.json \
  --label-mode annotations \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --window-size 15 \
  --stride 3 \
  --classifier hist_gradient_boosting \
  --test-size 0 \
  --prefall-weight 8.0 \
  --prefall-alert-threshold 0.06 \
  --use-accel
```

`--test-size 0` 表示不再留验证集，全部窗口都参与最终训练；这个模型适合实际推理，但不能用来报告独立验证分数。

`--use-accel` 启用加速度增强特征（`torso_angular_accel` 和 `vertical_accel`），
这些二阶导数特征帮助模型捕捉运动变化的"拐点"。

### 报告文件

```text
reports/stage1_lab_report.pdf          # Stage 1 & 2 完整实验报告
reports/stage1_lab_report.tex          # LaTeX 源文件
reports/yolo_tail60_prefall_accel_metrics.json          # Stage 2 三分类模型
reports/yolo_tail60_prefall_all_data_metrics.json       # Stage 1
reports/normal_start_combined_drop_grid_strict_validation_comparison.csv  # 网格搜索
reports/normal_start_combined_drop_grid_strict_validation_comparison.png  # 热力图
reports/fall_motion_threshold_candidates.csv            # Fall 动态确认阈值候选
reports/current_model_runtime_high_eval.json            # 高敏时序门控运行评估
reports/current_model_runtime_medium_eval.json          # 中敏时序门控运行评估
reports/current_model_runtime_low_eval.json             # 低敏时序门控运行评估
```

### 实时推理

**推荐命令（YOLO + 三分类模型 + 加速度特征 + HMM 平滑）：**

```bash
.venv/bin/python -m fall_prediction \
  --source 0 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor ml \
  --classifier-model models/yolo_tail60_prefall_accel_classifier.joblib \
  --use-accel \
  --use-hmm \
  --sensitivity medium \
  --show
```

- `--use-hmm`：启用 HMM Viterbi 时序平滑，减少 Normal↔Fall 跳变和单帧误报
- `--use-accel`：启用训练时同样使用的加速度增强特征
- `--sensitivity high|medium|low`：选择运行时序门控敏感度。三档使用同一个分类器，主要区别在于 `Pre-fall` 持续确认窗口和 `Fall` 事件链确认规则；低敏不额外抬高 `Pre-fall` 概率阈值，避免把真实早期信号直接丢掉
- 时序门控默认开启，用来减少“稳定躺着/低姿态/坐下过程直接判 Fall/Pre-fall”的误报；`Pre-fall` 必须从近期 `Normal` 启动并维持一小段窗口，`Fall` 必须来自近期 `Normal` 或已确认 `Pre-fall`，静态躺姿开局会被压回 `Normal`
- `--disable-temporal-fall-validation`：关闭 Fall 时序确认层，恢复只按模型/HMM 输出判断
- 不传 `--use-hmm` 则使用独立 argmax 分类（Pre-fall 召回略高，但误报更多）

当前三档在现有特征 CSV 上的运行评估摘要。`Core Pre-fall recall` 使用报告里的
20% lenient 标准，只统计 Pre-fall 中间 60% 核心过渡区：

| 敏感度 | Pre-fall recall | Core Pre-fall recall | Fall recall | Fall 视频检出 | ADL false Fall |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 0.683 | 0.793 | 0.924 | 47/54 | 0/39 |
| medium | 0.521 | 0.626 | 0.924 | 47/54 | 0/39 |
| low | 0.683 | 0.793 | 0.869 | 47/54 | 0/39 |

这里的 `low` 不再通过抬高 `Pre-fall` threshold 或延长 Pre-fall 确认来降低敏感度；
它主要加强 `Fall` 的事件链确认，避免静态躺姿/低姿态直接触发强报警。

**运行视频文件：**

```bash
.venv/bin/python -m fall_prediction \
  --source data/videos/urfall/fall-01-cam0-rgb \
  --image-fps 30 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor ml \
  --classifier-model models/yolo_tail60_prefall_accel_classifier.joblib \
  --use-accel \
  --use-hmm \
  --sensitivity medium \
  --output-csv outputs/ml_fall-01-cam0.csv \
  --output-video outputs/ml_fall-01-cam0_annotated.mp4
```

ML 模式会同时输出 `state` 和 `alert_state`：`state` 是模型概率最高的分类
（启用 HMM 时为 Viterbi 平滑后的状态），`alert_state` 是实时报警层。
当前模型 artifact 里保存的旧 `Pre-fall` 报警阈值是 `0.06`；新的时序门控会再按
`--sensitivity` 做二次确认，因此单个低概率 `Pre-fall` 窗口不会直接显示报警。

如果新数据也是图片序列，把 `--source` 换成图片目录即可：

```bash
.venv/bin/python -m fall_prediction \
  --source data/videos/urfall/fall-01-cam0-rgb \
  --image-fps 30 \
  --pose-backend yolo \
  --yolo-model models/yolo26n-pose.pt \
  --predictor ml \
  --classifier-model models/yolo_tail60_prefall_accel_classifier.joblib \
  --use-accel \
  --use-hmm \
  --sensitivity medium \
  --output-csv outputs/ml_fall-01-cam0-rgb.csv \
  --output-video outputs/ml_fall-01-cam0-rgb_annotated.mp4
```

也可以继续使用原来的规则系统：

```bash
python -m fall_prediction --source data/videos/test.mp4 --predictor rule
```

## 绘制特征曲线

生成 CSV 后，可以把风险分数和几个关键特征画成曲线：

```bash
python -m fall_prediction.plot_features outputs/test.csv --output outputs/test_curves.png
```

这会画出：

- 平滑风险分数
- 躯干角度
- 竖直速度
- 人体包围盒宽高比

这些曲线可以帮助你观察跌倒过程中哪些特征先发生变化。

## 运行测试

建议在项目虚拟环境中运行测试：

```bash
cd "/Users/tangjiajun/Desktop/SURF2026/Fall prediction"
source .venv/bin/activate
python -m unittest discover -s tests
```

测试内容包括：

- 站立姿态的躯干角度应该接近竖直
- 身体往下移动时，`vertical_velocity` 应该为正
- 倾斜姿态应该产生更大的 `torso_angle`
- 一段合成的跌倒序列应该能触发 `Pre-fall` 或 `Fall`
- 特征 CSV 可以被切成训练窗口，并能从 UR Fall 风格文件名推断标签
- 风险评分阈值、可见度修正、JSON 配置加载、ML 推理元数据优先级
- CSV 输出行和 `CSV_COLUMNS` 声明保持一致
- 训练脚本会保存验证集指标 JSON

如果安装了开发依赖，也可以运行：

```bash
python -m pytest
python -m ruff check .
python -m mypy fall_prediction
```

## 目前的局限

- 这是研究/课程原型，不是经过大量真实数据训练和验证的医疗级系统。
- 只处理单人姿态，MediaPipe Tasks 配置里 `num_poses=1`。
- 判断结果依赖摄像头角度、光照、遮挡、人物距离和关键点质量。
- `configs/default.json` 已经可以覆盖主要阈值和时间参数，但还没有做复杂配置校验或自动调参。
- 默认文件名标签只能训练粗粒度的 `Normal` / `Fall` 分类；要做提前预警，需要手动标注 `Pre-fall` 帧区间。
- 输出 CSV 主要保存特征和预测结果，没有保存原始 33 个关键点。如果要训练 LSTM、GRU、TCN、Transformer 等模型，可以扩展 CSV 或另存关键点序列。

## 后续可以怎么改进

- 用不同角度、不同人、不同动作采集更多 CSV 和视频。
- 给每段视频标注真实状态，用 CSV 特征训练时序分类模型。
- 增加配置校验、实验配置版本号和自动评估脚本。
- 加入报警逻辑，例如声音提醒、弹窗、发送消息。
- 增加评估脚本，统计准确率、召回率、误报率和提前预警时间。
