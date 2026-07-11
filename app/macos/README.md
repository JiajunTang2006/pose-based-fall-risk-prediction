# FallGuard — Native macOS Desktop App

FallGuard 是一个完全独立的 macOS 桌面应用，使用 AI 驱动的跌倒检测技术，通过摄像头实时监测人体姿态，
在跌倒发生时及时预警。

## 为什么这样设计

- 使用 HTML/CSS/JavaScript 做前端界面，接近真实软件产品。
- 使用 Python 做本机后端，负责摄像头、YOLO 和跌倒预测。
- 视频不会上传外网，只在本机 `127.0.0.1` 页面中显示。
- **默认使用 pywebview 打开原生 macOS 桌面窗口**（不需要浏览器）。
- **完全自包含，不依赖外部项目** —— 所有代码和模型都在本目录内。
- 可以先用 `launch.command` 双击运行，后面再用 `build_app.sh` 打包成 `.app`。

## 快速开始

```bash
# 1. 进入应用目录
cd apps/macos

# 2. 创建虚拟环境并安装
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e "."

# 3. 启动桌面应用
./launch.command
```

或者在 Finder 里双击 `launch.command`，会打开一个**原生 macOS 桌面窗口**。

## 启动选项

| 命令 | 效果 |
|------|------|
| `python -m fall_prediction_desktop` | **默认** — 原生桌面窗口 |
| `python -m fall_prediction_desktop --menubar` | 以菜单栏应用运行 |
| `python -m fall_prediction_desktop --connect http://127.0.0.1:8765/` | 连接到已运行的本地监控服务 |
| `python -m fall_prediction --source video.mp4 --pose-backend yolo --predictor ml --output-video annotated.mp4` | 命令行处理视频/图片序列 |

## 目录结构

```
macos/
├── src/
│   ├── fall_prediction_desktop/   # 产品层（窗口、数据库、事件、通知、媒体与任务编排）
│   └── fall_prediction/           # 实验算法层（姿态识别、特征、模型与跌倒预测）
├── web/                           # 前端 UI（HTML/CSS/JS）
├── models/                        # AI 模型文件
├── assets/                        # 图标和应用资源
├── configs/                       # 配置文件
├── launch.command                 # 双击启动
├── build_app.sh                   # 打包 .app
└── pyproject.toml                 # 依赖配置
```

## 实验代码替换边界

为了方便后续把实验部分的新算法替换进应用，工程分成两个明确层级：

### 可以替换的实验算法层

实验代码集中在 `src/fall_prediction/`，主要替换点如下：

| 内容 | 文件/目录 | 对产品层的约束 |
|------|-----------|----------------|
| 姿态估计器 | `pose.py`、`landmarks.py` | 保持 `process_bgr(frame, timestamp_ms)` 返回关键点序列或 `None` |
| 特征提取 | `features.py`、`ml_features.py` | 最终仍生成 `PoseFeatures`，新增字段尽量提供默认值 |
| ML 推理模型 | `ml_predictor.py`、`models/*.joblib` | `predict()` 必须返回统一的 `Prediction` |
| 规则预测器 | `predictor.py`、`risk.py` | `Prediction` 是算法层与产品层之间的稳定合同 |
| 时序实验 | `robustness.py`、HMM/Fall validator 相关代码 | 对外只通过 `Prediction.state`、`alert_state`、`risk_score` 输出 |

`Prediction` 的稳定输出字段至少包括：

- `state`：模型分类状态；
- `alert_state`：经过实验时序门控后的报警状态；
- `risk_score`：范围 `0.0～1.0`；
- `features.visibility_mean`：范围 `0.0～1.0`；
- `system_status`：可选的校准或模型状态提示。

### 不应随实验代码一起覆盖的产品层

以下内容位于 `src/fall_prediction_desktop/`，负责数据库、Session、FSM、事件合并、媒体证据、通知、导出和 UI。替换实验代码时不要整体覆盖这些文件：

- `frame_pipeline.py`：实验预测结果进入产品业务状态的唯一适配边界；
- `event_service.py`：一段连续风险只生成一个业务事件；
- `event_media_buffer.py`：事件截图和前后视频缓冲；
- `database/`：用户历史数据和 schema；
- `ui/`、`web_app.py`：桌面 UI、摄像头和导入任务生命周期。

Camera 和 Import Media 都通过 `FrameBusinessProcessor` 消费统一的 `Prediction`。后续实验算法发生变化时，应优先在 `src/fall_prediction/` 内完成适配；只有输出合同确实变化时，才修改 `frame_pipeline.py`，不要分别修改 Camera 和 Import 两条路径。

### 推荐替换流程

1. 将新实验模型放入 `models/`，不要覆盖旧模型，先使用新文件名。
2. 在 `ml_predictor.py` 或新的 predictor 模块中把实验输出转换成统一 `Prediction`。
3. 保持 `video_app.create_predictor()` 的创建入口稳定，必要时增加新的 predictor 类型或配置。
4. 先运行算法层测试，再运行桌面层全量测试。
5. 用同一段短视频分别走 CLI Import 和桌面 Import，确认状态、风险、事件数量一致。
6. 验证完成后再修改默认模型路径和打包清单。

## App 图标

把设计好的图标保存为 `assets/FallGuard.png`（建议 `1024x1024` PNG）。

pywebview 模式下会自动读取这个 PNG 作为窗口图标；打包 `.app` 时，`build_app.sh` 会生成并使用 `FallGuard.icns`。

## 打包成 .app

```bash
./build_app.sh
```

构建结果在 `dist/FallGuard.app`，可以像普通 macOS 应用一样双击打开、拖入 Applications 文件夹。

构建脚本默认不会覆盖桌面上的旧版本。确认构建成功后，如需自动复制到桌面，可运行：

```bash
DEPLOY_TO_DESKTOP=1 ./build_app.sh
```

## 数据存储

- SQLite、设置和 Profiles：`~/Library/Application Support/FallGuard/`
- 导入媒体和输出视频：优先使用 `~/Movies/FallGuard/`
- 模型、图标和前端资源只读保存在 `.app` 内，不会把运行数据写入应用包，因此不会破坏代码签名。

## 当前功能

- Web Dashboard 风格界面，接近真实产品页面。
- 默认使用 YOLO 姿态识别 + 机器学习跌倒预测模型。
- 实时显示摄像头画面、骨架点、当前状态、风险分数、事件记录和帧率。
- Camera 与 Import Media 共用同一业务帧处理层，统一生成 Session、风险样本和事件媒体证据。
- 确认风险状态后可以播放本地声音；菜单栏图标提供显示、开始、停止和退出操作。
- macOS 系统通知与登录自启动目前暂不实现，设置页保留禁用的扩展位置。
- 默认加载上半身增强模型，支持站立基准校准和部分遮挡特征。
- Fall 必须经过近期 Normal → 已确认 Pre-fall → Fall 的连续事件链。
- 已确认 Fall 会保持到持续 Normal 恢复，避免 Fall/Normal 状态闪烁。

当前模型文件：`models/yolo_tail60_prefall_accel_upperbody_classifier.joblib`。

仅上半身可见时可以继续判断；人体关键点完全丢失超过短暂容忍窗口时显示 Unknown。

第一次启动摄像头时，macOS 可能会询问是否允许使用摄像头，请选择允许。

如果摄像头打不开：

- 在“系统设置 > 隐私与安全性 > 摄像头”中允许 FallGuard；如果用 `launch.command` 启动，也请允许 Terminal 或 Python。
- 关闭 FaceTime、Zoom、浏览器会议等可能正在占用摄像头的软件。
- 使用打包版本时，请打开 `dist/FallGuard.app`，不要直接运行 `dist/FallGuard/FallGuard` 里的内部可执行文件。
