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
│   ├── fall_prediction_desktop/   # 桌面应用（窗口、服务器）
│   └── fall_prediction/           # AI 核心（姿态识别、跌倒预测）
├── web/                           # 前端 UI（HTML/CSS/JS）
├── models/                        # AI 模型文件
├── assets/                        # 图标和应用资源
├── configs/                       # 配置文件
├── launch.command                 # 双击启动
├── build_app.sh                   # 打包 .app
└── pyproject.toml                 # 依赖配置
```

## App 图标

把设计好的图标保存为 `assets/FallGuard.png`（建议 `1024x1024` PNG）。

pywebview 模式下会自动读取这个 PNG 作为窗口图标；打包 `.app` 时，`build_app.sh` 会生成并使用 `FallGuard.icns`。

## 打包成 .app

```bash
./build_app.sh
```

构建结果在 `dist/FallGuard.app`，可以像普通 macOS 应用一样双击打开、拖入 Applications 文件夹。

## 当前功能

- Web Dashboard 风格界面，接近真实产品页面。
- 默认使用 YOLO 姿态识别 + 机器学习跌倒预测模型。
- 实时显示摄像头画面、骨架点、当前状态、风险分数、事件记录和帧率。

第一次启动摄像头时，macOS 可能会询问是否允许使用摄像头，请选择允许。

如果摄像头打不开：

- 在“系统设置 > 隐私与安全性 > 摄像头”中允许 FallGuard；如果用 `launch.command` 启动，也请允许 Terminal 或 Python。
- 关闭 FaceTime、Zoom、浏览器会议等可能正在占用摄像头的软件。
- 使用打包版本时，请打开 `dist/FallGuard.app`，不要直接运行 `dist/FallGuard/FallGuard` 里的内部可执行文件。

