from __future__ import annotations

import base64
import csv
import queue
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .runner import ensure_repo_on_path, find_app_root


BG = "#f3f6f8"
PANEL = "#ffffff"
TEXT = "#17212b"
MUTED = "#667789"
BORDER = "#d7dee7"
VIDEO_BG = "#0b1117"
BLUE = "#176b87"
BLUE_DARK = "#0f4f65"
GREEN = "#1b7f4c"
AMBER = "#b7791f"
RED = "#b42318"
GRAY = "#5b6673"

DISPLAY_WIDTH = 860
DISPLAY_HEIGHT = 500
CAMERA_INDEX = 0


STATE_TEXT = {
    "Normal": "状态正常",
    "Pre-fall": "疑似跌倒风险",
    "Fall": "检测到跌倒",
    "Unknown": "未检测到人体",
}

STATE_DETAIL = {
    "Normal": "画面中的人体姿态看起来稳定。",
    "Pre-fall": "系统检测到失衡迹象，请关注画面。",
    "Fall": "系统检测到可能已经跌倒，请立即确认现场情况。",
    "Unknown": "请确认摄像头能看到完整人体。",
}

STATE_COLOR = {
    "Normal": GREEN,
    "Pre-fall": AMBER,
    "Fall": RED,
    "Unknown": GRAY,
}


@dataclass(frozen=True)
class CameraOutputs:
    csv_path: Path | None
    video_path: Path | None


class FallPredictionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FallGuard")
        self.minsize(1120, 760)
        self.geometry("1180x800")
        self.configure(bg=BG)

        self.app_root = find_app_root()
        ensure_repo_on_path(self.app_root)
        self.app_dir = self.app_root
        self.output_dir = self.app_root / "outputs" / "camera_sessions"

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._preload_worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.current_image: tk.PhotoImage | None = None
        self.last_outputs: CameraOutputs | None = None

        self.status_var = tk.StringVar(value="准备就绪")
        self.status_detail_var = tk.StringVar(value="点击“开始监测”即可打开摄像头。")
        self.camera_var = tk.StringVar(value="摄像头未启动")
        self.model_var = tk.StringVar(value="YOLO 姿态识别 + 机器学习跌倒预测")
        self.fps_var = tk.StringVar(value="实时帧率 --")
        self.record_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value=str(self.output_dir))

        self._configure_style()
        self._load_window_icon()
        self._build_menu()
        self._build_ui()
        self._start_model_preload()
        self.after(80, self._poll_events)

    def _start_model_preload(self) -> None:
        if self._preload_worker and self._preload_worker.is_alive():
            return
        self._preload_worker = threading.Thread(target=self._preload_models, daemon=True)
        self._preload_worker.start()

    def _preload_models(self) -> None:
        try:
            from fall_prediction.ml_predictor import load_model_artifact
            from fall_prediction.pose import preload_yolo_model

            preload_yolo_model(self.app_root / "models" / "yolo26n-pose.pt", warmup=True)
            load_model_artifact(self.app_root / "models" / "yolo_tail60_prefall_accel_classifier.joblib")
        except Exception:
            return

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("aqua")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Helvetica Neue", 26, "bold"))
        style.configure("Subtle.TLabel", background=BG, foreground=MUTED, font=("Helvetica Neue", 12))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT, font=("Helvetica Neue", 16, "bold"))
        style.configure("PanelText.TLabel", background=PANEL, foreground=MUTED, font=("Helvetica Neue", 12))
        style.configure("StatusTitle.TLabel", background=PANEL, foreground=TEXT, font=("Helvetica Neue", 22, "bold"))
        style.configure("StatusDetail.TLabel", background=PANEL, foreground=MUTED, font=("Helvetica Neue", 13))
        style.configure("Primary.TButton", font=("Helvetica Neue", 15, "bold"), padding=(24, 11))
        style.configure("Stop.TButton", font=("Helvetica Neue", 15, "bold"), padding=(24, 11))
        style.configure("Secondary.TButton", padding=(14, 8))
        style.configure("TCheckbutton", background=PANEL, font=("Helvetica Neue", 12))

    def _load_window_icon(self) -> None:
        icon_path = self.app_dir / "assets" / "FallGuard.png"
        if not icon_path.exists():
            return
        try:
            icon = tk.PhotoImage(file=str(icon_path))
        except tk.TclError:
            return
        self.iconphoto(True, icon)
        self._window_icon = icon

    def _build_menu(self) -> None:
        menu = tk.Menu(self)

        app_menu = tk.Menu(menu, tearoff=False)
        app_menu.add_command(label="开始监测", command=self._start_monitoring)
        app_menu.add_command(label="停止监测", command=self._stop_monitoring)
        app_menu.add_separator()
        app_menu.add_command(label="打开记录文件夹", command=self._open_output_dir)
        app_menu.add_separator()
        app_menu.add_command(label="退出 FallGuard", command=self.destroy)
        menu.add_cascade(label="监测", menu=app_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="关于", command=self._show_about)
        menu.add_cascade(label="帮助", menu=help_menu)
        self.config(menu=menu)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=(24, 22))
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="FallGuard", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="打开摄像头后，系统会自动使用 YOLO 模型进行实时识别。", style="Subtle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.start_button = ttk.Button(header, text="开始监测", style="Primary.TButton", command=self._start_monitoring)
        self.start_button.grid(row=0, column=1, rowspan=2, padx=(18, 10), sticky="e")
        self.stop_button = ttk.Button(header, text="停止", style="Stop.TButton", command=self._stop_monitoring)
        self.stop_button.grid(row=0, column=2, rowspan=2, sticky="e")
        self.stop_button.state(["disabled"])

        content = ttk.Frame(root, style="App.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_camera_panel(content)
        self._build_side_panel(content)

    def _build_camera_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        top = ttk.Frame(panel, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        top.columnconfigure(0, weight=1)
        ttk.Label(top, textvariable=self.status_var, style="StatusTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.status_dot = tk.Canvas(top, width=18, height=18, bg=PANEL, highlightthickness=0)
        self.status_dot.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.status_dot_id = self.status_dot.create_oval(3, 3, 15, 15, fill=GRAY, outline=GRAY)
        ttk.Label(top, textvariable=self.status_detail_var, style="StatusDetail.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        video_frame = tk.Frame(panel, bg=VIDEO_BG, highlightthickness=0)
        video_frame.grid(row=1, column=0, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_frame,
            text="摄像头画面将在这里显示",
            bg=VIDEO_BG,
            fg="#c7d2de",
            font=("Helvetica Neue", 18, "bold"),
            width=36,
            height=14,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(panel, style="Panel.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.camera_var, style="PanelText.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.fps_var, style="PanelText.TLabel").grid(row=0, column=1, sticky="e")

    def _build_side_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(8, weight=1)

        ttk.Label(panel, text="使用方式", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            panel,
            text="1. 确保摄像头能看到完整人体\n2. 点击开始监测\n3. 看到红色提示时立即确认现场",
            style="PanelText.TLabel",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 20))

        self._info_row(panel, 2, "AI 模型", self.model_var)
        self._info_row(panel, 3, "记录位置", self.output_var)

        options = ttk.Frame(panel, style="Panel.TFrame")
        options.grid(row=4, column=0, sticky="ew", pady=(6, 18))
        ttk.Checkbutton(options, text="自动保存检测记录", variable=self.record_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="同时保存标注视频", variable=self.save_video_var).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=5, column=0, sticky="ew", pady=(0, 20))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="打开记录文件夹", style="Secondary.TButton", command=self._open_output_dir).grid(
            row=0, column=0, sticky="ew"
        )

        ttk.Label(panel, text="事件记录", style="PanelTitle.TLabel").grid(row=6, column=0, sticky="w")
        ttk.Label(panel, text="系统只记录状态变化。", style="PanelText.TLabel").grid(row=7, column=0, sticky="w", pady=(4, 10))

        log_holder = tk.Frame(panel, bg="#101820")
        log_holder.grid(row=8, column=0, sticky="nsew")
        log_holder.columnconfigure(0, weight=1)
        log_holder.rowconfigure(0, weight=1)
        self.event_log = tk.Text(
            log_holder,
            bg="#101820",
            fg="#d7e3ee",
            relief="flat",
            wrap="word",
            state="disabled",
            font=("Menlo", 11),
            padx=10,
            pady=10,
            height=12,
        )
        self.event_log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_holder, command=self.event_log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.event_log.configure(yscrollcommand=scrollbar.set)
        self._append_event("系统已准备就绪。")

    def _info_row(self, parent: ttk.Frame, row: int, title: str, variable: tk.StringVar) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title, style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=variable, style="PanelText.TLabel", wraplength=265, justify="left").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

    def _start_monitoring(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        self.stop_event.clear()
        self.last_outputs = None
        self._set_running_ui(True)
        self._append_event("正在启动摄像头和 YOLO 模型...")

        record_enabled = self.record_var.get()
        save_video_enabled = self.save_video_var.get()
        self.worker = threading.Thread(
            target=self._camera_worker,
            args=(record_enabled, save_video_enabled),
            daemon=True,
        )
        self.worker.start()

    def _stop_monitoring(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("正在停止")
            self.status_detail_var.set("正在释放摄像头，请稍候。")
            self.stop_button.state(["disabled"])

    def _camera_worker(self, record_enabled: bool, save_video_enabled: bool) -> None:
        import cv2

        from fall_prediction.video_app import (
            CSV_COLUMNS,
            create_pose_estimator,
            create_predictor,
            draw_overlay,
            draw_person_box,
            prediction_to_row,
        )

        capture = None
        estimator = None
        writer = None
        csv_file = None
        outputs = CameraOutputs(csv_path=None, video_path=None)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"camera_{session_id}_predictions.csv" if record_enabled else None
            video_path = self.output_dir / f"camera_{session_id}_annotated.mp4" if save_video_enabled else None
            outputs = CameraOutputs(csv_path=csv_path, video_path=video_path)

            self.events.put(("status", ("正在加载模型", "首次启动可能需要几秒钟。", GRAY)))
            estimator = create_pose_estimator(
                pose_backend="yolo",
                yolo_model_path=self.app_root / "models" / "yolo26n-pose.pt",
            )
            predictor = create_predictor(
                predictor_type="ml",
                classifier_model_path=self.app_root / "models" / "yolo_tail60_prefall_accel_classifier.joblib",
                use_hmm=True,
                use_accel=True,
                use_temporal_fall_validation=True,
            )

            from fall_prediction.camera import CameraOpenError, open_camera_capture, summarize_camera_attempts

            try:
                capture = open_camera_capture(CAMERA_INDEX)
            except CameraOpenError as exc:
                if exc.permission and not exc.permission.allowed:
                    raise RuntimeError(str(exc)) from exc
                raise RuntimeError(
                    "无法打开摄像头。请在“系统设置 > 隐私与安全性 > 摄像头”中允许 FallGuard"
                    "（如果通过 launch.command 启动，也需要允许 Terminal/Python），"
                    "并关闭 FaceTime、Zoom 等可能占用摄像头的软件。"
                    f"\n尝试过：{summarize_camera_attempts(exc.attempts)}。"
                ) from exc

            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 1e-6:
                fps = 20.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            if video_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            csv_writer = None
            if csv_path:
                csv_file = csv_path.open("w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                csv_writer.writeheader()

            self.events.put(("ready", outputs))
            frame_index = 0
            last_state = ""
            last_fps_time = time.monotonic()
            fps_frames = 0

            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("摄像头画面读取失败。")

                timestamp = frame_index / fps
                landmarks = estimator.process_bgr(frame, timestamp_ms=int(timestamp * 1000))
                prediction = predictor.predict(landmarks, frame_index, timestamp)
                bbox = draw_person_box(frame, landmarks)
                draw_overlay(frame, prediction, bbox)

                if csv_writer:
                    csv_writer.writerow(prediction_to_row(prediction))
                if writer:
                    writer.write(frame)

                display_state = prediction.alert_state or prediction.state
                if display_state != last_state:
                    last_state = display_state
                    self.events.put(("prediction", display_state))

                frame_data = self._frame_to_photo_data(frame)
                if frame_data:
                    self.events.put(("frame", frame_data))

                frame_index += 1
                fps_frames += 1
                now = time.monotonic()
                if now - last_fps_time >= 1.0:
                    live_fps = fps_frames / (now - last_fps_time)
                    self.events.put(("fps", live_fps))
                    fps_frames = 0
                    last_fps_time = now

        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if estimator:
                estimator.close()
            if capture:
                capture.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            self.events.put(("stopped", outputs))

    def _frame_to_photo_data(self, frame) -> str | None:
        import cv2

        height, width = frame.shape[:2]
        scale = min(DISPLAY_WIDTH / max(width, 1), DISPLAY_HEIGHT / max(height, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        ok, buffer = cv2.imencode(".png", frame)
        if not ok:
            return None
        return base64.b64encode(buffer).decode("ascii")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "frame":
                    self._show_frame(str(payload))
                elif event == "prediction":
                    self._update_prediction(str(payload))
                elif event == "fps":
                    self.fps_var.set(f"实时帧率 {float(payload):.1f} FPS")
                elif event == "status":
                    title, detail, color = payload
                    self._set_status(str(title), str(detail), str(color))
                elif event == "ready":
                    self.last_outputs = payload if isinstance(payload, CameraOutputs) else None
                    self.camera_var.set("摄像头已启动")
                    self._set_status("正在监测", "系统正在实时分析摄像头画面。", GREEN)
                    self._append_event("监测已开始。")
                    if self.last_outputs and self.last_outputs.csv_path:
                        self._append_event(f"记录文件：{self.last_outputs.csv_path.name}")
                elif event == "error":
                    self._handle_error(str(payload))
                elif event == "stopped":
                    if isinstance(payload, CameraOutputs):
                        self.last_outputs = payload
                    self._handle_stopped()
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _show_frame(self, data: str) -> None:
        try:
            image = tk.PhotoImage(data=data, format="png")
        except tk.TclError:
            return
        self.current_image = image
        self.video_label.configure(image=image, text="", width=1, height=1)

    def _update_prediction(self, state: str) -> None:
        title = STATE_TEXT.get(state, state)
        detail = STATE_DETAIL.get(state, "系统正在实时分析。")
        color = STATE_COLOR.get(state, GRAY)
        self._set_status(title, detail, color)
        if state in {"Pre-fall", "Fall", "Unknown"}:
            self._append_event(f"{title}：{detail}")

    def _set_status(self, title: str, detail: str, color: str) -> None:
        self.status_var.set(title)
        self.status_detail_var.set(detail)
        self.status_dot.itemconfig(self.status_dot_id, fill=color, outline=color)

    def _handle_error(self, message: str) -> None:
        self._append_event(f"错误：{message}")
        self._set_status("运行出错", message, RED)
        messagebox.showerror("跌倒监测", message)

    def _handle_stopped(self) -> None:
        self._set_running_ui(False)
        self.camera_var.set("摄像头未启动")
        self.fps_var.set("实时帧率 --")
        if self.status_var.get() not in {"运行出错"}:
            self._set_status("准备就绪", "点击“开始监测”即可打开摄像头。", GRAY)
        self._append_event("监测已停止。")

    def _set_running_ui(self, running: bool) -> None:
        if running:
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            self._set_status("正在启动", "正在打开摄像头和加载模型。", GRAY)
        else:
            self.start_button.state(["!disabled"])
            self.stop_button.state(["disabled"])

    def _append_event(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.configure(state="normal")
        self.event_log.insert("end", f"[{stamp}] {message}\n")
        self.event_log.see("end")
        self.event_log.configure(state="disabled")

    def _open_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(self.output_dir)], check=False)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "关于 FallGuard",
            "FallGuard\n\n普通用户版摄像头实时监测界面。\n默认使用 YOLO 姿态识别和机器学习跌倒预测模型。",
        )


def main() -> None:
    app = FallPredictionApp()
    app.mainloop()
