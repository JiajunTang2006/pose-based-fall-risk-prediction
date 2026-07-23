"""
视频/摄像头入口程序：把整个跌倒预测系统跑起来。

这个文件的职责是"串联"所有模块：
1. 打开视频文件或摄像头
2. 逐帧读取图像
3. 用 MediaPipe 或 YOLO-pose 提取关键点 (pose.py)
4. 用 FallPredictor 分析跌倒风险 (predictor.py)
5. 在画面上画人物框和状态信息
6. 可选地输出 CSV 结果文件和标注视频

使用方式：
    # 使用默认摄像头（编号 0）
    python -m fall_prediction --show

    # 处理一个视频文件
    python -m fall_prediction --source my_video.mp4 --show

    # 输出 CSV 结果（需要显式指定）
    python -m fall_prediction --source my_video.mp4 --output-csv results.csv

    # 输出标注后的视频
    python -m fall_prediction --source my_video.mp4 --output-video annotated.mp4
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from .config import load_predictor_config
from .landmarks import LANDMARK_COUNT
from .pose import MediaPipePoseEstimator, YOLOPoseEstimator, draw_person_box
from .predictor import FallPredictor, PredictorConfig


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()


# 支持作为“图片序列视频”读取的图片格式。
# UR Fall 的 RGB 数据通常是 .png，这里额外兼容一些常见图片扩展名。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


    # CSV 文件的列名（显式导出结果时会包含这些字段）
CSV_COLUMNS = (
    "frame",                  # 帧编号
    "time",                   # 时间（秒）
    "state",                  # 最终状态（Normal/Pre-fall/Fall/Unknown）
    "alert_state",            # 报警状态；ML 模型可比 state 更早触发 Pre-fall
    "advisory_state",         # 双模型低级别辅助提示，不等同于正式报警
    "decision_tier",          # 双模型的分级判断层级
    "instant_state",          # 瞬时状态（未平滑）
    "risk_score",             # 瞬时风险分数
    "smoothed_risk_score",    # 平滑后的风险分数
    "has_pose",               # 是否检测到人体姿态（1/0）
    "torso_angle",            # 躯干倾斜角度（度）
    "torso_angular_velocity", # 躯干角速度（度/秒）
    "body_center_y",          # 身体中心 Y 坐标
    "body_center_delta",      # 身体中心相对上一帧的变化
    "vertical_velocity",      # 垂直速度
    "aspect_ratio",           # 宽高比
    "body_width",             # 人体包围盒宽度
    "body_height",            # 人体包围盒高度
    "visibility_mean",        # 平均可见度
    "center_drop",            # 身体中心下降量
    "system_status",          # 校准/运行状态说明
    "torso_signed_angle",     # 带方向躯干角，供固定站立校准
    "torso_valid",            # 躯干关键点是否完整可靠
    "center_valid",           # 身体中心特征是否可靠
    "bbox_valid",             # 可见人体框是否可靠
    "feature_coverage",       # 上述三组特征的覆盖率
    "shoulder_center_y",
    "shoulder_center_delta",
    "shoulder_vertical_velocity",
    "shoulder_line_angle",
    "shoulder_line_angular_velocity",
    "upper_body_width",
    "upper_body_height",
    "upper_body_aspect_ratio",
    "upper_body_valid",
    "upper_body_visibility_mean",
)

LANDMARK_CSV_COLUMNS = ("frame", "time") + tuple(
    f"kp{index:02d}_{field}"
    for index in range(LANDMARK_COUNT)
    for field in ("x", "y", "z", "visibility")
)


def process_video(
    source: str | int,
    output_csv: str | Path | None = None,
    output_video: str | Path | None = None,
    model_path: str | Path | None = None,
    pose_backend: str = "mediapipe",
    yolo_model_path: str | Path | None = None,
    show: bool = False,
    predictor_type: str = "rule",
    classifier_model_path: str | Path | None = None,
    fusion_model_path: str | Path | None = None,
    prefall_alert_threshold: float | None = None,
    prefall_alert_frames: int | None = None,
    use_hmm: bool = False,
    use_accel: bool | None = None,
    use_temporal_fall_validation: bool = True,
    temporal_sensitivity: str = "medium",
    automatic_fall_recovery: bool = False,
    fusion_fall_confirmation_steps: int = 3,
    image_sequence_fps: float = 30.0,
    predictor_config: PredictorConfig | None = None,
    output_landmarks_csv: str | Path | None = None,
    use_static_lying_adl_filter: bool = True,
) -> None:
    """
    处理视频或摄像头流，进行跌倒预测。

    这是整个程序的核心函数，串联了所有模块。

    参数:
        source:       视频文件路径，或者摄像头编号（0=默认摄像头）
        output_csv:   输出 CSV 文件路径；None 或空字符串表示不保存每帧 CSV
        output_video: 输出标注视频路径（在原视频上叠加骨架和状态）
        model_path:   MediaPipe Tasks 模型路径
        pose_backend: 姿态估计后端，"mediapipe" 或 "yolo"
        yolo_model_path: YOLO-pose .pt 模型路径
        show:         是否显示实时预览窗口
        predictor_type: "rule" 使用原规则系统，"ml" 使用训练好的机器学习模型
        classifier_model_path: 机器学习分类器路径；ensemble 模式下是树模型路径
        fusion_model_path: ensemble 模式使用的骨架+特征融合模型路径
        image_sequence_fps: 当 source 是图片目录时，假设这组图片的帧率是多少
    """
    import cv2

    # ---- 打开视频源 ----
    # source 现在支持三种形式：
    # 1. 摄像头编号：0
    # 2. 视频文件：xxx.mp4 / xxx.avi
    # 3. 图片目录：data/videos/fall-01-cam0-rgb
    capture = open_frame_source(source, image_sequence_fps=image_sequence_fps)
    writer = None
    csv_file = None
    csv_writer = None
    landmarks_file = None
    landmarks_writer = None
    estimator = None

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # 获取视频帧率（fps），如果获取不到就用默认值 30
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 30.0

        # 获取视频分辨率
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # ---- 设置输出视频写入器 ----
        if output_video:
            output_path = Path(output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 编码
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # ---- 设置 CSV 输出 ----
        if output_csv:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()  # 写入表头

        if output_landmarks_csv:
            landmarks_path = Path(output_landmarks_csv)
            landmarks_path.parent.mkdir(parents=True, exist_ok=True)
            landmarks_file = landmarks_path.open("w", newline="", encoding="utf-8")
            landmarks_writer = csv.DictWriter(landmarks_file, fieldnames=LANDMARK_CSV_COLUMNS)
            landmarks_writer.writeheader()

        # ---- 初始化核心模块 ----
        estimator = create_pose_estimator(
            pose_backend=pose_backend,
            model_path=model_path,
            yolo_model_path=yolo_model_path,
        )
        predictor = create_predictor(
            predictor_type,
            classifier_model_path,
            predictor_config,
            fusion_model_path=fusion_model_path,
            prefall_alert_threshold=prefall_alert_threshold,
            prefall_alert_frames=prefall_alert_frames,
            use_hmm=use_hmm,
            use_accel=use_accel,
            use_temporal_fall_validation=use_temporal_fall_validation,
            temporal_sensitivity=temporal_sensitivity,
            automatic_fall_recovery=automatic_fall_recovery,
            fusion_fall_confirmation_steps=fusion_fall_confirmation_steps,
            use_static_lying_adl_filter=use_static_lying_adl_filter,
        )  # 跌倒预测器

        frame_index = 0
        while True:
            # 读取一帧
            ok, frame = capture.read()
            if not ok:
                break  # 视频播放完毕

            # 计算当前时间戳（秒）
            timestamp = frame_index / fps

            # ---- 第一步：姿态估计后端提取关键点 ----
            landmarks = estimator.process_bgr(frame, timestamp_ms=int(timestamp * 1000))

            # ---- 第二步：跌倒预测 ----
            prediction = predictor.predict(landmarks, frame_index, timestamp)

            # ---- 第三步：绘制可视化 ----
            person_bbox = draw_person_box(frame, landmarks)
            draw_overlay(frame, prediction, person_bbox)  # 画状态文字（正常/预跌倒/跌倒）

            # ---- 第四步：输出数据 ----
            if csv_writer:
                csv_writer.writerow(prediction_to_row(prediction))
            if landmarks_writer:
                landmarks_writer.writerow(landmarks_to_row(landmarks, frame_index, timestamp))

            if writer:
                writer.write(frame)  # 写入标注后的视频帧

            # ---- 第五步：显示预览窗口 ----
            if show:
                cv2.imshow("Fall prediction", frame)
                key = cv2.waitKey(1) & 0xFF
                # 按 Q 键退出；按 R 键由操作员确认并解除已锁存 Fall。
                if key == ord("q"):
                    break
                if key == ord("r") and hasattr(predictor, "acknowledge_fall"):
                    predictor.acknowledge_fall()

            frame_index += 1
    finally:
        # ---- 清理资源 ----
        # finally 保证无论是否出错，都会释放资源
        if estimator:
            estimator.close()   # 释放姿态估计器资源
        capture.release()       # 释放摄像头/视频
        if writer:
            writer.release()    # 关闭视频文件
        if csv_file:
            csv_file.close()    # 关闭 CSV 文件
        if landmarks_file:
            landmarks_file.close()
        if show:
            cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口


def draw_overlay(frame, prediction, person_bbox: tuple[int, int, int, int] | None = None) -> None:
    """
    在视频帧上叠加状态信息的文字覆盖层。

    显示内容包括：
    - 当前报警/判断状态（Normal / Pre-fall / Fall / Unknown）
    - 如果报警状态和模型原始分类不同，额外显示模型原始分类

    颜色编码：
    - Normal   → 绿色（安全）
    - Pre-fall → 黄色（警告！即将跌倒）
    - Fall     → 红色（跌倒！）
    - Unknown  → 灰色（未检测到人）
    """
    import cv2

    display_state = prediction.alert_state or prediction.state
    advisory_state = getattr(prediction, "advisory_state", None)

    # 根据报警状态选择文字颜色
    color = {
        "Normal": (80, 220, 120),    # 绿色
        "Pre-fall": (0, 200, 255),   # 黄色（OpenCV 是 BGR，所以 (0,200,255)=黄色）
        "Fall": (0, 80, 255),        # 红色
        "Unknown": (160, 160, 160),  # 灰色
    }.get(display_state, (255, 255, 255))
    if advisory_state and display_state == "Normal":
        color = (0, 200, 255) if advisory_state == "Pre-fall" else (0, 120, 255)

    # 只保留最关键的信息，避免画面被参数遮住。
    lines = [f"State: {display_state}"]
    if advisory_state and advisory_state != display_state:
        lines.append(f"Advisory: {advisory_state}")
    if prediction.system_status:
        lines.append(prediction.system_status)
    if display_state != prediction.state:
        lines.insert(1, f"Model: {prediction.state}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.82
    thickness = 3
    padding = 10
    line_height = 32
    frame_height, frame_width = frame.shape[:2]
    text_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in lines)
    block_width = text_width + padding * 2
    block_height = line_height * len(lines) + padding

    # 文字跟随人物框，但放在框外：优先在框上方；空间不够时放在框下方。
    if person_bbox is None:
        x, top = 18, 18
    else:
        x1, y1, _x2, y2 = person_bbox
        x = max(8, min(x1, frame_width - block_width - 8))
        if y1 - block_height - 8 >= 8:
            top = y1 - block_height - 8
        elif y2 + block_height + 8 <= frame_height - 8:
            top = y2 + 8
        else:
            top = max(8, min(y1, frame_height - block_height - 8))

    bottom = top + block_height
    cv2.rectangle(frame, (x, top), (x + block_width, bottom), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, top), (x + block_width, bottom), color, 1)

    for index, text in enumerate(lines):
        y_pos = top + padding + 24 + index * line_height
        cv2.putText(frame, text, (x + padding, y_pos), font, font_scale, color, thickness)


def prediction_to_row(prediction) -> dict[str, str | int]:
    """
    将 Prediction 对象转换为 CSV 的一行数据。

    把所有数值格式化为 4 位小数，方便后续在 Excel 或 Python 中分析。
    """
    features = prediction.features
    return {
        "frame": prediction.frame_index,
        "time": f"{prediction.timestamp:.4f}",
        "state": prediction.state,
        "alert_state": prediction.alert_state or prediction.state,
        "advisory_state": getattr(prediction, "advisory_state", None) or "",
        "decision_tier": getattr(prediction, "decision_tier", None) or "",
        "instant_state": prediction.instant_state,
        "risk_score": f"{prediction.risk_score:.4f}",
        "smoothed_risk_score": f"{prediction.smoothed_risk_score:.4f}",
        "has_pose": int(features.has_pose),
        "torso_angle": f"{features.torso_angle_deg:.4f}",
        "torso_angular_velocity": f"{features.torso_angular_velocity:.4f}",
        "body_center_y": f"{features.body_center_y:.4f}",
        "body_center_delta": f"{features.body_center_delta:.4f}",
        "vertical_velocity": f"{features.vertical_velocity:.4f}",
        "aspect_ratio": f"{features.aspect_ratio:.4f}",
        "body_width": f"{features.body_width:.4f}",
        "body_height": f"{features.body_height:.4f}",
        "visibility_mean": f"{features.visibility_mean:.4f}",
        "center_drop": f"{prediction.breakdown.center_drop:.4f}",
        "system_status": prediction.system_status or "",
        "torso_signed_angle": f"{features.torso_signed_angle_deg:.4f}",
        "torso_valid": int(features.torso_valid),
        "center_valid": int(features.center_valid),
        "bbox_valid": int(features.bbox_valid),
        "feature_coverage": f"{(float(features.torso_valid) + float(features.center_valid) + float(features.bbox_valid)) / 3.0:.4f}",
        "shoulder_center_y": f"{features.shoulder_center_y:.4f}",
        "shoulder_center_delta": f"{features.shoulder_center_delta:.4f}",
        "shoulder_vertical_velocity": f"{features.shoulder_vertical_velocity:.4f}",
        "shoulder_line_angle": f"{features.shoulder_line_angle_deg:.4f}",
        "shoulder_line_angular_velocity": f"{features.shoulder_line_angular_velocity:.4f}",
        "upper_body_width": f"{features.upper_body_width:.4f}",
        "upper_body_height": f"{features.upper_body_height:.4f}",
        "upper_body_aspect_ratio": f"{features.upper_body_aspect_ratio:.4f}",
        "upper_body_valid": int(features.upper_body_valid),
        "upper_body_visibility_mean": f"{features.upper_body_visibility_mean:.4f}",
    }


def landmarks_to_row(landmarks, frame_index: int, timestamp: float) -> dict[str, str | int]:
    """Serialize all raw keypoints and confidences for future mask-aware training."""
    row: dict[str, str | int] = {"frame": frame_index, "time": f"{timestamp:.4f}"}
    for index in range(LANDMARK_COUNT):
        point = landmarks[index] if landmarks is not None and index < len(landmarks) else None
        row[f"kp{index:02d}_x"] = f"{point.x:.6f}" if point is not None else "0.000000"
        row[f"kp{index:02d}_y"] = f"{point.y:.6f}" if point is not None else "0.000000"
        row[f"kp{index:02d}_z"] = f"{point.z:.6f}" if point is not None else "0.000000"
        row[f"kp{index:02d}_visibility"] = (
            f"{point.visibility:.6f}" if point is not None else "0.000000"
        )
    return row


def create_pose_estimator(
    pose_backend: str,
    model_path: str | Path | None = None,
    yolo_model_path: str | Path | None = None,
):
    """Create the requested pose estimation backend."""
    if pose_backend == "mediapipe":
        return MediaPipePoseEstimator(model_path=model_path)
    if pose_backend == "yolo":
        return YOLOPoseEstimator(model_path=yolo_model_path)
    raise ValueError(f"Unknown pose backend: {pose_backend}")


def create_predictor(
    predictor_type: str,
    classifier_model_path: str | Path | None,
    predictor_config: PredictorConfig | None = None,
    fusion_model_path: str | Path | None = None,
    prefall_alert_threshold: float | None = None,
    prefall_alert_frames: int | None = None,
    use_hmm: bool = False,
    use_accel: bool | None = None,
    use_temporal_fall_validation: bool = True,
    temporal_sensitivity: str = "medium",
    automatic_fall_recovery: bool = False,
    fusion_fall_confirmation_steps: int = 3,
    use_static_lying_adl_filter: bool = True,
):
    """Create the requested prediction backend."""
    if predictor_type == "rule":
        return FallPredictor(config=predictor_config)
    if predictor_type == "ensemble":
        from .ensemble_predictor import (
            DEFAULT_FUSION_MODEL_PATH,
            DEFAULT_TREE_MODEL_PATH,
            DualModelFallPredictor,
        )

        return DualModelFallPredictor(
            tree_model_path=classifier_model_path or DEFAULT_TREE_MODEL_PATH,
            fusion_model_path=fusion_model_path or DEFAULT_FUSION_MODEL_PATH,
            predictor_config=predictor_config,
            prefall_alert_threshold=prefall_alert_threshold,
            prefall_alert_consecutive_frames=prefall_alert_frames,
            fusion_use_hmm=use_hmm,
            use_accel=use_accel,
            fusion_fall_confirmation_steps=fusion_fall_confirmation_steps,
            use_static_lying_adl_filter=use_static_lying_adl_filter,
        )
    if predictor_type in {"ml", "deep", "fusion"}:
        if classifier_model_path is None:
            classifier_model_path = (
                "models/skeleton_feature_fusion_tuned.pt"
                if predictor_type == "fusion"
                else
                "models/tcn_prefall_classifier.pt"
                if predictor_type == "deep"
                else "models/yolo_tail60_prefall_accel_robust_classifier.joblib"
            )
        from .ml_predictor import MachineLearningFallPredictor

        return MachineLearningFallPredictor(
            classifier_model_path,
            baseline_frames=predictor_config.baseline_frames if predictor_config else None,
            smoothing_window=predictor_config.smoothing_window if predictor_config else None,
            min_visibility=(
                predictor_config.risk.min_visibility
                if predictor_config
                else DEFAULT_PREDICTOR_CONFIG.risk.min_visibility
            ),
            prefall_alert_threshold=prefall_alert_threshold,
            prefall_alert_consecutive_frames=prefall_alert_frames,
            use_hmm=use_hmm,
            use_accel=use_accel,
            use_temporal_fall_validation=use_temporal_fall_validation,
            temporal_sensitivity=temporal_sensitivity,
            automatic_fall_recovery=automatic_fall_recovery,
        )
    raise ValueError(f"Unknown predictor type: {predictor_type}")


class ImageSequenceCapture:
    """
    把一个图片文件夹包装成类似 cv2.VideoCapture 的对象。

    OpenCV 的 VideoCapture 可以读取 mp4 视频，但不能直接把一个目录当视频读。
    UR Fall 的图片版数据集是一帧一张图，所以这里做一个小适配器：

    - isOpened(): 判断目录里有没有可读图片
    - read(): 每次返回下一张图片，就像读取视频下一帧
    - get(): 提供 fps、宽、高等信息，方便后面的代码不用区分视频/图片目录
    - release(): 保持和 VideoCapture 一样的接口

    这样 process_video() 里后续的姿态估计、预测、写 CSV、写标注视频逻辑
    都可以复用，不需要为图片数据集再写一套流程。
    """

    def __init__(self, image_dir: str | Path, fps: float = 30.0) -> None:
        self.image_dir = Path(image_dir)
        self.image_paths = find_image_sequence_files(self.image_dir)
        self.fps = fps if fps > 0 else infer_image_sequence_fps(self.image_paths)
        self.index = 0
        self.width = 0
        self.height = 0

        # 读取第一张图片，确定分辨率。后面写 MP4 标注视频时要求所有帧同尺寸。
        if self.image_paths:
            import cv2

            first_frame = cv2.imread(str(self.image_paths[0]))
            if first_frame is not None:
                self.height, self.width = first_frame.shape[:2]

    def isOpened(self) -> bool:
        """目录中有图片，并且第一张图能被 OpenCV 正常读取，就认为打开成功。"""
        return bool(self.image_paths) and self.width > 0 and self.height > 0

    def read(self):
        """
        读取下一张图片。

        返回格式和 cv2.VideoCapture.read() 一样：
            (True, frame)  表示成功读取一帧
            (False, None)  表示序列结束
        """
        import cv2

        while self.index < len(self.image_paths):
            image_path = self.image_paths[self.index]
            self.index += 1
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: could not read image, skipped: {image_path}")
                continue

            # 如果某些图片尺寸不一致，就缩放到第一张图的尺寸。
            # 这能避免 VideoWriter 因为帧尺寸变化而写出损坏视频。
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            return True, frame

        return False, None

    def get(self, prop_id: int) -> float:
        """模拟 cv2.VideoCapture.get()，返回帧率、宽、高、帧数等信息。"""
        import cv2

        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.image_paths))
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.index)
        return 0.0

    def release(self) -> None:
        """图片序列没有需要释放的底层句柄，这里只是为了接口兼容。"""
        return None


def open_frame_source(source: str | int, image_sequence_fps: float = 30.0):
    """
    根据 source 类型打开帧源。

    返回的对象一定提供 read/isOpened/get/release 这些方法，
    所以后面的主循环可以统一处理。
    """
    import cv2

    if isinstance(source, int):
        return cv2.VideoCapture(source)

    source_path = Path(source)
    if source_path.is_dir():
        return ImageSequenceCapture(source_path, fps=image_sequence_fps)

    return cv2.VideoCapture(str(source_path))


def find_image_sequence_files(image_dir: str | Path) -> list[Path]:
    """
    找到目录中的图片，并按“自然顺序”排序。

    普通字符串排序会出现这种问题：
        1.png, 10.png, 2.png

    自然排序会按数字大小排：
        1.png, 2.png, 10.png

    UR Fall 文件名类似 fall-01-cam0-rgb-001.png，
    所以自然排序可以确保帧顺序正确。
    """
    directory = Path(image_dir)
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_sort_key)


def infer_image_sequence_fps(image_paths: list[Path], default_fps: float = 30.0) -> float:
    """
    Infer FPS from timestamp-style image names.

    UP Fall RGB frames use names such as:
        2018-07-04T12_04_26.648452.png

    UR Fall frame names do not contain wall-clock timestamps, so those fall back
    to default_fps.
    """
    timestamps = [_timestamp_from_image_name(path) for path in image_paths]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if len(timestamps) < 2:
        return default_fps

    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    if duration <= 0:
        return default_fps
    return (len(timestamps) - 1) / duration


def _timestamp_from_image_name(path: Path) -> datetime | None:
    """Parse UP Fall timestamp image names, returning None for other naming styles."""
    try:
        return datetime.strptime(path.stem, "%Y-%m-%dT%H_%M_%S.%f")
    except ValueError:
        return None


def natural_sort_key(path: str | Path) -> list[int | str]:
    """把文件名拆成文字和数字片段，用于自然排序。"""
    name = Path(path).name.lower()
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def parse_source(value: str) -> str | int:
    """
    解析 --source 参数：如果是纯数字就当作摄像头编号（int），否则当作文件路径（str）。

    示例:
        "0" → 0（摄像头 0）
        "my_video.mp4" → "my_video.mp4"（文件路径）
    """
    if value.isdigit():
        return int(value)
    return value


def main() -> None:
    """
    命令行入口函数。

    支持的参数：
        --source       视频文件路径或摄像头编号（默认 0=默认摄像头）
        --output-csv   CSV 输出路径（默认不导出）
        --output-video 标注视频输出路径（默认不输出）
        --model        MediaPipe 模型路径（默认自动检测）
        --pose-backend 姿态估计后端：mediapipe 或 yolo
        --yolo-model   YOLO-pose 模型路径
        --image-fps    source 是图片目录时使用的帧率
        --disable-temporal-fall-validation 关闭 Fall 时序确认层
        --show         显示实时预览窗口
    """
    parser = argparse.ArgumentParser(description="Run fall prediction on a video, image sequence, or webcam.")
    parser.add_argument("--source", default="0", help="视频路径、图片目录或摄像头编号。默认使用摄像头 0。")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path. Omit this option to disable CSV export.",
    )
    parser.add_argument(
        "--output-landmarks-csv",
        default=None,
        help="可选：额外保存逐帧完整关键点坐标与置信度，供遮挡增强训练。",
    )
    parser.add_argument("--output-video", default=None, help="Optional annotated MP4 output path.")
    parser.add_argument("--model", default=None, help="Optional MediaPipe Tasks pose landmarker .task model path.")
    parser.add_argument(
        "--pose-backend",
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help="姿态估计后端：mediapipe 使用原流程，yolo 使用 Ultralytics YOLO-pose。",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo26n-pose.pt",
        help="当 --pose-backend yolo 时加载的 YOLO-pose .pt 模型路径。",
    )
    parser.add_argument("--config", default=None, help="可选：JSON 配置文件，例如 configs/default.json。")
    parser.add_argument("--image-fps", type=float, default=30.0, help="当 --source 是图片目录时使用的帧率，默认 30。")
    parser.add_argument(
        "--predictor",
        choices=("rule", "ml", "deep", "fusion", "ensemble"),
        default="rule",
        help=(
            "预测后端：rule规则，ml树模型，deep因果TCN，fusion骨架图网络+TCN，"
            "ensemble树模型确认+融合模型提前预警。"
        ),
    )
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="ml/deep 分类器路径；ensemble 模式下是树模型路径。",
    )
    parser.add_argument(
        "--fusion-model",
        default=None,
        help="ensemble 模式的骨架+特征融合模型路径；省略时使用默认模型。",
    )
    parser.add_argument(
        "--prefall-alert-threshold",
        type=float,
        default=None,
        help="ML 预警阈值：即使 state 仍为 Normal，Pre-fall 概率连续偏高也显示 Alert: Pre-fall。",
    )
    parser.add_argument(
        "--prefall-alert-frames",
        type=int,
        default=None,
        help="ML 预警需要连续多少帧超过 --prefall-alert-threshold，默认 1。",
    )
    parser.add_argument(
        "--use-hmm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用 HMM；deep/fusion/ensemble 默认启用，rule/ml 默认不启用。",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="推理时使用加速度增强特征（需模型训练时也启用了 --use-accel）。",
    )
    parser.add_argument(
        "--disable-temporal-fall-validation",
        action="store_true",
        help="关闭运行时 Fall 时序确认层。deep/fusion/ensemble 默认已经关闭。",
    )
    parser.add_argument(
        "--enable-temporal-fall-validation",
        action="store_true",
        help="显式开启运行时 Fall 时序确认层；deep/fusion/ensemble 默认关闭以避免重复平滑。",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("high", "medium", "low"),
        default="medium",
        help="ML 时序门控敏感度：high 较早提醒，medium 平衡误报，low 最保守。",
    )
    parser.add_argument(
        "--auto-fall-recovery",
        action="store_true",
        help="允许在持续可靠直立 Normal 后自动解除 Fall；默认必须按 R 或调用 acknowledge_fall() 人工确认。",
    )
    parser.add_argument(
        "--fusion-fall-confirmation-steps",
        type=int,
        default=3,
        help="ensemble 中仅融合模型判断 Fall 时，需要连续确认的模型输出次数，默认 3。",
    )
    parser.add_argument(
        "--static-lying-adl-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "ensemble 中启用静态躺姿 ADL 后处理（默认启用）；"
            "可用 --no-static-lying-adl-filter 关闭以比较模型原始输出。"
        ),
    )
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview window.")
    args = parser.parse_args()
    if args.disable_temporal_fall_validation and args.enable_temporal_fall_validation:
        parser.error("Cannot enable and disable temporal Fall validation at the same time")
    predictor_config = load_predictor_config(args.config) if args.config else None
    resolved_use_hmm = (
        args.predictor in {"deep", "fusion", "ensemble"}
        if args.use_hmm is None
        else args.use_hmm
    )
    if args.enable_temporal_fall_validation:
        resolved_temporal_validation = True
    elif args.disable_temporal_fall_validation:
        resolved_temporal_validation = False
    else:
        resolved_temporal_validation = args.predictor not in {"deep", "fusion", "ensemble"}

    process_video(
        source=parse_source(args.source),
        output_csv=args.output_csv,
        output_video=args.output_video,
        model_path=args.model,
        pose_backend=args.pose_backend,
        yolo_model_path=args.yolo_model,
        show=args.show,
        predictor_type=args.predictor,
        classifier_model_path=args.classifier_model,
        fusion_model_path=args.fusion_model,
        prefall_alert_threshold=args.prefall_alert_threshold,
        prefall_alert_frames=args.prefall_alert_frames,
        use_hmm=resolved_use_hmm,
        use_accel=args.use_accel if args.use_accel else None,
        use_temporal_fall_validation=resolved_temporal_validation,
        temporal_sensitivity=args.sensitivity,
        automatic_fall_recovery=args.auto_fall_recovery,
        fusion_fall_confirmation_steps=args.fusion_fall_confirmation_steps,
        use_static_lying_adl_filter=args.static_lying_adl_filter,
        image_sequence_fps=args.image_fps,
        predictor_config=predictor_config,
        output_landmarks_csv=args.output_landmarks_csv,
    )


if __name__ == "__main__":
    main()
