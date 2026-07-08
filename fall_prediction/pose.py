"""
姿势估计适配器 + OpenCV 可视化绘制工具。

这个文件主要做两件事：
1. 封装 MediaPipe/YOLO-pose，把 BGR 图像输入进去，拿到项目内部统一的 33 个关键点
2. 在视频帧上画出骨架线和关键点，方便直观查看

MediaPipe 有两种 API 模式：
- Legacy Solutions API（旧版，调用 mp.solutions.pose）
- Tasks API（新版，需要下载 .task 模型文件）
本代码会自动检测你安装的是哪个版本，并使用对应的 API。

YOLO-pose 默认输出 COCO 17 个关键点；这里会映射到项目内部使用的
MediaPipe 33 点编号。YOLO 没有的点会填成 visibility=0，后续特征提取会自动忽略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .landmarks import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    POSE_CONNECTIONS,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
)


# 默认模型文件路径：项目根目录下的 models/pose_landmarker_full.task
# 仅在使用新版 MediaPipe Tasks API 时需要
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"
DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo26n-pose.pt"


# YOLO-pose 默认 COCO 17 点编号 -> 项目内部 MediaPipe 33 点编号。
# COCO 的点集合缺少手指、脚跟、脚尖等细节；这些缺失点会保持 visibility=0。
COCO17_TO_MEDIAPIPE = {
    0: NOSE,
    1: LEFT_EYE,
    2: RIGHT_EYE,
    3: LEFT_EAR,
    4: RIGHT_EAR,
    5: LEFT_SHOULDER,
    6: RIGHT_SHOULDER,
    7: LEFT_ELBOW,
    8: RIGHT_ELBOW,
    9: LEFT_WRIST,
    10: RIGHT_WRIST,
    11: LEFT_HIP,
    12: RIGHT_HIP,
    13: LEFT_KNEE,
    14: RIGHT_KNEE,
    15: LEFT_ANKLE,
    16: RIGHT_ANKLE,
}


class MediaPipePoseEstimator:
    """
    MediaPipe 姿势估计器。

    功能：输入一张 BGR 格式的图像帧，输出人体 33 个关键点的坐标和可见度。
    如果图像中没有检测到人，返回 None。

    使用方式：
        estimator = MediaPipePoseEstimator()
        landmarks = estimator.process_bgr(frame)  # frame 是 OpenCV 读取的 BGR 图像
        # landmarks 是一个包含 33 个 Landmark 的列表，或者 None
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        初始化姿势估计器。

        参数:
            model_path:              Tasks API 的模型文件路径（仅新版需要）
            model_complexity:        模型复杂度（0/1/2），越高越准但越慢（仅旧版）
            min_detection_confidence: 检测到人的最低置信度（低于此值认为没有人）
            min_tracking_confidence:  追踪关键点的最低置信度
        """
        # 尝试导入 mediapipe，如果没安装就给出友好的错误提示
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed in this Python environment. "
                "Create a Python 3.10 or 3.11 environment and run: "
                "python -m pip install -r requirements.txt"
            ) from exc

        # 自动检测是旧版 API（solutions）还是新版 API（tasks）
        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"

        if self._backend == "solutions":
            # --- 旧版 Legacy Solutions API ---
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,          # 视频模式（非静态图片）
                model_complexity=model_complexity, # 模型复杂度
                smooth_landmarks=True,             # 开启关键点平滑
                enable_segmentation=False,         # 不需要人像分割
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        # --- 新版 Tasks API ---
        # 确定模型文件路径
        model = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model.exists():
            raise RuntimeError(
                "This MediaPipe version uses the newer Tasks API, which requires "
                f"a pose landmarker model file. Expected model path: {model}\n"
                "Download the full model from:\n"
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task\n"
                "Then save it as models/pose_landmarker_full.task."
            )

        # 配置并创建 PoseLandmarker
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=VisionRunningMode.VIDEO,  # 视频模式（逐帧处理）
            num_poses=1,                            # 最多检测 1 个人
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,        # 不需要人像分割
        )
        self._pose = PoseLandmarker.create_from_options(options)

    def process_bgr(self, frame, timestamp_ms: int | None = None) -> list[Landmark] | None:
        """
        处理一帧 BGR 图像，提取人体关键点。

        参数:
            frame:        OpenCV 格式的 BGR 图像（numpy 数组，shape: [高, 宽, 3]）
            timestamp_ms: 当前帧的时间戳（毫秒），仅 Tasks API 需要

        返回:
            list[Landmark] | None: 33 个关键点的列表；如果没检测到人则返回 None
        """
        import cv2

        # MediaPipe 需要 RGB 格式，OpenCV 默认是 BGR，所以先转换
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            # --- 旧版 API ---
            rgb.flags.writeable = False  # 提高性能的小技巧
            result = self._pose.process(rgb)
            if not result.pose_landmarks:  # 没检测到人
                return None
            points = result.pose_landmarks.landmark
        else:
            # --- 新版 Tasks API ---
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect_for_video(mp_image, timestamp_ms or 0)
            if not result.pose_landmarks:  # 没检测到人
                return None
            points = result.pose_landmarks[0]  # 只取第一个人的关键点

        # 将 MediaPipe 的结果转换为我们自己定义的 Landmark 数据结构
        return [
            Landmark(
                x=point.x,
                y=point.y,
                z=point.z,
                visibility=point.visibility,
            )
            for point in points
        ]

    def close(self) -> None:
        """释放 MediaPipe 资源（用完后一定要调用，否则会内存泄漏）。"""
        self._pose.close()


class YOLOPoseEstimator:
    """
    YOLO-pose 姿势估计器。

    功能：输入一张 BGR 图像，使用 Ultralytics YOLO-pose 输出 COCO 17 个关键点，
    再映射成项目内部统一的 33 个 Landmark。这样 features.py、risk.py 和 ML
    分类器不用关心底层姿势模型来自 MediaPipe 还是 YOLO。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics YOLO is not installed in this Python environment. "
                "Install it with: python -m pip install -U ultralytics"
            ) from exc

        model = Path(model_path) if model_path else DEFAULT_YOLO_MODEL_PATH
        if not model.exists():
            raise RuntimeError(
                f"YOLO pose model not found: {model}\n"
                "Download a YOLO-pose model first, for example:\n"
                "  yolo pose predict model=yolo26n-pose.pt source=some_image.png\n"
                "Then move it to models/yolo26n-pose.pt, or pass --yolo-model."
            )

        self.model_path = model
        self.min_detection_confidence = min_detection_confidence
        self._model = YOLO(str(model))

    def process_bgr(self, frame, timestamp_ms: int | None = None) -> list[Landmark] | None:
        """
        处理一帧 BGR 图像，提取人体关键点。

        YOLO 接受 numpy 图像输入。这里关闭 verbose，避免批量导出时刷屏。
        如果检测到多个人，选择检测框置信度最高的人。
        """
        results = self._model(frame, verbose=False)
        if not results:
            return None

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        if len(result.keypoints.xy) == 0:
            return None

        person_index = self._best_person_index(result)
        if person_index is None:
            return None

        height, width = frame.shape[:2]
        xy = result.keypoints.xy[person_index].detach().cpu().numpy()
        conf = None
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[person_index].detach().cpu().numpy()

        return coco17_to_mediapipe_landmarks(
            xy=xy,
            conf=conf,
            image_width=width,
            image_height=height,
        )

    def _best_person_index(self, result) -> int | None:
        if result.boxes is None or result.boxes.conf is None or len(result.boxes.conf) == 0:
            return 0

        confidences = result.boxes.conf.detach().cpu().numpy()
        best_index = int(confidences.argmax())
        if confidences[best_index] < self.min_detection_confidence:
            return None
        return best_index

    def close(self) -> None:
        """YOLO 模型没有显式 close API；保留方法用于和 MediaPipe 接口兼容。"""
        return None


def coco17_to_mediapipe_landmarks(
    xy,
    conf,
    image_width: int,
    image_height: int,
) -> list[Landmark]:
    """
    把 YOLO COCO 17 点转换成项目内部 MediaPipe 33 点格式。

    参数:
        xy:           shape=(17, 2)，像素坐标。
        conf:         shape=(17,)，每个点的置信度；如果为 None，就默认 1.0。
        image_width:  图像宽度，用于归一化 x。
        image_height: 图像高度，用于归一化 y。
    """
    landmarks = [Landmark(0.0, 0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)

    for coco_index, mediapipe_index in COCO17_TO_MEDIAPIPE.items():
        if coco_index >= len(xy):
            continue
        x, y = xy[coco_index]
        visibility = float(conf[coco_index]) if conf is not None and coco_index < len(conf) else 1.0
        landmarks[mediapipe_index] = Landmark(
            x=float(x) / width,
            y=float(y) / height,
            z=0.0,
            visibility=max(0.0, min(1.0, visibility)),
        )

    return landmarks


def visible_landmark_bbox(
    landmarks: Sequence[Landmark] | None,
    image_width: int,
    image_height: int,
    min_visibility: float = 0.2,
    padding_ratio: float = 0.08,
) -> tuple[int, int, int, int] | None:
    """
    根据可见关键点估算人体框。

    返回:
        (x1, y1, x2, y2)，单位是像素；如果没有可靠关键点则返回 None。
    """
    if not landmarks:
        return None

    visible_points = [
        point
        for point in landmarks
        if point.visibility >= min_visibility and 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0
    ]
    if not visible_points:
        return None

    min_x = min(point.x for point in visible_points)
    max_x = max(point.x for point in visible_points)
    min_y = min(point.y for point in visible_points)
    max_y = max(point.y for point in visible_points)

    width = max_x - min_x
    height = max_y - min_y
    padding_x = max(width * padding_ratio, 0.02)
    padding_y = max(height * padding_ratio, 0.02)

    x1 = int(max(0.0, min_x - padding_x) * image_width)
    y1 = int(max(0.0, min_y - padding_y) * image_height)
    x2 = int(min(1.0, max_x + padding_x) * image_width)
    y2 = int(min(1.0, max_y + padding_y) * image_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_person_box(
    frame,
    landmarks: Sequence[Landmark] | None,
    label: str | None = None,
    min_visibility: float = 0.2,
    color: tuple[int, int, int] = (0, 220, 255),
) -> tuple[int, int, int, int] | None:
    """在视频帧上用可见关键点估算并绘制人体框，返回框坐标。"""
    import cv2

    height, width = frame.shape[:2]
    bbox = visible_landmark_bbox(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        min_visibility=min_visibility,
    )
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        text_y = max(20, y1 - 8)
        cv2.putText(frame, label, (x1 + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return bbox


def draw_pose(
    frame,
    landmarks: Sequence[Landmark] | None,
    min_visibility: float = 0.2,
    connections: Sequence[tuple[int, int]] | None = None,
    line_color: tuple[int, int, int] = (80, 220, 120),
    point_ring_color: tuple[int, int, int] = (40, 120, 255),
) -> None:
    """
    在视频帧上画出人体骨架线和关键点。

    这个函数会直接修改传入的 frame（原地绘制），不需要返回值。

    参数:
        frame:          OpenCV 图像（会被直接修改）
        landmarks:      关键点列表，为 None 或空时不画任何东西
        min_visibility: 可见度低于此值的点会被跳过（不画）
        connections:    骨架连接关系；默认使用 MediaPipe 33 点连接
        line_color:     骨架线颜色，OpenCV BGR
        point_ring_color: 关键点外圈颜色，OpenCV BGR
    """
    if not landmarks:
        return

    import cv2

    pose_connections = POSE_CONNECTIONS if connections is None else connections
    height, width = frame.shape[:2]  # 获取图像尺寸

    # --- 第一步：画出骨骼连线 ---
    for first_idx, second_idx in pose_connections:
        first = landmarks[first_idx]
        second = landmarks[second_idx]
        # 如果连接的两端有一个不可见，就跳过这条线
        if first.visibility < min_visibility or second.visibility < min_visibility:
            continue
        # 将归一化坐标（0~1）转换为图像上的像素坐标
        first_xy = (int(first.x * width), int(first.y * height))
        second_xy = (int(second.x * width), int(second.y * height))
        cv2.line(frame, first_xy, second_xy, line_color, 2)

    # --- 第二步：画出每个关键点 ---
    for point in landmarks:
        if point.visibility < min_visibility:
            continue
        xy = (int(point.x * width), int(point.y * height))
        # 白色实心圆（半径 3）+ 红色空心圆（半径 4）= 白心红边的小圆点
        cv2.circle(frame, xy, 3, (255, 255, 255), -1)  # -1 表示填充
        cv2.circle(frame, xy, 4, point_ring_color, 1)  # 1 表示线宽
