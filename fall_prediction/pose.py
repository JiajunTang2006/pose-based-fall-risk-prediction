

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


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"
DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo26n-pose.pt"


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


    def __init__(
        self,
        model_path: str | Path | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:


        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed in this Python environment. "
                "Create a Python 3.10 or 3.11 environment and run: "
                "python -m pip install -r requirements.txt"
            ) from exc


        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"

        if self._backend == "solutions":

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return


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


        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose = PoseLandmarker.create_from_options(options)

    def process_bgr(self, frame, timestamp_ms: int | None = None) -> list[Landmark] | None:

        import cv2


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":

            rgb.flags.writeable = False
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                return None
            points = result.pose_landmarks.landmark
        else:

            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect_for_video(mp_image, timestamp_ms or 0)
            if not result.pose_landmarks:
                return None
            points = result.pose_landmarks[0]


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

        self._pose.close()


class YOLOPoseEstimator:


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

        return None


def coco17_to_mediapipe_landmarks(
    xy,
    conf,
    image_width: int,
    image_height: int,
) -> list[Landmark]:

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

    if not landmarks:
        return

    import cv2

    pose_connections = POSE_CONNECTIONS if connections is None else connections
    height, width = frame.shape[:2]


    for first_idx, second_idx in pose_connections:
        first = landmarks[first_idx]
        second = landmarks[second_idx]

        if first.visibility < min_visibility or second.visibility < min_visibility:
            continue

        first_xy = (int(first.x * width), int(first.y * height))
        second_xy = (int(second.x * width), int(second.y * height))
        cv2.line(frame, first_xy, second_xy, line_color, 2)


    for point in landmarks:
        if point.visibility < min_visibility:
            continue
        xy = (int(point.x * width), int(point.y * height))

        cv2.circle(frame, xy, 3, (255, 255, 255), -1)
        cv2.circle(frame, xy, 4, point_ring_color, 1)
