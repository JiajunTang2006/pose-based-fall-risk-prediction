from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from .camera import open_camera_capture
from .config import load_predictor_config
from .lowlight import enhance_low_light
from .pose import MediaPipePoseEstimator, YOLOPoseEstimator, draw_person_box
from .predictor import FallPredictor, PredictorConfig
from .sensitivity import SENSITIVITY_LEVELS, ml_config_for_sensitivity, predictor_config_for_sensitivity


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


CSV_COLUMNS = (
    "frame",
    "time",
    "state",
    "alert_state",
    "advisory_state",
    "decision_tier",
    "instant_state",
    "risk_score",
    "smoothed_risk_score",
    "has_pose",
    "torso_angle",
    "torso_angular_velocity",
    "body_center_y",
    "body_center_delta",
    "vertical_velocity",
    "aspect_ratio",
    "body_width",
    "body_height",
    "visibility_mean",
    "center_drop",
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
    fall_validator_settings: Mapping[str, float | int] | None = None,
    temporal_sensitivity: str = "high",
    fusion_fall_confirmation_steps: int = 3,
    use_static_lying_adl_filter: bool = True,
    image_sequence_fps: float = 30.0,
    predictor_config: PredictorConfig | None = None,
    on_prediction: Callable[[object, int, float, object], None] | None = None,
    enhance_low_light_frames: bool = True,
) -> None:
    import cv2

    capture = open_frame_source(source, image_sequence_fps=image_sequence_fps)
    writer = None
    csv_file = None
    csv_writer = None
    estimator = None

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        if output_video:
            output_path = Path(output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if output_csv:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()

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
            fall_validator_settings=fall_validator_settings,
            temporal_sensitivity=temporal_sensitivity,
            fusion_fall_confirmation_steps=fusion_fall_confirmation_steps,
            use_static_lying_adl_filter=use_static_lying_adl_filter,
        )

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps

            if enhance_low_light_frames:
                frame = enhance_low_light(frame)

            landmarks = estimator.process_bgr(frame, timestamp_ms=int(timestamp * 1000))

            prediction = predictor.predict(landmarks, frame_index, timestamp)
            if on_prediction is not None:
                on_prediction(prediction, frame_index, timestamp, frame)

            person_bbox = draw_person_box(frame, landmarks)
            draw_overlay(frame, prediction, person_bbox)

            if csv_writer:
                csv_writer.writerow(prediction_to_row(prediction))

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Fall prediction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        if estimator:
            estimator.close()
        capture.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        if show:
            cv2.destroyAllWindows()


def draw_overlay(frame, prediction, person_bbox: tuple[int, int, int, int] | None = None) -> None:
    import cv2

    display_state = prediction.alert_state or prediction.state
    advisory_state = getattr(prediction, "advisory_state", None)

    color = {
        "Normal": (80, 220, 120),
        "Pre-fall": (0, 200, 255),
        "Fall": (0, 80, 255),
        "Unknown": (160, 160, 160),
    }.get(display_state, (255, 255, 255))
    if advisory_state and display_state == "Normal":
        color = (0, 200, 255) if advisory_state == "Pre-fall" else (0, 120, 255)

    lines = [f"State: {display_state}"]
    if advisory_state and advisory_state != display_state:
        lines.append(f"Advisory: {advisory_state}")
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
    }


def create_pose_estimator(
    pose_backend: str,
    model_path: str | Path | None = None,
    yolo_model_path: str | Path | None = None,
):
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
    fall_validator_settings: Mapping[str, float | int] | None = None,
    temporal_sensitivity: str = "high",
    fusion_fall_confirmation_steps: int = 3,
    use_static_lying_adl_filter: bool = True,
):
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
                else "models/tcn_prefall_classifier.pt"
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
            fall_validator_settings=fall_validator_settings,
            temporal_sensitivity=temporal_sensitivity,
        )
    raise ValueError(f"Unknown predictor type: {predictor_type}")


class ImageSequenceCapture:

    def __init__(self, image_dir: str | Path, fps: float = 30.0) -> None:
        self.image_dir = Path(image_dir)
        self.image_paths = find_image_sequence_files(self.image_dir)
        self.fps = fps if fps > 0 else infer_image_sequence_fps(self.image_paths)
        self.index = 0
        self.width = 0
        self.height = 0

        if self.image_paths:
            import cv2

            first_frame = cv2.imread(str(self.image_paths[0]))
            if first_frame is not None:
                self.height, self.width = first_frame.shape[:2]

    def isOpened(self) -> bool:
        return bool(self.image_paths) and self.width > 0 and self.height > 0

    def read(self):
        import cv2

        while self.index < len(self.image_paths):
            image_path = self.image_paths[self.index]
            self.index += 1
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: could not read image, skipped: {image_path}")
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            return True, frame

        return False, None

    def get(self, prop_id: int) -> float:
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
        return None


def open_frame_source(source: str | int, image_sequence_fps: float = 30.0):
    import cv2

    if isinstance(source, int):
        return open_camera_capture(source)

    source_path = Path(source)
    if source_path.is_dir():
        return ImageSequenceCapture(source_path, fps=image_sequence_fps)

    return cv2.VideoCapture(str(source_path))


def find_image_sequence_files(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_sort_key)


def infer_image_sequence_fps(image_paths: list[Path], default_fps: float = 30.0) -> float:
    timestamps = [_timestamp_from_image_name(path) for path in image_paths]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if len(timestamps) < 2:
        return default_fps

    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    if duration <= 0:
        return default_fps
    return (len(timestamps) - 1) / duration


def _timestamp_from_image_name(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, "%Y-%m-%dT%H_%M_%S.%f")
    except ValueError:
        return None


def natural_sort_key(path: str | Path) -> list[int | str]:
    name = Path(path).name.lower()
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def parse_source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fall prediction on a video, image sequence, or webcam.")
    parser.add_argument("--source", default="0", help="Video path, image directory, or camera index. Default: camera 0.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path. Omit this option to disable CSV export.",
    )
    parser.add_argument("--output-video", default=None, help="Optional annotated MP4 output path.")
    parser.add_argument("--model", default=None, help="Optional MediaPipe Tasks pose landmarker .task model path.")
    parser.add_argument(
        "--pose-backend",
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help="Pose backend: mediapipe uses the legacy pipeline; yolo uses Ultralytics YOLO-pose.",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo26n-pose.pt",
        help="Path to the YOLO-pose .pt model used with --pose-backend yolo.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON configuration file, for example configs/default.json.")
    parser.add_argument(
        "--sensitivity",
        choices=SENSITIVITY_LEVELS,
        default=None,
        help="Detection sensitivity: low is conservative, medium is the default, and high is more responsive. Uses the configured or built-in default when omitted.",
    )
    parser.add_argument("--image-fps", type=float, default=30.0, help="Frame rate used when --source is an image directory. Default: 30.")
    parser.add_argument(
        "--predictor",
        choices=("rule", "ml", "deep", "fusion", "ensemble"),
        default="rule",
        help="Predictor backend: rule, tree-based ml, deep fusion, or cooperative dual-model ensemble.",
    )
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Classifier path; in ensemble mode, this is the tree-model path.",
    )
    parser.add_argument(
        "--fusion-model",
        default="models/skeleton_feature_fusion_tuned.pt",
        help="Skeleton-feature fusion model path used in ensemble mode.",
    )
    parser.add_argument(
        "--prefall-alert-threshold",
        type=float,
        default=None,
        help="ML alert threshold: emit Alert: Pre-fall when Pre-fall probability remains elevated even if state is Normal.",
    )
    parser.add_argument(
        "--prefall-alert-frames",
        type=int,
        default=None,
        help="Consecutive frames required above --prefall-alert-threshold. Default: 1.",
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        help="Enable HMM Viterbi smoothing to reduce state jitter and one-frame false alarms.",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="Use acceleration features during inference; the model must also be trained with --use-accel.",
    )
    parser.add_argument(
        "--disable-temporal-fall-validation",
        action="store_true",
        help="Disable runtime temporal Fall confirmation and rely on model/HMM output.",
    )
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview window.")
    args = parser.parse_args()
    sensitivity_ml_config = ml_config_for_sensitivity(args.sensitivity) if args.sensitivity else None
    predictor_config = (
        load_predictor_config(args.config)
        if args.config
        else predictor_config_for_sensitivity(args.sensitivity)
        if args.sensitivity
        else None
    )
    prefall_alert_threshold = (
        args.prefall_alert_threshold
        if args.prefall_alert_threshold is not None or sensitivity_ml_config is None
        else sensitivity_ml_config.prefall_alert_threshold
    )
    prefall_alert_frames = (
        args.prefall_alert_frames
        if args.prefall_alert_frames is not None or sensitivity_ml_config is None
        else sensitivity_ml_config.prefall_alert_frames
    )
    fall_validator_settings = (
        sensitivity_ml_config.fall_validator_settings
        if sensitivity_ml_config is not None
        else None
    )

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
        prefall_alert_threshold=prefall_alert_threshold,
        prefall_alert_frames=prefall_alert_frames,
        use_hmm=args.use_hmm,
        use_accel=args.use_accel if args.use_accel else None,
        use_temporal_fall_validation=not args.disable_temporal_fall_validation,
        fall_validator_settings=fall_validator_settings,
        temporal_sensitivity=args.sensitivity or "high",
        image_sequence_fps=args.image_fps,
        predictor_config=predictor_config,
    )


if __name__ == "__main__":
    main()
