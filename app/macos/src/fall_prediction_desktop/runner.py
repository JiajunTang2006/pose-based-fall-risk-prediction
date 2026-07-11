from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]
PredictionCallback = Callable[[object, int, float, object], None]


@dataclass(frozen=True)
class PredictionJob:
    source: str
    output_dir: Path
    pose_backend: str = "yolo"
    predictor: str = "rule"
    sensitivity: str | None = None
    config_path: Path | None = None
    mediapipe_model_path: Path | None = None
    yolo_model_path: Path | None = None
    classifier_model_path: Path | None = None
    write_csv: bool = True
    write_video: bool = True
    show_preview: bool = False
    image_fps: float = 30.0
    prefall_alert_threshold: float | None = None
    prefall_alert_frames: int | None = None
    use_hmm: bool = False
    use_accel: bool = False
    use_temporal_fall_validation: bool = True


@dataclass(frozen=True)
class RunResult:
    output_csv: Path | None
    output_video: Path | None


def find_app_root() -> Path:
    """Find the macos app directory (contains ``web/``, ``models/``, ``assets/``).

    Inside a PyInstaller bundle, ``sys._MEIPASS`` points to the ``Resources/``
    directory where the models, web, and assets folders are stored.  When
    running from source we search upward from this file's location.
    """
    # PyInstaller sets _MEIPASS to the bundled Resources/ directory.
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)

    source_dir = Path(__file__).resolve().parent
    for candidate in (source_dir, *source_dir.parents):
        if (candidate / "web").is_dir() and (candidate / "models").is_dir():
            return candidate
    # Fallback: ../.. from this file (fall_prediction_desktop → src → macos)
    return source_dir.parent.parent


def find_repo_root() -> Path:
    """Deprecated — the app is now self-contained.  Returns ``find_app_root()``."""
    return find_app_root()


def ensure_repo_on_path(app_root: Path) -> None:
    """Ensure the local ``src/`` directory is on ``sys.path``.

    Inside a PyInstaller bundle the ``src/`` directory does not exist;
    the import system resolves modules from the bundled archive instead,
    so this is a harmless no-op in that environment.
    """
    src_dir = str(app_root / "src")
    if src_dir not in sys.path and Path(src_dir).is_dir():
        sys.path.insert(0, src_dir)


def run_prediction_job(
    job: PredictionJob,
    log: LogCallback | None = None,
    on_prediction: PredictionCallback | None = None,
) -> RunResult:
    app_root = find_app_root()
    ensure_repo_on_path(app_root)

    source = normalize_source(job.source)
    validate_job(job, source, app_root)
    output_csv, output_video = build_output_paths(job, source)

    emit = log or (lambda _message: None)
    emit(f"App: {app_root}")
    emit(f"Source: {source}")
    if output_csv:
        emit(f"CSV: {output_csv}")
    if output_video:
        emit(f"Video: {output_video}")
    emit("Starting prediction...")

    from fall_prediction.config import load_predictor_config
    from fall_prediction.sensitivity import (
        ml_config_for_sensitivity,
        normalize_sensitivity,
        predictor_config_for_sensitivity,
    )
    from fall_prediction.video_app import process_video

    sensitivity = normalize_sensitivity(job.sensitivity) if job.sensitivity else None
    sensitivity_ml_config = ml_config_for_sensitivity(sensitivity) if sensitivity else None
    predictor_config = (
        load_predictor_config(job.config_path)
        if job.config_path
        else predictor_config_for_sensitivity(sensitivity)
        if sensitivity
        else None
    )
    prefall_alert_threshold = (
        job.prefall_alert_threshold
        if job.prefall_alert_threshold is not None or sensitivity_ml_config is None
        else sensitivity_ml_config.prefall_alert_threshold
    )
    prefall_alert_frames = (
        job.prefall_alert_frames
        if job.prefall_alert_frames is not None or sensitivity_ml_config is None
        else sensitivity_ml_config.prefall_alert_frames
    )
    fall_validator_settings = (
        sensitivity_ml_config.fall_validator_settings
        if sensitivity_ml_config is not None
        else None
    )
    process_video(
        source=source,
        output_csv=output_csv,
        output_video=output_video,
        model_path=resolve_optional_path(job.mediapipe_model_path, app_root),
        pose_backend=job.pose_backend,
        yolo_model_path=resolve_optional_path(job.yolo_model_path, app_root),
        show=job.show_preview,
        predictor_type=job.predictor,
        classifier_model_path=resolve_optional_path(job.classifier_model_path, app_root),
        prefall_alert_threshold=prefall_alert_threshold,
        prefall_alert_frames=prefall_alert_frames,
        use_hmm=job.use_hmm,
        use_accel=job.use_accel if job.use_accel else None,
        use_temporal_fall_validation=job.use_temporal_fall_validation,
        fall_validator_settings=fall_validator_settings,
        temporal_sensitivity=sensitivity or "high",
        image_sequence_fps=job.image_fps,
        predictor_config=predictor_config,
        on_prediction=on_prediction,
    )

    emit("Done.")
    return RunResult(output_csv=output_csv, output_video=output_video)


def normalize_source(source_text: str) -> str | int:
    text = source_text.strip()
    if not text:
        raise ValueError("Please choose a video file, image folder, or camera.")
    if text.isdigit():
        return int(text)
    return str(Path(text).expanduser())


def validate_job(job: PredictionJob, source: str | int, repo_root: Path) -> None:
    if job.pose_backend not in {"mediapipe", "yolo"}:
        raise ValueError(f"Unknown pose backend: {job.pose_backend}")
    if job.predictor not in {"rule", "ml"}:
        raise ValueError(f"Unknown predictor: {job.predictor}")
    if job.image_fps <= 0:
        raise ValueError("Image sequence FPS must be greater than 0.")
    if not (job.write_csv or job.write_video or job.show_preview):
        raise ValueError("Choose at least one output or enable preview.")
    if isinstance(source, int) and not job.show_preview:
        raise ValueError("Camera mode needs preview enabled so you can press q to stop.")
    if isinstance(source, str) and not Path(source).exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    for label, maybe_path in (
        ("Config", job.config_path),
        ("MediaPipe model", job.mediapipe_model_path),
        ("YOLO model", job.yolo_model_path if job.pose_backend == "yolo" else None),
        ("Classifier model", job.classifier_model_path if job.predictor == "ml" else None),
    ):
        resolved = resolve_optional_path(maybe_path, repo_root)
        if resolved is not None and not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")


def build_output_paths(job: PredictionJob, source: str | int) -> tuple[Path | None, Path | None]:
    output_dir = job.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = f"camera{source}" if isinstance(source, int) else Path(source).stem
    safe_name = safe_filename(source_name or "source")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_csv = output_dir / f"{safe_name}_{timestamp}_predictions.csv" if job.write_csv else None
    output_video = output_dir / f"{safe_name}_{timestamp}_annotated.mp4" if job.write_video else None
    return output_csv, output_video


def resolve_optional_path(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return repo_root / expanded


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._ ")
    return cleaned or "source"
