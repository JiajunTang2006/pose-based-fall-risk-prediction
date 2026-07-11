"""Capture event thumbnails and bounded pre/post-roll video evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database.init_db import AppRepositories


@dataclass(frozen=True)
class BufferedFrame:
    timestamp: float
    jpeg: bytes


@dataclass
class ActiveCapture:
    event_id: str
    frames: list[BufferedFrame]
    post_until: float
    thumbnail_path: Path
    clip_path: Path


class EventMediaBuffer:
    """Maintain a bounded JPEG ring and persist evidence outside the app bundle."""

    def __init__(
        self,
        repos: "AppRepositories | None",
        output_root: Path,
        session_id: str | None,
        fps: float,
        pre_seconds: float = 5.0,
        post_seconds: float = 10.0,
        clip_fps: float = 10.0,
    ) -> None:
        self._repos = repos
        self._output_root = output_root
        self._session_id = session_id
        self._pre_seconds = max(pre_seconds, 0.0)
        self._post_seconds = max(post_seconds, 0.0)
        self._clip_fps = max(clip_fps, 1.0)
        max_frames = max(1, int(round(self._pre_seconds * self._clip_fps)) + 2)
        self._pre_roll: deque[BufferedFrame] = deque(maxlen=max_frames)
        self._active: ActiveCapture | None = None
        self._last_sample_at = float("-inf")
        self._source_fps = max(fps, 1.0)

    def add_frame(self, frame, timestamp: float, event_id: str | None) -> None:
        if timestamp - self._last_sample_at < 1.0 / self._clip_fps:
            return
        jpeg = self._encode_jpeg(frame)
        if jpeg is None:
            return
        self._last_sample_at = timestamp
        buffered = BufferedFrame(timestamp=timestamp, jpeg=jpeg)
        self._pre_roll.append(buffered)

        if event_id is not None and (
            self._active is None or self._active.event_id != event_id
        ):
            self._start_capture(event_id, buffered, timestamp)
        if self._active is not None:
            if not self._active.frames or self._active.frames[-1].timestamp != timestamp:
                self._active.frames.append(buffered)
            if event_id is not None:
                self._active.post_until = timestamp + self._post_seconds
            elif timestamp >= self._active.post_until:
                self._finalize_active()

    def close(self) -> None:
        self._finalize_active()

    def _start_capture(self, event_id: str, trigger: BufferedFrame, timestamp: float) -> None:
        if self._active is not None:
            self._finalize_active()
        event_dir = self._output_root / "events" / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        thumbnail = event_dir / "thumbnail.jpg"
        clip = event_dir / "clip.mp4"
        thumbnail.write_bytes(trigger.jpeg)
        self._active = ActiveCapture(
            event_id=event_id,
            frames=list(self._pre_roll),
            post_until=timestamp + self._post_seconds,
            thumbnail_path=thumbnail,
            clip_path=clip,
        )
        if self._repos is not None:
            self._repos.events.update_media_paths(event_id, thumbnail_path=str(thumbnail))
            record = self._repos.media.create(
                file_path=str(thumbnail),
                media_type="thumbnail",
                session_id=self._session_id,
                event_id=event_id,
                file_size_bytes=len(trigger.jpeg),
            )
            self._repos.media.update_status(record["id"], "complete")

    def _finalize_active(self) -> None:
        capture = self._active
        self._active = None
        if capture is None or not capture.frames:
            return
        try:
            self._write_clip(capture)
            if self._repos is not None:
                self._repos.events.update_media_paths(
                    capture.event_id,
                    video_clip_path=str(capture.clip_path),
                )
                record = self._repos.media.create(
                    file_path=str(capture.clip_path),
                    media_type="event_clip",
                    session_id=self._session_id,
                    event_id=capture.event_id,
                    file_size_bytes=capture.clip_path.stat().st_size,
                    fps=self._clip_fps,
                    duration_seconds=len(capture.frames) / self._clip_fps,
                )
                self._repos.media.update_status(record["id"], "complete")
        except Exception:
            capture.clip_path.unlink(missing_ok=True)

    def _write_clip(self, capture: ActiveCapture) -> None:
        import cv2
        import numpy as np

        first = cv2.imdecode(np.frombuffer(capture.frames[0].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError("Could not decode buffered event frame.")
        height, width = first.shape[:2]
        temporary = capture.clip_path.with_suffix(".partial.mp4")
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._clip_fps,
            (width, height),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Could not open event clip writer.")
            for buffered in capture.frames:
                frame = cv2.imdecode(
                    np.frombuffer(buffered.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    writer.write(frame)
        finally:
            writer.release()
        temporary.replace(capture.clip_path)

    @staticmethod
    def _encode_jpeg(frame) -> bytes | None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if ok else None
