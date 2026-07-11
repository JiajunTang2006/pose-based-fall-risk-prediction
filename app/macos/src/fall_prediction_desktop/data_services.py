"""Export and safely clear FallGuard's persisted history."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExportResult:
    path: Path
    counts: dict[str, int]


class ExportService:
    TABLES = ("profiles", "monitoring_sessions", "events", "risk_samples", "media_files")

    def __init__(self, db) -> None:
        self._db = db

    def export(self, destination: Path) -> ExportResult:
        destination = destination.expanduser()
        connection = self._db.get_connection()
        data = {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table}").fetchall()]
            for table in self.TABLES
        }
        counts = {table: len(rows) for table, rows in data.items()}
        manifest = {
            "formatVersion": 1,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
        }

        if destination.suffix.lower() == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {"manifest": manifest, **data}
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return ExportResult(destination, counts)

        destination = destination.with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for table, rows in data.items():
                with (root / f"{table}.csv").open("w", newline="", encoding="utf-8") as stream:
                    fieldnames = list(rows[0]) if rows else []
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    if fieldnames:
                        writer.writeheader()
                        writer.writerows(rows)
            temporary = destination.with_suffix(".partial.zip")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for item in root.iterdir():
                    archive.write(item, item.name)
            temporary.replace(destination)
        return ExportResult(destination, counts)


class HistoryService:
    def __init__(self, db, media_root: Path) -> None:
        self._db = db
        self._media_root = media_root.resolve()

    def clear(self) -> list[str]:
        connection = self._db.get_connection()
        rows = connection.execute("SELECT file_path FROM media_files").fetchall()
        warnings: list[str] = []
        connection.execute("BEGIN")
        try:
            connection.execute("DELETE FROM media_files")
            connection.execute("DELETE FROM monitoring_sessions")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        for row in rows:
            path = Path(row["file_path"]).expanduser()
            try:
                resolved = path.resolve()
                if resolved == self._media_root or self._media_root in resolved.parents:
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink(missing_ok=True)
            except OSError as exc:
                warnings.append(f"{path}: {exc}")
        return warnings
