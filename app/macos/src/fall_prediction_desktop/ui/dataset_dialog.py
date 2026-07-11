"""Simple manager for FallGuard-owned imported and event media."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class DatasetDialog(QDialog):
    def __init__(self, media_repository, media_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._media = media_repository
        self._media_root = media_root.resolve()
        self.setWindowTitle("Dataset Management")
        self.resize(720, 460)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Imported media, annotated outputs, and event evidence"))
        self._list = QListWidget()
        layout.addWidget(self._list, 1)
        actions = QHBoxLayout()
        reveal = QPushButton("Reveal in Finder")
        delete = QPushButton("Delete Managed File")
        close = QPushButton("Close")
        reveal.clicked.connect(self._reveal)
        delete.clicked.connect(self._delete)
        close.clicked.connect(self.accept)
        actions.addWidget(reveal)
        actions.addWidget(delete)
        actions.addStretch(1)
        actions.addWidget(close)
        layout.addLayout(actions)
        self._reload()

    def _reload(self) -> None:
        self._list.clear()
        for row in self._media.list_all():
            path = Path(row["file_path"])
            status = row.get("status", "unknown")
            item = QListWidgetItem(f"{row['media_type']} · {status} · {path.name}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            self._list.addItem(item)

    def _selected(self) -> dict | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _reveal(self) -> None:
        row = self._selected()
        if row:
            path = Path(row["file_path"])
            target = path if path.exists() else path.parent
            subprocess.run(["open", "-R", str(target)], check=False)

    def _delete(self) -> None:
        row = self._selected()
        if not row:
            return
        path = Path(row["file_path"]).expanduser()
        try:
            resolved = path.resolve()
            if resolved != self._media_root and self._media_root not in resolved.parents:
                QMessageBox.warning(self, "Dataset Management", "External source files are never deleted.")
                return
            if resolved.is_dir():
                QMessageBox.warning(self, "Dataset Management", "Managed folders cannot be deleted here.")
                return
            resolved.unlink(missing_ok=True)
            self._media.delete(row["id"])
            self._reload()
        except OSError as exc:
            QMessageBox.warning(self, "Dataset Management", str(exc))
