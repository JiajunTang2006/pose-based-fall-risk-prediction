"""
Reusable custom PySide6 widgets — M3 design language.
Full i18n coverage via t() from .i18n.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QEasingCurve, QTimer, QVariantAnimation
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QPushButton,
    QSizePolicy,
)

# Relative imports work when running as `python -m fall_prediction_desktop`.
# Absolute imports work inside a PyInstaller bundle where relative imports fail.
try:
    from .i18n import t
except ImportError:
    from fall_prediction_desktop.ui.i18n import t  # type: ignore[no-redef]

# 8px grid
S4, S8, S12, S16, S20, S24, S28, S32 = 4, 8, 12, 16, 20, 24, 28, 32


# ── Card ───────────────────────────────────────────────────────────

class Card(QFrame):
    """M3 elevated card: surface_bright background + outline_variant border."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "Card")


# ── Risk Ring ──────────────────────────────────────────────────────

class RiskRing(QWidget):
    """Circular risk gauge with colored arc."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._percent = 0
        self._color = QColor("#34A853")
        self._track = QColor("#E5E7EB")
        self.setFixedSize(140, 140)

    def set_risk(self, percent: int, color: str, track: str) -> None:
        self._percent = max(0, min(100, percent))
        self._color = QColor(color)
        self._track = QColor(track)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        ring_w = 12
        outer = min(w, h) / 2 - 4

        # Track (full ring)
        pen = QPen(self._track, ring_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(QRectF(cx - outer, cy - outer, outer * 2, outer * 2), 0, 360 * 16)

        # Colored arc
        if self._percent > 0:
            pen.setColor(self._color)
            p.setPen(pen)
            span = int(-self._percent / 100 * 360 * 16)
            p.drawArc(QRectF(cx - outer, cy - outer, outer * 2, outer * 2), 90 * 16, span)

        # Percent
        font = QFont()
        font.setPixelSize(40)
        font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(self._color)
        p.drawText(QRectF(0, cy - 30, w, 44), Qt.AlignmentFlag.AlignCenter,
                   f"{self._percent}{t('common.percent', '%')}")

        # Subtitle
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor("#6B7280"))
        p.drawText(QRectF(0, cy + 14, w, 18), Qt.AlignmentFlag.AlignCenter,
                   t("widgets.riskSubtitle", "Risk"))

        p.end()


# ── Activity Row ────────────────────────────────────────────────────

class ActivityRow(QFrame):
    """Single activity item with colored indicator dot."""

    _LEVELS = {
        "normal":  ("#34A853", "#E6F4EA"),
        "warning": ("#FBBC04", "#FEF7E0"),
        "danger":  ("#EA4335", "#FCE8E6"),
        "muted":   ("#6B7280", "#F1F3F4"),
    }

    def __init__(self, level: str, title: str, time_str: str, risk: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        fg, _ = self._LEVELS.get(level, self._LEVELS["muted"])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S12, 0, S12, 0)
        layout.setSpacing(S12)

        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {fg}; border-radius: 5px; border: none;")
        layout.addWidget(dot)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: 600; font-size: 13px; border: none;")
        layout.addWidget(title_lbl, 1)

        time_lbl = QLabel(time_str)
        time_lbl.setStyleSheet("font-size: 12px; border: none;")
        time_lbl.setFixedWidth(90)
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(time_lbl)

        risk_text = t("widgets.riskScoreFmt", "Risk {risk}%").replace("{risk}", str(risk))
        risk_lbl = QLabel(risk_text)
        risk_lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {fg}; border: none;")
        risk_lbl.setFixedWidth(72)
        risk_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(risk_lbl)


# ── Connection Pill ─────────────────────────────────────────────────

class ConnectionPill(QFrame):
    """Status pill: green/gray dot + label with border."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S12, 0, S12, 0)
        layout.setSpacing(S8)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        layout.addWidget(self._dot)

        self._text = QLabel(t("topbar.cameraReady", "Camera Ready"))
        self._text.setStyleSheet("font-size: 13px; font-weight: 600; border: none;")
        layout.addWidget(self._text)

        self.set_connected(False)

    def set_connected(self, connected: bool) -> None:
        if connected:
            fg = "#137333"
            bg = "#E6F4EA"
            border = "#CEEAD6"
        else:
            fg = "#5F6368"
            bg = "#F1F3F4"
            border = "#E0E3EB"
        self._dot.setStyleSheet(f"background: {fg}; border-radius: 4px; border: none;")
        self.setStyleSheet(
            f"background: {bg}; border: 1px solid {border}; border-radius: 16px;"
        )
        self._text.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {fg}; border: none;"
        )

    def set_text(self, text: str) -> None:
        self._text.setText(text)


# ── Monitoring Tag ──────────────────────────────────────────────────

class MonitoringTag(QFrame):
    """M3 status chip: Active / Idle / Warning / Critical with border."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S12, 0, S12, 0)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        layout.addWidget(self._dot)

        self._text = QLabel(t("monitoring.idle", "Idle"))
        self._text.setStyleSheet("font-size: 13px; font-weight: 600; border: none;")
        layout.addWidget(self._text)

        self.set_state(t("monitoring.idle", "Idle"), "#F1F3F4", "#5F6368", "#E0E3EB")

    def set_state(self, text: str, bg: str, fg: str, border: str = "#E0E3EB") -> None:
        self._text.setText(text)
        self._dot.setStyleSheet(f"background: {fg}; border-radius: 4px; border: none;")
        self._text.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {fg}; border: none;")
        self.setStyleSheet(
            f"background: {bg}; border: 1px solid {border}; border-radius: 16px;"
        )


# ── Video Shell ─────────────────────────────────────────────────────

class VideoShell(QFrame):
    """Live camera feed that fills the whole rounded preview surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: #111827; border-radius: 16px; border: none;")
        self.setMinimumHeight(250)
        self._source_pixmap: QPixmap | None = None

        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.setStyleSheet("background: transparent; border: none;")

        self._placeholder = QLabel(self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "background: transparent; color: #D1D5DB; font-size: 14px; "
            "font-weight: 500; border: none;"
        )
        self._placeholder.setText(
            t("widgets.cameraPlaceholder",
              "No camera feed\n\nClick Start Monitoring to begin live detection")
        )

        self._fps_lbl = self._make_overlay_label(
            t("monitor.fps", "FPS:") + " " + t("common.na", "--")
        )
        self._res_lbl = self._make_overlay_label(
            t("monitor.resolution", "Resolution:") + " " + t("common.na", "--")
        )

    def set_frame(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self._placeholder.hide()
        self._render_frame()
        QTimer.singleShot(0, self._render_frame)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_children()
        if self._source_pixmap is not None:
            self._render_frame()

    def _make_overlay_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        label.setStyleSheet(
            "background: rgba(17,24,39,0.78); color: #E5E7EB; "
            "font-size: 12px; font-weight: 600; border: none; "
            "border-radius: 8px; padding: 5px 10px;"
        )
        label.adjustSize()
        label.raise_()
        return label

    def _layout_children(self) -> None:
        self._image.setGeometry(self.rect())
        self._placeholder.setGeometry(self.rect().adjusted(S16, S16, -S16, -S16))
        for label in (self._fps_lbl, self._res_lbl):
            label.adjustSize()
            label.raise_()
        overlay_y = max(S12, self.height() - self._fps_lbl.height() - S16)
        self._fps_lbl.move(S16, overlay_y)
        self._res_lbl.move(max(S16, self.width() - self._res_lbl.width() - S16), overlay_y)

    def _render_frame(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        self._layout_children()
        target_size = self.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - target_size.width()) // 2)
        y = max(0, (scaled.height() - target_size.height()) // 2)
        cropped = scaled.copy(x, y, target_size.width(), target_size.height())
        rounded = QPixmap(target_size)
        rounded.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, target_size.width(), target_size.height()), 16, 16)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, cropped)
        painter.end()
        self._image.setPixmap(rounded)

    def show_placeholder(self) -> None:
        self._source_pixmap = None
        self._image.clear()
        self._placeholder.show()

    def set_info(self, fps: float, resolution: str) -> None:
        na = t("common.na", "--")
        self._fps_lbl.setText(
            t("monitor.fps", "FPS:") + f" {fps:.1f}" if fps > 0
            else t("monitor.fps", "FPS:") + " " + na
        )
        self._res_lbl.setText(
            t("monitor.resolution", "Resolution:") + f" {resolution}"
            if resolution and resolution != "--"
            else t("monitor.resolution", "Resolution:") + " " + na
        )
        self._layout_children()


# ── Sidebar ─────────────────────────────────────────────────────────

class Sidebar(QFrame):
    """Dashboard + Settings sidebar for the desktop shell."""

    def __init__(self, assets_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._width = 216
        self._collapsed = False
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(self._width)
        self._build(assets_dir)

    def _build(self, assets_dir: Path | None) -> None:
        self._content = QFrame(self)
        self._content.setObjectName("sidebarContent")
        self._content.setFixedWidth(self._width)
        self._content.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(S20, S28, S20, S20)
        layout.setSpacing(0)

        # Brand with logo
        brand = QHBoxLayout()
        brand.setSpacing(S16)

        self._logo = QLabel()
        self._logo.setFixedSize(40, 40)
        if assets_dir:
            p = assets_dir / "FallGuard.png"
            if p.exists():
                pix = QPixmap(str(p))
                self._logo.setPixmap(pix.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        self._logo.setStyleSheet("border-radius: 12px; border: none;")
        brand.addWidget(self._logo)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        self._brand_title = QLabel(t("brand.name", "FallGuard"))
        self._brand_title.setStyleSheet("font-size: 18px; font-weight: 700; background: transparent; border: none;")
        text_col.addWidget(self._brand_title)
        self._brand_sub = QLabel(t("sidebar.aiFallDetection", "AI fall detection"))
        self._brand_sub.setStyleSheet("font-size: 11px; background: transparent; border: none;")
        text_col.addWidget(self._brand_sub)
        brand.addLayout(text_col, 1)
        layout.addLayout(brand)

        layout.addSpacing(32)

        self._nav_btn = QPushButton(f"  {t('sidebar.navDashboard', 'Dashboard')}")
        self._nav_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_btn.setObjectName("navDashboard")
        self._nav_btn.setFixedHeight(48)
        layout.addWidget(self._nav_btn)
        layout.addSpacing(S8)

        self._settings_btn = QPushButton(f"  {t('sidebar.settings', 'Settings')}")
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setObjectName("sidebarSettingsBtn")
        self._settings_btn.setFixedHeight(48)
        layout.addWidget(self._settings_btn)

        layout.addStretch()

        self._status_box = QFrame()
        self._status_box.setObjectName("sidebarStatusBox")
        status_layout = QVBoxLayout(self._status_box)
        status_layout.setContentsMargins(S16, S16, S16, S16)
        status_layout.setSpacing(S8)
        self._status_title = QLabel(t("status.systemStatus", "System Status"))
        self._status_title.setObjectName("sidebarStatusTitle")
        self._status_detail = QLabel(t("status.systemWaiting", "System is waiting to start."))
        self._status_detail.setWordWrap(True)
        self._status_detail.setObjectName("sidebarStatusDetail")
        status_layout.addWidget(self._status_title)
        status_layout.addWidget(self._status_detail)
        layout.addWidget(self._status_box)

    def toggle(self) -> None:
        if hasattr(self, "_anim") and self._anim.state() == QVariantAnimation.State.Running:
            self._anim.stop()
        self._collapsed = not self._collapsed
        if not self._collapsed:
            self._content.show()

        start_width = self.width()
        end_width = 0 if self._collapsed else self._width

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(start_width)
        self._anim.setEndValue(end_width)
        self._anim.valueChanged.connect(lambda value: self._set_sidebar_width(int(value)))
        self._anim.finished.connect(lambda: self._finish_toggle(end_width))
        self._anim.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_content"):
            self._content.setGeometry(0, 0, self._width, self.height())

    def _set_sidebar_width(self, width: int) -> None:
        width = max(0, min(self._width, width))
        self.setFixedWidth(width)
        self.updateGeometry()

    def _finish_toggle(self, width: int) -> None:
        self._set_sidebar_width(width)
        self._content.setVisible(width > 0)

    def refresh_labels(self) -> None:
        """Re-translate all sidebar labels after a language switch."""
        self._brand_title.setText(t("brand.name", "FallGuard"))
        self._brand_sub.setText(t("sidebar.aiFallDetection", "AI fall detection"))
        self._nav_btn.setText(f"  {t('sidebar.navDashboard', 'Dashboard')}")
        self._settings_btn.setText(f"  {t('sidebar.settings', 'Settings')}")
        self._status_title.setText(t("status.systemStatus", "System Status"))
        self._status_detail.setText(t("status.systemWaiting", "System is waiting to start."))

    @property
    def settings_button(self) -> QPushButton:
        return self._settings_btn

    @property
    def dashboard_button(self) -> QPushButton:
        return self._nav_btn

    @property
    def width_default(self) -> int:
        return self._width

    def update_theme_colors(self, c: dict[str, str]) -> None:
        hover = c["surface_container_low"]
        text = c["on_surface"]
        sep_color = c["outline_variant"]
        active_bg = c["primary_container"]
        active_color = c["primary"]

        btn_style = (
            f"QPushButton {{ text-align: left; padding: 0 16px; border-radius: 16px; "
            f"font-size: 14px; font-weight: 500; border: none; color: {c['on_surface_variant']}; }}"
            f"QPushButton:hover {{ background: {hover}; border: none; }}"
        )
        for name in ("navDashboard", "sidebarSettingsBtn"):
            for btn in self.findChildren(QPushButton, name):
                btn.setStyleSheet(btn_style)
                if name == "navDashboard":
                    btn.setStyleSheet(
                        f"QPushButton {{ text-align: left; padding: 0 16px; border-radius: 16px; "
                        f"font-size: 14px; font-weight: 600; border: none; "
                        f"color: {active_color}; background: {active_bg}; }}"
                        f"QPushButton:hover {{ background: {active_bg}; border: none; }}"
                    )

        for s in self.findChildren(QFrame, "sidebarSep"):
            s.setStyleSheet(f"background: {sep_color}; border: none;")
        self._status_box.setStyleSheet(
            f"#sidebarStatusBox {{ background: {c['success_bg']}; "
            f"border: 1px solid {c['secondary_container']}; border-radius: 18px; }}"
        )
        self._status_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {c['success_fg']}; "
            f"background: transparent; border: none;"
        )
        self._status_detail.setStyleSheet(
            f"font-size: 12px; color: {c['on_surface_secondary']}; "
            f"background: transparent; border: none;"
        )
