"""
Main window — M3 Google-style AI Safety Dashboard.
8px spacing grid. Surface layering: bg → surface → card.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QFileDialog, QMessageBox, QMenu,
)

from .theme import ThemeManager, build_stylesheet, LIGHT, DARK
from .i18n import get_i18n, t, t_dynamic, init_i18n
from .widgets import (
    RiskRing, ActivityRow, ConnectionPill,
    MonitoringTag, VideoShell, Sidebar, Card,
)
from .settings_dialog import SettingsDialog

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── Spacing constants (8px grid) ──────────────────────────────────
S4, S8, S12, S16, S20, S24, S32 = 4, 8, 12, 16, 20, 24, 32


# ── Risk Chart ────────────────────────────────────────────────────

class RiskChart(FigureCanvasQTAgg):
    """M3-style risk trend chart using matplotlib."""

    def __init__(self, parent=None) -> None:
        self._fig = Figure(figsize=(4.0, 2.2), dpi=100, tight_layout=True)
        self._fig.patch.set_alpha(0)
        self._ax = self._fig.add_subplot(111)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setMinimumSize(320, 130)
        self.setMaximumHeight(140)
        self.setStyleSheet("background: transparent; border: none;")
        self._history: list[int] = []
        self._colors = LIGHT
        self._ax.set_facecolor("none")
        self._ax.set_ylim(0, 105)
        self._ax.set_yticks([0, 25, 50, 75, 100])
        self._ax.set_xticks([])
        self._ax.tick_params(labelsize=8, pad=2)
        self._ax.spines["top"].set_visible(False)
        self._ax.spines["right"].set_visible(False)

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        self._colors = colors
        self._ax.tick_params(colors=colors.get("chart_text", "#5F6368"))
        self._ax.spines["left"].set_color(colors.get("chart_grid", "#E2E4E8"))
        self._ax.spines["bottom"].set_color(colors.get("chart_grid", "#E2E4E8"))
        self.draw_idle()

    def update_data(self, risk_history: list[int], risk_color: str) -> None:
        self._history = risk_history[-48:]
        self._ax.clear()
        self._ax.set_facecolor("none")
        self._ax.set_ylim(0, 105)
        self._ax.set_yticks([0, 25, 50, 75, 100])
        tcol = self._colors.get("chart_text", "#5F6368")
        gcol = self._colors.get("chart_grid", "#E2E4E8")
        self._ax.tick_params(labelsize=8, pad=2, colors=tcol)
        self._ax.spines["top"].set_visible(False)
        self._ax.spines["right"].set_visible(False)
        self._ax.spines["left"].set_color(gcol)
        self._ax.spines["bottom"].set_color(gcol)
        if not self._history:
            self.draw_idle()
            return
        values = list(self._history)
        if len(values) == 1:
            values = [0, values[0]]
        tick_positions = [0, max(len(values) - 1, 1)]
        self._ax.set_xticks(tick_positions)
        self._ax.set_xticklabels(["-60s", "Now"], color=tcol)
        self._ax.fill_between(range(len(values)), values, alpha=0.12, color=risk_color)
        self._ax.plot(values, color=risk_color, linewidth=2.5, solid_capstyle="round")
        self._ax.plot(len(values) - 1, values[-1], "o", color=risk_color, markersize=8)
        self._ax.set_xlim(-0.5, max(len(values) - 1, 1) + 0.5)
        self._fig.tight_layout(pad=0.3)
        self.draw_idle()


# ── Main Window ───────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, monitor, media_processor, profile_manager,
                 settings, app_root: Path, locales_dir: Path,
                 assets_dir: Path, app_version: str = "0.2.0") -> None:
        super().__init__()
        self._monitor = monitor
        self._media_processor = media_processor
        self._profile_manager = profile_manager
        self._app_settings = settings
        self._app_root = app_root
        self._app_version = app_version
        self._assets_dir = assets_dir

        init_i18n(locales_dir)
        self._theme = ThemeManager(settings.theme)
        self._theme.theme_changed.connect(self._on_theme_changed)
        self._theme_colors = LIGHT

        self._risk_history: list[int] = []
        self._frame_timer: QTimer | None = None
        self._status_timer: QTimer | None = None
        self._last_media_state = ""
        self._media_activities: list[dict] = []
        self._app_started_at = time.monotonic()
        self._monitor_started_at: float | None = None

        self.setWindowTitle(t("window.title", "FallGuard — Smart Safety"))
        self.resize(1300, 880)
        self.setMinimumSize(1060, 700)

        self._build_ui()
        self._on_theme_changed(self._theme.effective)
        self._start_timers()

    # ── Build ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(S8, S8, S8, S8)
        root.setSpacing(S8)

        # Sidebar
        self._sidebar = Sidebar(self._assets_dir)
        self._sidebar.settings_button.clicked.connect(self._open_settings)
        root.addWidget(self._sidebar)

        # Main panel (surface)
        self._main_panel = QFrame()
        self._main_panel.setObjectName("mainPanel")
        main_layout = QVBoxLayout(self._main_panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_topbar())

        # Scroll content
        scroll = QScrollArea()
        scroll.setObjectName("contentScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("contentWidget")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(S16, S16, S16, S16)
        layout.setSpacing(S16)

        top = QHBoxLayout()
        top.setSpacing(S16)
        top.addWidget(self._build_monitor_card(), 2)
        right = QVBoxLayout()
        right.setSpacing(S16)
        right.addWidget(self._build_current_risk_card())
        right.addWidget(self._build_time_card())
        top.addLayout(right, 1)
        layout.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(S16)
        middle.addWidget(self._build_risk_trend_card(), 1)
        middle.addWidget(self._build_detection_status_card(), 1)
        middle.addWidget(self._build_events_card(), 1)
        layout.addLayout(middle)

        layout.addWidget(self._build_metrics_card())

        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        root.addWidget(self._main_panel, 1)

    def _build_topbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(48)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(S16, 0, S16, 0)
        layout.setSpacing(S12)

        self._hamburger = QPushButton("☰")
        self._hamburger.setFixedSize(40, 40)
        self._hamburger.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hamburger.clicked.connect(self._sidebar.toggle)
        layout.addWidget(self._hamburger)

        self._conn_pill = ConnectionPill()
        layout.addWidget(self._conn_pill)

        self._mon_tag = MonitoringTag()
        layout.addWidget(self._mon_tag)

        layout.addStretch()
        return bar

    # ── Cards ─────────────────────────────────────────────────────

    def _build_stat_pair(self, label: str, value: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title = QLabel(label)
        title.setObjectName("mutedLabel")
        val = QLabel(value)
        val.setObjectName("statValue")
        layout.addWidget(title)
        layout.addWidget(val)
        return box

    def _build_monitor_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S16)

        header = QHBoxLayout()
        self._monitor_title = QLabel(t("monitor.title", "Live Monitor"))
        self._monitor_title.setObjectName("cardTitle")
        header.addWidget(self._monitor_title)
        header.addStretch()
        layout.addLayout(header)

        self._video = VideoShell()
        layout.addWidget(self._video, 1)

        info_row = QHBoxLayout()
        info_row.setSpacing(S24)
        self._monitor_fps_value = QLabel(t("common.na", "--"))
        self._monitor_resolution_value = QLabel(t("common.na", "--"))
        self._monitor_sensitivity_value = QLabel(self._settings_sensitivity_label())
        self._monitor_info_labels: dict[str, QLabel] = {}
        for key, label, widget in [
            ("fps", t("monitor.fpsLabel", "FPS"), self._monitor_fps_value),
            ("resolution", t("monitor.resolutionLabel", "Resolution"), self._monitor_resolution_value),
            ("sensitivity", t("settings.sensitivity", "Sensitivity"), self._monitor_sensitivity_value),
        ]:
            item = QWidget()
            item_layout = QVBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(2)
            item_label = QLabel(label)
            item_label.setObjectName("mutedLabel")
            widget.setObjectName("statValue")
            item_layout.addWidget(item_label)
            item_layout.addWidget(widget)
            info_row.addWidget(item, 1)
            self._monitor_info_labels[key] = item_label
        layout.addLayout(info_row)

        self._start_btn = QPushButton(t("buttons.startMonitoring", "Start Monitoring"))
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self._start_monitoring)

        self._import_btn = QPushButton(t("buttons.importMedia", "Import Media"))
        self._import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_menu = QMenu(self._import_btn)
        self._import_files_action = QAction(t("media.selectFiles", "Import Media"), self)
        self._import_files_action.triggered.connect(self._import_media_files)
        self._import_folder_action = QAction(t("media.selectFolder", "Import Photo Folder"), self)
        self._import_folder_action.triggered.connect(self._import_media_folder)
        self._import_menu.addAction(self._import_files_action)
        self._import_menu.addAction(self._import_folder_action)
        self._import_btn.setMenu(self._import_menu)

        self._stop_btn = QPushButton(t("buttons.stop", "Stop"))
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_monitoring)
        actions = QHBoxLayout()
        actions.setSpacing(S12)
        actions.addWidget(self._start_btn)
        actions.addWidget(self._import_btn)
        actions.addWidget(self._stop_btn)
        layout.addLayout(actions)

        return card

    def _build_current_risk_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S16)

        header = QHBoxLayout()
        self._current_risk_title = QLabel(t("risk.currentRisk", "Current Risk"))
        self._current_risk_title.setObjectName("cardTitle")
        header.addWidget(self._current_risk_title)
        header.addStretch()
        self._risk_badge = QLabel(t("risk.badgeNormal", "Normal"))
        self._risk_badge.setFixedHeight(28)
        self._risk_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self._risk_badge)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(S16)

        self._risk_ring = RiskRing()
        body.addWidget(self._risk_ring)

        info_w = QWidget()
        info_layout = QVBoxLayout(info_w)
        info_layout.setContentsMargins(0, S8, 0, S8)
        info_layout.setSpacing(S12)

        self._risk_info = {}
        self._risk_labels = {}
        for key, label_text in [
            ("riskLevel", t("risk.riskLevel", "Risk Level")),
            ("confidence", t("risk.confidence", "Confidence")),
        ]:
            row = QHBoxLayout()
            row.setSpacing(S8)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 13px; color: #374151; border: none;")
            row.addWidget(lbl)
            row.addStretch()
            val = QLabel(t("common.na", "--"))
            val.setStyleSheet("font-size: 13px; font-weight: 600; color: #137333; border: none;")
            row.addWidget(val)
            info_layout.addLayout(row)
            self._risk_info[key] = val
            self._risk_labels[key] = lbl

        body.addWidget(info_w, 1)
        layout.addLayout(body)
        self._current_chart = RiskChart()
        self._current_chart.setMaximumHeight(68)
        self._current_chart.setMinimumSize(220, 68)
        layout.addWidget(self._current_chart)
        return card

    def _build_time_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S12)
        self._time_title = QLabel(t("dashboard.currentTime", "Current Time"))
        self._time_title.setObjectName("cardTitle")
        layout.addWidget(self._time_title)
        layout.addStretch()
        self._time_value = QLabel("--:--:--")
        self._time_value.setObjectName("timeValue")
        self._time_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._date_value = QLabel("--")
        self._date_value.setObjectName("mutedLabel")
        self._date_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timezone_value = QLabel("--")
        self._timezone_value.setObjectName("mutedLabel")
        self._timezone_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._time_value)
        layout.addWidget(self._date_value)
        layout.addWidget(self._timezone_value)
        layout.addStretch()
        return card

    def _build_risk_trend_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S16)
        header = QHBoxLayout()
        self._trend_title = QLabel(t("risk.trendTitle", "Risk Trend"))
        self._trend_title.setObjectName("cardTitle")
        header.addWidget(self._trend_title)
        header.addStretch()
        self._trend_window = QLabel(t("chart.subtitle", "Last 60 seconds"))
        self._trend_window.setObjectName("mutedLabel")
        header.addWidget(self._trend_window)
        layout.addLayout(header)
        self._trend_chart = RiskChart()
        layout.addWidget(self._trend_chart)
        self._trend_level = QLabel(t("risk.lowRisk", "Low Risk"))
        self._trend_level.setObjectName("pillLabel")
        layout.addWidget(self._trend_level)
        return card

    def _build_detection_status_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S16)

        self._detection_title = QLabel(t("dashboard.detectionStatus", "Detection Status"))
        self._detection_title.setObjectName("cardTitle")
        layout.addWidget(self._detection_title)

        self._detection_badge = QLabel(t("status.systemWaiting", "System is waiting to start."))
        self._detection_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detection_badge.setObjectName("pillLabel")
        layout.addWidget(self._detection_badge)

        self._detection_detail = QLabel("")
        self._detection_detail.setWordWrap(True)
        self._detection_detail.setObjectName("mutedLabel")
        self._detection_detail.hide()
        layout.addWidget(self._detection_detail)

        self._status_labels: dict[str, QLabel] = {}
        for label_text, key in [
            (t("status.poseDetection", "Pose Detection"), "pose"),
            (t("status.aiModel", "AI Model"), "model"),
            (t("status.camera", "Camera"), "camera"),
            (t("status.processingState", "Processing State"), "processing"),
            (t("risk.confidence", "Confidence"), "confidence"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(S8)
            name = QLabel(label_text)
            name.setObjectName("mutedLabel")
            row.addWidget(name)
            row.addStretch()
            value = QLabel(t("common.na", "--"))
            value.setObjectName("statValue")
            row.addWidget(value)
            layout.addLayout(row)
            self._status_labels[key] = name
            if key == "camera":
                self._cam_status = value
            elif key == "model":
                self._model_status = value
            elif key == "pose":
                self._pose_status = value
            elif key == "processing":
                self._processing_status = value
            elif key == "confidence":
                self._confidence_status = value

        self._repair_btn = QPushButton(t("buttons.repairCamera", "Repair Camera Access"))
        self._repair_btn.clicked.connect(self._repair_camera)
        self._repair_btn.hide()
        layout.addWidget(self._repair_btn)
        layout.addStretch()
        return card

    def _build_events_card(self) -> QFrame:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(S20, S20, S20, S20)
        layout.setSpacing(S12)

        self._act_title = QLabel(t("activity.title", "Recent Events"))
        self._act_title.setObjectName("cardTitle")
        layout.addWidget(self._act_title)

        self._act_container = QVBoxLayout()
        self._act_container.setSpacing(0)
        layout.addLayout(self._act_container)

        self._act_empty = QLabel(t("activity.empty", "No activity yet"))
        self._act_empty.setStyleSheet("font-size: 13px; padding: 16px 0; border: none;")
        layout.addWidget(self._act_empty)
        return card

    def _build_metrics_card(self) -> QFrame:
        card = Card()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(S24, S20, S24, S20)
        layout.setSpacing(S24)

        self._metric_labels: dict[str, QLabel] = {}
        self._metric_values: dict[str, QLabel] = {}
        for key, label_text in [
            ("monitoringTime", t("metrics.monitoringTime", "Monitoring Time")),
            ("totalAlerts", t("metrics.totalAlerts", "Total Alerts")),
            ("highRisk", t("metrics.highRiskEvents", "High Risk Events")),
            ("avgRisk", t("metrics.avgRiskScore", "Avg. Risk Score")),
            ("uptime", t("metrics.systemUptime", "System Uptime")),
        ]:
            item = QWidget()
            item_layout = QVBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(4)
            label = QLabel(label_text)
            label.setObjectName("mutedLabel")
            value = QLabel("--")
            value.setObjectName("metricValue")
            item_layout.addWidget(label)
            item_layout.addWidget(value)
            layout.addWidget(item, 1)
            self._metric_labels[key] = label
            self._metric_values[key] = value

        return card

    # ── Theme ─────────────────────────────────────────────────────

    def _on_theme_changed(self, effective: str) -> None:
        c = DARK if effective == "dark" else LIGHT
        self._theme_colors = c

        self.setStyleSheet(build_stylesheet(c))

        # Body bg
        cw = self.findChild(QWidget, "centralWidget")
        if cw:
            cw.setStyleSheet(f"background: {c['surface_dim']};")

        # Main panel
        self._main_panel.setStyleSheet(
            f"#mainPanel {{ background: {c['surface_bright']}; border-radius: 20px; border: 1px solid {c['outline_variant']}; }}"
        )

        # Sidebar — clean white with right border
        self._sidebar.setStyleSheet(
            f"#sidebar {{ background: {c['surface_bright']}; border-right: 1px solid {c['outline_variant']}; border-radius: 20px; }}"
        )
        self._sidebar.update_theme_colors(c)

        # Topbar
        for w in self.findChildren(QFrame, "topbar"):
            w.setStyleSheet(f"#topbar {{ background: transparent; }}")

        # Content
        for w in self.findChildren(QWidget, "contentWidget"):
            w.setStyleSheet("background: transparent;")

        # Card titles
        style = f"font-size: 16px; font-weight: 600; color: {c['on_surface']}; border: none;"
        for w in self.findChildren(QLabel, "cardTitle"):
            w.setStyleSheet(style)
        for w in self.findChildren(QLabel, "mutedLabel"):
            w.setStyleSheet(f"font-size: 12px; color: {c['on_surface_secondary']}; border: none;")
        for w in self.findChildren(QLabel, "statValue"):
            w.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {c['on_surface']}; border: none;")
        for w in self.findChildren(QLabel, "timeValue"):
            w.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {c['on_surface']}; border: none;")
        for w in self.findChildren(QLabel, "metricValue"):
            w.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {c['on_surface']}; border: none;")
        for w in self.findChildren(QLabel, "pillLabel"):
            w.setStyleSheet(
                f"background: {c['primary_container']}; color: {c['success_fg']}; "
                f"font-size: 12px; font-weight: 700; border-radius: 14px; "
                f"padding: 6px 12px; border: none;"
            )

        # Hamburger
        self._hamburger.setStyleSheet(
            f"QPushButton {{ font-size: 20px; border: none; border-radius: 12px; "
            f"color: {c['on_surface']}; background: transparent; }}"
            f"QPushButton:hover {{ background: {c['surface_container_high']}; }}"
        )

        # Buttons — direct inline styles, no property selectors
        self._start_btn.setStyleSheet(
            f"QPushButton {{ background: {c['primary']}; color: {c['on_primary']}; "
            f"border: none; border-radius: 14px; padding: 0 24px; "
            f"font-size: 14px; font-weight: 600; min-height: 48px; }}"
            f"QPushButton:hover {{ background: {c['primary_hover']}; }}"
            f"QPushButton:pressed {{ background: {c['primary_pressed']}; }}"
            f"QPushButton:disabled {{ background: {c['disabled_bg']}; color: {c['disabled_fg']}; }}"
        )
        self._import_btn.setStyleSheet(
            f"QPushButton {{ background: {c['surface_bright']}; color: {c['primary']}; "
            f"border: 1px solid #DADCE0; border-radius: 14px; "
            f"padding: 0 24px; font-size: 14px; font-weight: 600; min-height: 48px; }}"
            f"QPushButton:hover {{ background: {c['surface_dim']}; border-color: {c['primary']}; }}"
            f"QPushButton:disabled {{ color: #9AA0A6; border-color: {c['outline_variant']}; }}"
        )
        self._stop_btn.setStyleSheet(
            f"QPushButton {{ background: #F1F3F4; color: #9AA0A6; "
            f"border: 1px solid #E0E3EB; border-radius: 14px; "
            f"padding: 0 24px; font-size: 14px; font-weight: 600; min-height: 48px; }}"
            f"QPushButton:disabled {{ background: #F1F3F4; color: #9AA0A6; "
            f"border: 1px solid #E0E3EB; }}"
            f"QPushButton:enabled {{ background: {c['surface_bright']}; color: {c['error']}; "
            f"border: 1px solid #FAD2CF; }}"
            f"QPushButton:enabled:hover {{ background: {c['error_container']}; }}"
        )
        self._repair_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {c['error']}; "
            f"border: 1px solid #FAD2CF; border-radius: 14px; "
            f"padding: 0 24px; font-size: 14px; font-weight: 600; min-height: 48px; }}"
            f"QPushButton:hover {{ background: {c['error_container']}; }}"
        )
        self._import_menu.setStyleSheet(
            f"QMenu {{ background: {c['surface_bright']}; color: {c['on_surface']}; "
            f"border: 1px solid {c['outline_variant']}; border-radius: 12px; padding: 6px; }}"
            f"QMenu::item {{ padding: 8px 24px 8px 12px; border-radius: 8px; }}"
            f"QMenu::item:selected {{ background: {c['primary_container']}; color: {c['on_primary_container']}; }}"
        )

        # Divider
        for w in self.findChildren(QFrame, "statusDivider"):
            w.setStyleSheet(f"background: {c['outline_variant']}; border: none;")

        # Charts
        self._current_chart.set_theme_colors(c)
        self._trend_chart.set_theme_colors(c)

        # Empty state labels
        empty_style = f"color: {c['on_surface_secondary']}; font-size: 13px; padding: 16px 0; border: none;"
        self._act_empty.setStyleSheet(empty_style)

    # ── Timers ─────────────────────────────────────────────────────

    def _start_timers(self) -> None:
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(700)

        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._refresh_frame)
        self._frame_timer.start(67)

    # ── Status polling ────────────────────────────────────────────

    def _poll_status(self) -> None:
        try:
            snap = self._monitor.snapshot()
            snap["mediaJob"] = self._media_processor.snapshot()
            active = self._profile_manager.active
            if active:
                snap["activeProfile"] = active.to_dict()
            self._update_ui(snap)
        except Exception as exc:
            # Surface polling errors so they are visible instead of silently
            # keeping the UI stuck at the initial placeholder text.
            import traceback
            traceback.print_exc()
            try:
                self._detection_badge.setText(f"Polling error: {exc}")
                self._detection_badge.setStyleSheet(
                    f"background: #FEE2E2; color: #B91C1C; font-size: 12px; font-weight: 700; "
                    f"border: 1px solid #FAD2CF; border-radius: 16px; padding: 8px 12px;"
                )
            except Exception:
                pass

    def _update_ui(self, snap: dict) -> None:
        running = snap.get("running", False)
        loading = snap.get("loading", False)
        connected = snap.get("cameraConnected", False)
        state = str(snap.get("state", "Idle"))
        risk = int(snap.get("riskPercent", 0))
        media_job = snap.get("mediaJob", {})
        media_running = bool(media_job.get("running")) if isinstance(media_job, dict) else False
        c = self._theme_colors

        # Buttons
        self._start_btn.setEnabled(not running and not loading and not media_running)
        self._import_btn.setEnabled(not running and not loading and not media_running)
        self._stop_btn.setEnabled(running or loading)

        # Connection pill
        self._conn_pill.set_connected(connected)
        self._conn_pill.set_text(
            t("monitoring.cameraConnected", "Camera Connected") if connected
            else t("monitoring.cameraReady", "Camera Ready")
        )

        app_state = self._display_state(state, running, loading, media_running)
        if app_state == "Danger":
            self._mon_tag.set_state(t("status.danger", "Danger"), c["danger_bg"], c["danger_fg"], "#FAD2CF")
        elif app_state == "Warning":
            self._mon_tag.set_state(t("status.warning", "Warning"), c["warning_bg"], c["warning_fg"], "#FDE293")
        elif app_state == "Monitoring":
            self._mon_tag.set_state(t("monitoring.monitoring", "Monitoring"), c["success_bg"], c["success_fg"], "#CEEAD6")
        elif app_state == "Starting":
            self._mon_tag.set_state(t("monitoring.starting", "Starting"), c["warning_bg"], c["warning_fg"], "#FDE293")
        elif app_state == "Ready":
            self._mon_tag.set_state(t("status.ready", "Ready"), c["success_bg"], c["success_fg"], "#CEEAD6")
        else:
            self._mon_tag.set_state(t("monitoring.idle", "Idle"), c["idle_bg"], c["idle_fg"], "#E0E3EB")

        detail_lower = f"{snap.get('title','')} {snap.get('detail','')} {snap.get('error','')}".lower()
        self._repair_btn.setVisible(state == "Error" and "camera" in detail_lower)

        status_color = c["success_fg"]
        if app_state == "Warning":
            status_color = c["warning_fg"]
        elif app_state == "Danger":
            status_color = c["danger_fg"]

        self._cam_status.setText(
            t("status.connected", "Connected") if connected
            else t("status.ready", "Ready") if not running
            else t("monitoring.disconnected", "Disconnected")
        )
        self._cam_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {status_color}; border: none;")
        model = snap.get("modelActive", False)
        self._model_status.setText(
            t("status.active", "Active") if model
            else t("status.loading", "Loading") if loading
            else t("status.ready", "Ready")
        )
        self._model_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {status_color}; border: none;")
        self._pose_status.setText(
            t("status.active", "Active") if running and state != "Unknown"
            else t("status.waiting", "Waiting") if not running
            else t("status.needsBetterView", "Needs Better View")
        )
        self._pose_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {status_color}; border: none;")
        self._processing_status.setText(t(f"status.{app_state.lower()}", app_state))
        self._processing_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {status_color}; border: none;")
        confidence = snap.get("confidencePercent", None)
        conf_text = t("common.na", "--") if confidence in (None, "--") else f"{confidence}{t('common.percent', '%')}"
        self._confidence_status.setText(conf_text)
        self._confidence_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {status_color}; border: none;")
        self._set_status_badge(app_state, status_color)

        # Show error detail when something went wrong
        if state == "Error":
            detail = snap.get("detail", "") or snap.get("error", "")
            self._detection_detail.setText(detail)
            self._detection_detail.show()
        else:
            self._detection_detail.hide()

        # Risk
        self._update_risk(risk, state, snap)
        self._update_time_card()
        self._update_metrics(snap, risk, app_state)

        # Activities
        if isinstance(media_job, dict):
            self._track_media_activity(media_job)
        self._render_activities(list(snap.get("activities", [])) + self._media_activities)

        # Video
        fps = float(snap.get("fps", 0) or 0)
        resolution = str(snap.get("resolution", t("common.na", "--")))
        self._video.set_info(fps, resolution)
        self._monitor_fps_value.setText(f"{fps:.1f}" if fps > 0 else t("common.na", "--"))
        self._monitor_resolution_value.setText(resolution if resolution and resolution != "--" else t("common.na", "--"))
        self._monitor_sensitivity_value.setText(self._settings_sensitivity_label())

        # Frame refresh
        if connected:
            self._frame_timer.start()
        elif not running and not loading:
            self._frame_timer.stop()
            self._video.show_placeholder()

    def _update_risk(self, risk: int, state: str, snap: dict) -> None:
        self._risk_history.append(risk)
        if len(self._risk_history) > 48:
            self._risk_history = self._risk_history[-48:]

        if risk >= 65 or state == "Fall":
            color, badge = "#EF4444", t("risk.badgeTakeCare", "Critical")
            badge_bg, badge_fg = "#FEE2E2", "#B91C1C"
        elif risk >= 35 or state == "Pre-fall":
            color, badge = "#F59E0B", t("risk.badgeWatch", "Warning")
            badge_bg, badge_fg = "#FEF3C7", "#B45309"
        else:
            color, badge = "#22C55E", t("risk.badgeNormal", "Normal")
            badge_bg, badge_fg = "#EAF7EF", "#16A34A"

        self._risk_ring.set_risk(risk, color, self._theme_colors.get("ring_track", "#E2E4E8"))

        self._risk_badge.setText(badge)
        self._risk_badge.setStyleSheet(
            f"background: {badge_bg}; color: {badge_fg}; font-size: 13px; font-weight: 600; "
            f"border-radius: 14px; padding: 6px 14px; border: none;"
        )

        levels = {"Fall": t("risk.levelHigh", "High"), "Pre-fall": t("risk.levelMedium", "Medium")}
        level_text = levels.get(state, t("risk.levelLow", "Low"))
        self._risk_info["riskLevel"].setText(level_text)
        self._risk_info["riskLevel"].setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {badge_fg}; border: none;"
        )
        confidence = snap.get("confidencePercent", None)
        self._risk_info["confidence"].setText(
            t("common.na", "--") if confidence in (None, "--") else f"{confidence}{t('common.percent', '%')}"
        )
        self._risk_info["confidence"].setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {self._theme_colors['on_surface']}; border: none;"
        )
        self._trend_level.setText(
            t("risk.highRisk", "High Risk") if risk >= 65 or state == "Fall"
            else t("risk.mediumRisk", "Medium Risk") if risk >= 35 or state == "Pre-fall"
            else t("risk.lowRisk", "Low Risk")
        )
        self._trend_level.setStyleSheet(
            f"background: {badge_bg}; color: {badge_fg}; font-size: 12px; font-weight: 700; "
            f"border-radius: 14px; padding: 6px 12px; border: none;"
        )

        self._current_chart.update_data(self._risk_history, color)
        self._trend_chart.update_data(self._risk_history, color)

    def _settings_sensitivity_label(self) -> str:
        levels = {
            "low": t("settings.sensitivityLow", "Low"),
            "medium": t("settings.sensitivityMedium", "Medium"),
            "high": t("settings.sensitivityHigh", "High"),
        }
        return levels.get(str(getattr(self._app_settings, "sensitivity", "medium")), t("settings.sensitivityMedium", "Medium"))

    def _display_state(self, state: str, running: bool, loading: bool, media_running: bool) -> str:
        if state == "Fall":
            return "Danger"
        if state in {"Pre-fall", "Error"}:
            return "Warning" if state == "Pre-fall" else "Danger"
        if running or media_running:
            return "Monitoring"
        if loading:
            return "Starting"
        return "Ready"

    def _set_status_badge(self, app_state: str, fg: str) -> None:
        c = self._theme_colors
        labels = {
            "Idle": t("monitoring.idle", "Idle"),
            "Ready": t("status.ready", "Ready"),
            "Starting": t("monitoring.starting", "Starting"),
            "Monitoring": t("monitoring.monitoring", "Monitoring"),
            "Warning": t("status.warning", "Warning"),
            "Danger": t("status.danger", "Danger"),
        }
        bg = c["primary_container"]
        border = c["secondary_container"]
        if app_state == "Warning":
            bg, border = c["warning_bg"], "#FDE293"
        elif app_state == "Danger":
            bg, border = c["danger_bg"], "#FAD2CF"
        self._detection_badge.setText(labels.get(app_state, app_state))
        self._detection_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; font-size: 12px; font-weight: 700; "
            f"border: 1px solid {border}; border-radius: 16px; padding: 8px 12px;"
        )

    def _update_time_card(self) -> None:
        now = datetime.now().astimezone()
        self._time_value.setText(now.strftime("%H:%M:%S"))
        self._date_value.setText(now.strftime("%Y-%m-%d"))
        self._timezone_value.setText(now.tzname() or now.strftime("%z"))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _update_metrics(self, snap: dict, risk: int, app_state: str) -> None:
        running = bool(snap.get("running", False) or snap.get("loading", False))
        if running and self._monitor_started_at is None:
            self._monitor_started_at = time.monotonic()
        elif not running and app_state not in {"Warning", "Danger"}:
            self._monitor_started_at = None

        monitor_seconds = 0.0
        if self._monitor_started_at is not None:
            monitor_seconds = time.monotonic() - self._monitor_started_at

        activities = list(snap.get("activities", []))
        active = snap.get("activeProfile") if isinstance(snap.get("activeProfile"), dict) else {}
        fall_events = list(active.get("fallEvents", [])) if isinstance(active, dict) else []
        alert_count = len(fall_events) + sum(1 for item in activities if item.get("level") in {"warning", "danger"})
        high_count = sum(1 for item in activities if item.get("level") == "danger")
        high_count += sum(1 for item in fall_events if item.get("state") == "Fall")
        avg = round(sum(self._risk_history) / len(self._risk_history)) if self._risk_history else risk

        self._metric_values["monitoringTime"].setText(self._format_duration(monitor_seconds))
        self._metric_values["totalAlerts"].setText(str(alert_count))
        self._metric_values["highRisk"].setText(str(high_count))
        self._metric_values["avgRisk"].setText(f"{avg}{t('common.percent', '%')}")
        self._metric_values["uptime"].setText(self._format_duration(time.monotonic() - self._app_started_at))

    def _render_activities(self, items: list[dict]) -> None:
        while self._act_container.count():
            w = self._act_container.takeAt(0).widget()
            if w: w.deleteLater()
        if not items:
            self._act_empty.show()
        else:
            self._act_empty.hide()
            for item in list(reversed(items))[-12:]:
                self._act_container.addWidget(ActivityRow(
                    level=item.get("level", "muted"),
                    title=t_dynamic(str(item.get("title", ""))),
                    time_str=str(item.get("time", t("common.na", "--"))),
                    risk=int(item.get("risk", 0)),
                ))

    # ── Frame refresh ──────────────────────────────────────────────

    def _refresh_frame(self) -> None:
        frame = self._monitor.jpeg_frame()
        if frame is None:
            return
        pix = QPixmap()
        pix.loadFromData(frame)
        if not pix.isNull():
            self._video.set_frame(pix)

    # ── Actions ────────────────────────────────────────────────────

    def _start_monitoring(self) -> None:
        import threading
        self._risk_history = []
        threading.Thread(target=self._monitor.start, daemon=True).start()

    def _stop_monitoring(self) -> None:
        self._monitor.stop()
        self._frame_timer.stop()
        self._video.show_placeholder()

    def _import_media_files(self) -> None:
        import threading
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            t("media.selectFiles", "Import Media"),
            "",
            t("media.fileFilter", "Media (*.mp4 *.mov *.m4v *.avi *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tiff *.heic *.heif);;All (*)"),
        )
        if not paths:
            return
        threading.Thread(target=self._run_media_import, args=([Path(p) for p in paths],), daemon=True).start()

    def _import_media_folder(self) -> None:
        import threading
        folder = QFileDialog.getExistingDirectory(
            self,
            t("media.selectFolder", "Import Photo Folder"),
            "",
        )
        if not folder:
            return
        threading.Thread(target=self._run_media_import, args=([Path(folder)],), daemon=True).start()

    def _run_media_import(self, paths: list[Path]) -> None:
        try:
            self._media_processor.start_from_paths(paths)
        except Exception as exc:
            message = str(exc)
            QTimer.singleShot(0, lambda: self._show_error(message))

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, t("media.importFailed", "Import Failed"), message)

    def _track_media_activity(self, media_job: dict) -> None:
        state = str(media_job.get("state", ""))
        title = str(media_job.get("title", ""))
        key = f"{state}:{title}"
        if state not in {"Processing", "Complete", "Error"} or key == self._last_media_state:
            return
        self._last_media_state = key
        level = "danger" if state == "Error" else "normal" if state == "Complete" else "warning"
        self._media_activities.append({
            "level": level,
            "title": title or state,
            "time": str(media_job.get("finishedAt") or media_job.get("startedAt") or t("common.na", "--")),
            "risk": 0,
        })
        self._media_activities = self._media_activities[-4:]

    def _repair_camera(self) -> None:
        import shutil
        tcc = shutil.which("tccutil")
        if tcc:
            subprocess.run([tcc, "reset", "Camera", "com.fallguard.desktop"], check=False)
        subprocess.run(
            ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"],
            check=False,
        )

    def _open_settings(self) -> None:
        dlg = SettingsDialog(
            settings=self._app_settings,
            profile_manager=self._profile_manager,
            theme_manager=self._theme,
            app_version=self._app_version,
            parent=self,
        )
        dlg.theme_changed.connect(lambda m: self._on_theme_changed(self._theme.effective))
        dlg.language_changed.connect(lambda lang: self._on_language_changed(lang))
        dlg.exec()

    def _on_language_changed(self, lang: str) -> None:
        """Refresh all static UI labels after language switch."""
        self.setWindowTitle(t("window.title", "FallGuard — Smart Safety"))

        # Monitor card
        self._monitor_title.setText(t("monitor.title", "Live Monitor"))
        self._monitor_info_labels["fps"].setText(t("monitor.fpsLabel", "FPS"))
        self._monitor_info_labels["resolution"].setText(t("monitor.resolutionLabel", "Resolution"))
        self._monitor_info_labels["sensitivity"].setText(t("settings.sensitivity", "Sensitivity"))
        self._monitor_sensitivity_value.setText(self._settings_sensitivity_label())
        self._start_btn.setText(t("buttons.startMonitoring", "Start Monitoring"))
        self._import_btn.setText(t("buttons.importMedia", "Import Media"))
        self._import_files_action.setText(t("media.selectFiles", "Import Media"))
        self._import_folder_action.setText(t("media.selectFolder", "Import Photo Folder"))
        self._stop_btn.setText(t("buttons.stop", "Stop"))

        # Dashboard cards
        self._current_risk_title.setText(t("risk.currentRisk", "Current Risk"))
        self._risk_labels["riskLevel"].setText(t("risk.riskLevel", "Risk Level"))
        self._risk_labels["confidence"].setText(t("risk.confidence", "Confidence"))
        self._time_title.setText(t("dashboard.currentTime", "Current Time"))
        self._trend_title.setText(t("risk.trendTitle", "Risk Trend"))
        self._trend_window.setText(t("chart.subtitle", "Last 60 seconds"))
        self._detection_title.setText(t("dashboard.detectionStatus", "Detection Status"))
        self._act_title.setText(t("activity.title", "Recent Events"))
        self._act_empty.setText(t("activity.empty", "No activity yet"))

        # Status card labels
        if "pose" in self._status_labels:
            self._status_labels["pose"].setText(t("status.poseDetection", "Pose Detection"))
        if "camera" in self._status_labels:
            self._status_labels["camera"].setText(t("status.camera", "Camera"))
        if "model" in self._status_labels:
            self._status_labels["model"].setText(t("status.aiModel", "AI Model"))
        if "processing" in self._status_labels:
            self._status_labels["processing"].setText(t("status.processingState", "Processing State"))
        if "confidence" in self._status_labels:
            self._status_labels["confidence"].setText(t("risk.confidence", "Confidence"))
        self._repair_btn.setText(t("buttons.repairCamera", "Repair Camera Access"))

        for key, label in [
            ("monitoringTime", t("metrics.monitoringTime", "Monitoring Time")),
            ("totalAlerts", t("metrics.totalAlerts", "Total Alerts")),
            ("highRisk", t("metrics.highRiskEvents", "High Risk Events")),
            ("avgRisk", t("metrics.avgRiskScore", "Avg. Risk Score")),
            ("uptime", t("metrics.systemUptime", "System Uptime")),
        ]:
            self._metric_labels[key].setText(label)

        # Sidebar labels
        self._sidebar.refresh_labels()

        # Re-trigger theme update to fix stylesheet-based text
        self._on_theme_changed(self._theme.effective)

    def closeEvent(self, event) -> None:
        self._monitor.stop()
        if self._status_timer: self._status_timer.stop()
        if self._frame_timer: self._frame_timer.stop()
        super().closeEvent(event)
