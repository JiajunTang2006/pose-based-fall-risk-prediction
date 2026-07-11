"""
Settings dialog — Material Design 3 layout.
Sidebar nav + content panel. Full i18n coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QWidget, QFrame, QStackedWidget, QScrollArea, QCheckBox,
    QFileDialog, QMessageBox,
)

# Relative imports work when running as `python -m fall_prediction_desktop`.
# Absolute imports work inside a PyInstaller bundle where relative imports fail.
try:
    from .theme import ThemeManager
    from .i18n import get_i18n, t
except ImportError:
    from fall_prediction_desktop.ui.theme import ThemeManager  # type: ignore[no-redef]
    from fall_prediction_desktop.ui.i18n import get_i18n, t  # type: ignore[no-redef]

# ── M3 color tokens for settings dialog ──────────────────────────────

M3_LIGHT = {
    "bg":             "#FFFFFF",
    "sidebar_bg":     "#F8FAFC",
    "panel_bg":       "#FFFFFF",
    "active_bg":      "#EAF7EF",
    "active_bar":     "#22C55E",
    "hover_bg":       "#F3F6FB",
    "border":         "#E5E7EB",
    "input_bg":       "#FFFFFF",
    "input_border":   "#E5E7EB",
    "focus_border":   "#22C55E",
    "text":           "#111827",
    "text_secondary": "#6B7280",
    "text_muted":     "#9CA3AF",
    "accent":         "#22C55E",
    "accent_hover":   "#16A34A",
    "search_bg":      "#F8FAFC",
    "section_line":   "#E5E7EB",
    "badge_bg":       "#22C55E",
    "badge_fg":       "#FFFFFF",
    "delete_hover":   "#EF4444",
}

M3_DARK = {
    "bg":             "#101828",
    "sidebar_bg":     "#0F172A",
    "panel_bg":       "#101828",
    "active_bg":      "rgba(34, 197, 94, 0.18)",
    "active_bar":     "#4ADE80",
    "hover_bg":       "#1A2740",
    "border":         "#263244",
    "input_bg":       "#172033",
    "input_border":   "#334155",
    "focus_border":   "#4ADE80",
    "text":           "#F9FAFB",
    "text_secondary": "#CBD5E1",
    "text_muted":     "#94A3B8",
    "accent":         "#4ADE80",
    "accent_hover":   "#86EFAC",
    "search_bg":      "#0F172A",
    "section_line":   "#263244",
    "badge_bg":       "#4ADE80",
    "badge_fg":       "#052E16",
    "delete_hover":   "#F28B82",
}


class SettingsDialog(QDialog):
    """M3-style settings dialog with sidebar nav + content panel."""

    theme_changed = Signal(str)
    language_changed = Signal(str)

    def __init__(self, settings, profile_manager, theme_manager: ThemeManager,
                 app_version: str = "0.2.0", parent=None, repos=None,
                 media_root: Path | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._profile_manager = profile_manager
        self._theme_manager = theme_manager
        self._app_version = app_version
        self._repos = repos
        self._media_root = media_root
        self._i18n = get_i18n()

        self._m3 = M3_DARK if theme_manager.effective == "dark" else M3_LIGHT

        self.setWindowTitle(t("settings.title", "Settings"))
        self.setMinimumSize(780, 540)
        self.resize(820, 580)
        self.setModal(True)

        self._build_ui()
        self._apply_m3_style()
        self._load()

    # ── Build ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Body: sidebar + content
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(self._build_sidebar())

        self._stack = QStackedWidget()
        self._stack.addWidget(self._page_general())
        self._stack.addWidget(self._page_detection())
        self._stack.addWidget(self._page_alerts())
        self._stack.addWidget(self._page_data_management())
        self._stack.addWidget(self._page_about())
        body.addWidget(self._stack, 1)

        root.addLayout(body, 1)

    def _build_sidebar(self) -> QFrame:
        nav = QFrame()
        nav.setObjectName("sidebarNav")
        nav.setFixedWidth(220)

        layout = QVBoxLayout(nav)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)

        self._nav_list = QListWidget()
        self._nav_list.setObjectName("navList")
        self._nav_list.setSpacing(0)
        self._nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._nav_items: list[tuple[str, str, QListWidgetItem]] = []
        for key, i18n_key in [
            ("general",   "settings.general"),
            ("detection", "settings.detection"),
            ("alerts",    "settings.alerts"),
            ("data",      "settings.dataManagement"),
            ("about",     "settings.about"),
        ]:
            label = t(i18n_key, key.title())
            item = QListWidgetItem(f"  {label}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._nav_list.addItem(item)
            self._nav_items.append((key, label, item))

        self._nav_list.itemClicked.connect(self._on_nav_clicked)
        layout.addWidget(self._nav_list, 1)

        self._set_active_nav(0)
        return nav

    # ── Content pages ─────────────────────────────────────────────

    def _page_general(self) -> QWidget:
        self._lang_combo = self._make_combo([
            (t("common.english", "English"), "en"),
            (t("common.chinese", "中文"), "zh"),
        ])
        self._theme_combo = self._make_combo([
            (t("settings.themeSystem", "System"), "system"),
            (t("settings.themeLight", "Light"), "light"),
            (t("settings.themeDark", "Dark"), "dark"),
        ])
        return self._content_page(t("settings.general", "General"), [
            self._setting_row_combo(
                t("settings.language", "Language"),
                t("settings.languageDesc", "Choose the display language"),
                self._lang_combo, self._on_lang,
            ),
            self._setting_row_combo(
                t("settings.theme", "Theme"),
                t("settings.themeDesc", "Select the color theme"),
                self._theme_combo, self._on_theme,
            ),
            self._setting_row_toggle(
                t("settings.startOnBoot", "Start on System Boot"),
                t("settings.startOnBootDesc", "Reserved for a future macOS login-item integration."),
                checked=False,
                enabled=False,
            ),
            self._setting_row_toggle(
                t("settings.minimizeToTray", "Minimize to System Tray"),
                t("settings.minimizeToTrayDesc", "Keep FallGuard available from the menu bar."),
                checked=self._settings.minimize_to_tray,
                callback=lambda checked: self._on_bool_setting("minimize_to_tray", checked),
            ),
        ])

    def _page_detection(self) -> QWidget:
        self._sens_combo = self._make_combo([
            (t("settings.sensitivityLow", "Low"), "low"),
            (t("settings.sensitivityMedium", "Medium"), "medium"),
            (t("settings.sensitivityHigh", "High"), "high"),
        ])
        return self._content_page(t("settings.detection", "Detection"), [
            self._setting_row_combo(
                t("settings.sensitivity", "Sensitivity"),
                t("settings.sensitivityDesc", "Fall detection sensitivity level"),
                self._sens_combo, self._on_sensitivity,
            ),
            self._setting_row_text(
                t("settings.thresholdSettings", "Threshold Settings"),
                t("settings.thresholdSettingsDesc", "Risk thresholds are provided by the backend predictor."),
            ),
            self._setting_row_text(
                t("settings.modelConfiguration", "Model Configuration"),
                t("settings.modelConfigurationDesc", "Current ML model and pose detector settings."),
            ),
        ])

    def _page_alerts(self) -> QWidget:
        return self._content_page(t("settings.alerts", "Alerts"), [
            self._setting_row_toggle(
                t("settings.soundAlert", "Sound Alert"),
                t("settings.soundAlertDesc", "Play a sound when risk becomes high."),
                checked=self._settings.sound_alert,
                callback=lambda checked: self._on_bool_setting("sound_alert", checked),
            ),
            self._setting_row_toggle(
                t("settings.popupAlert", "Popup Alert"),
                t("settings.popupAlertDesc", "Reserved for a future macOS notification integration."),
                checked=False,
                enabled=False,
            ),
            self._setting_row_toggle(
                t("settings.emailNotification", "Email / Notification"),
                t("settings.emailNotificationDesc", "Reserved for future alert delivery integrations."),
                checked=False,
                enabled=False,
            ),
        ])

    def _page_data_management(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QLabel(t("settings.dataManagement", "Data Management"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        desc = QLabel(t("settings.dataManagementDesc",
                       "Export logs, clear history, and manage datasets."))
        desc.setObjectName("sectionDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(12)
        layout.addWidget(self._setting_row_button(
            t("settings.exportLogs", "Export Logs"),
            t("settings.exportLogsDesc", "Export monitoring events and media analysis records."),
            t("settings.exportLogs", "Export Logs"),
            self._export_logs,
        ))
        layout.addWidget(self._row_separator())
        layout.addWidget(self._setting_row_button(
            t("settings.clearHistory", "Clear History"),
            t("settings.clearHistoryDesc", "Remove local event history for demo cleanup."),
            t("settings.clearHistory", "Clear History"),
            self._clear_history,
        ))
        layout.addWidget(self._row_separator())
        layout.addWidget(self._setting_row_button(
            t("settings.datasetManagement", "Dataset Management"),
            t("settings.datasetManagementDesc", "Manage imported photo and video analysis data."),
            t("settings.datasetManagement", "Dataset Management"),
            self._open_dataset_folder,
        ))

        profile_title = QLabel(t("settings.profiles", "Profiles"))
        profile_title.setObjectName("sectionTitle")
        layout.addSpacing(16)
        layout.addWidget(profile_title)

        self._profile_list = QListWidget()
        self._profile_list.setObjectName("profileList")
        self._profile_list.itemClicked.connect(self._on_profile_click)
        self._profile_list.setMinimumHeight(160)
        layout.addWidget(self._profile_list, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._profile_input = QLineEdit()
        self._profile_input.setPlaceholderText(
            t("settings.profileNamePlaceholder", "Profile name..."))
        self._profile_input.setMaxLength(20)
        self._profile_input.returnPressed.connect(self._add_profile)
        self._profile_input.setObjectName("profileInput")
        row.addWidget(self._profile_input, 1)

        add_btn = QPushButton(t("settings.addProfileBtn", "Add Profile"))
        add_btn.setObjectName("addProfileBtn")
        add_btn.setFixedWidth(110)
        add_btn.clicked.connect(self._add_profile)
        row.addWidget(add_btn)
        layout.addLayout(row)

        return page

    def _page_about(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)

        title = QLabel(t("settings.about", "About"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        name = QLabel(t("brand.name", "FallGuard"))
        name.setObjectName("aboutName")
        layout.addWidget(name)

        ver = QLabel(f"{t('settings.versionLabel', 'Version')} {self._app_version}")
        ver.setObjectName("aboutVersion")
        layout.addWidget(ver)

        desc = QLabel(t("settings.aboutDesc",
                       "AI-powered fall detection system."))
        desc.setObjectName("aboutDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()
        return page

    # ── Content helpers ───────────────────────────────────────────

    def _content_page(self, title: str, rows: list[QWidget]) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(0)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("sectionTitle")
        layout.addWidget(title_lbl)
        layout.addSpacing(20)

        for i, row_widget in enumerate(rows):
            if i > 0:
                sep = QFrame()
                sep.setFixedHeight(1)
                sep.setObjectName("rowSep")
                layout.addWidget(sep)
                layout.addSpacing(16)
            layout.addWidget(row_widget)
            layout.addSpacing(16)

        layout.addStretch()
        scroll.setWidget(page)
        return scroll

    def _setting_row_combo(self, label: str, description: str,
                           combo: QComboBox, handler) -> QWidget:
        row = QWidget()
        row.setObjectName("settingRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("settingLabel")
        left.addWidget(lbl)
        dsc = QLabel(description)
        dsc.setObjectName("settingDesc")
        dsc.setWordWrap(True)
        left.addWidget(dsc)
        layout.addLayout(left, 1)

        combo.setObjectName("settingCombo")
        combo.setFixedWidth(160)
        combo.currentIndexChanged.connect(handler)
        layout.addWidget(combo)

        return row

    def _setting_row_text(self, label: str, description: str) -> QWidget:
        row = QWidget()
        row.setObjectName("settingRow")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("settingLabel")
        layout.addWidget(lbl)
        dsc = QLabel(description)
        dsc.setObjectName("settingDesc")
        dsc.setWordWrap(True)
        layout.addWidget(dsc)
        return row

    def _setting_row_toggle(self, label: str, description: str,
                            checked: bool = False, enabled: bool = True,
                            callback=None) -> QWidget:
        row = QWidget()
        row.setObjectName("settingRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("settingLabel")
        left.addWidget(lbl)
        dsc = QLabel(description)
        dsc.setObjectName("settingDesc")
        dsc.setWordWrap(True)
        left.addWidget(dsc)
        layout.addLayout(left, 1)

        toggle = QCheckBox()
        toggle.setObjectName("settingToggle")
        toggle.setChecked(checked)
        toggle.setEnabled(enabled)
        if callback is not None:
            toggle.toggled.connect(callback)
        layout.addWidget(toggle)
        return row

    def _setting_row_button(self, label: str, description: str,
                            button_text: str, handler) -> QWidget:
        row = QWidget()
        row.setObjectName("settingRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("settingLabel")
        left.addWidget(lbl)
        dsc = QLabel(description)
        dsc.setObjectName("settingDesc")
        dsc.setWordWrap(True)
        left.addWidget(dsc)
        layout.addLayout(left, 1)

        btn = QPushButton(button_text)
        btn.setObjectName("settingsActionBtn")
        btn.clicked.connect(handler)
        layout.addWidget(btn)
        return row

    def _row_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setObjectName("rowSep")
        return sep

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _make_combo(options: list[tuple[str, str]]) -> QComboBox:
        cb = QComboBox()
        for label, data in options:
            cb.addItem(label, data)
        return cb

    def _set_active_nav(self, index: int) -> None:
        nav_keys_map = {
            "general": "settings.general",
            "detection": "settings.detection",
            "alerts": "settings.alerts",
            "data": "settings.dataManagement",
            "about": "settings.about",
        }
        for i, (key, label, item) in enumerate(self._nav_items):
            label = t(nav_keys_map.get(key, key), label)
            if i == index:
                item.setText(f"  {label}")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            else:
                item.setText(f"  {label}")
                font = item.font()
                font.setBold(False)
                item.setFont(font)
        self._nav_list.setCurrentRow(index)

    # ── M3 style application ─────────────────────────────────────

    def _apply_m3_style(self) -> None:
        c = self._m3

        self.setStyleSheet(f"""
            SettingsDialog {{
                background: {c["bg"]};
            }}

            #sidebarNav {{
                background: {c["sidebar_bg"]};
                border-right: 1px solid {c["border"]};
            }}
            #navList {{
                background: transparent;
                border: none;
                outline: none;
                font-size: 14px;
                color: {c["text"]};
            }}
            #navList::item {{
                padding: 10px 16px;
                border: none;
                border-radius: 12px;
                margin: 2px 8px;
                color: {c["text"]};
            }}
            #navList::item:hover {{
                background: {c["hover_bg"]};
            }}
            #navList::item:selected {{
                background: {c["active_bg"]};
                color: {c["active_bar"]};
            }}

            QStackedWidget {{
                background: {c["panel_bg"]};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}

            #sectionTitle {{
                font-size: 20px;
                font-weight: 700;
                color: {c["text"]};
                border: none;
            }}
            #sectionDesc {{
                font-size: 13px;
                color: {c["text_secondary"]};
                border: none;
            }}

            #settingLabel {{
                font-size: 14px;
                font-weight: 600;
                color: {c["text"]};
                border: none;
            }}
            #settingDesc {{
                font-size: 12px;
                color: {c["text_secondary"]};
                border: none;
            }}
            #rowSep {{
                background: {c["border"]};
                border: none;
            }}

            #settingCombo {{
                background: {c["input_bg"]};
                color: {c["text"]};
                border: 1px solid {c["input_border"]};
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            #settingCombo:hover {{
                border-color: {c["text_muted"]};
            }}
            #settingCombo:focus {{
                border-color: {c["focus_border"]};
            }}
            #settingCombo::drop-down {{
                border: none;
                width: 24px;
            }}
            #settingCombo QAbstractItemView {{
                background: {c["input_bg"]};
                border: 1px solid {c["input_border"]};
                border-radius: 10px;
                selection-background-color: {c["active_bg"]};
                selection-color: {c["text"]};
                outline: none;
            }}
            #settingCombo QAbstractItemView::item {{
                padding: 6px 12px;
            }}

            #profileList {{
                background: {c["input_bg"]};
                border: 1px solid {c["input_border"]};
                border-radius: 12px;
                font-size: 13px;
                color: {c["text"]};
                outline: none;
            }}
            #profileList::item {{
                padding: 10px 16px;
                border: none;
                border-radius: 8px;
                margin: 1px 4px;
                color: {c["text"]};
            }}
            #profileList::item:hover {{
                background: {c["hover_bg"]};
            }}
            #profileList::item:selected {{
                background: {c["active_bg"]};
            }}
            #profileInput {{
                background: {c["input_bg"]};
                color: {c["text"]};
                border: 1px solid {c["input_border"]};
                border-radius: 14px;
                padding: 8px 14px;
                font-size: 13px;
            }}
            #profileInput:focus {{
                border-color: {c["focus_border"]};
            }}
            #addProfileBtn {{
                background: {c["accent"]};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            #addProfileBtn:hover {{
                background: {c["accent_hover"]};
            }}
            #settingsActionBtn {{
                background: {c["accent"]};
                color: {c["badge_fg"]};
                border: none;
                border-radius: 14px;
                padding: 9px 16px;
                font-size: 13px;
                font-weight: 600;
                min-width: 132px;
            }}
            #settingsActionBtn:hover {{
                background: {c["accent_hover"]};
            }}
            #settingToggle {{
                spacing: 0;
                border: none;
            }}
            #settingToggle::indicator {{
                width: 44px;
                height: 24px;
                border-radius: 12px;
                background: {c["input_border"]};
            }}
            #settingToggle::indicator:checked {{
                background: {c["accent"]};
            }}
            #settingToggle::indicator:disabled {{
                background: {c["section_line"]};
            }}

            #aboutName {{
                font-size: 22px;
                font-weight: 700;
                color: {c["text"]};
                border: none;
            }}
            #aboutVersion {{
                font-size: 13px;
                color: {c["text_secondary"]};
                border: none;
            }}
            #aboutDesc {{
                font-size: 13px;
                color: {c["text_secondary"]};
                border: none;
                line-height: 1.6;
            }}

            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {c["input_border"]};
                min-height: 30px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {c["text_muted"]};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

    # ── Load ──────────────────────────────────────────────────────

    def _load(self) -> None:
        for cb, val in [
            (self._lang_combo, self._settings.lang),
            (self._theme_combo, self._settings.theme),
            (self._sens_combo, self._settings.sensitivity),
        ]:
            idx = cb.findData(val)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        if hasattr(self, "_profile_list"):
            self._refresh_profiles()

    def _refresh_profiles(self) -> None:
        if not hasattr(self, "_profile_list"):
            return
        self._profile_list.clear()
        profiles = self._profile_manager.list_all()
        active_id = self._profile_manager.active_id
        for p in profiles:
            txt = p.name
            if p.id == active_id:
                txt += f"  {t('settings.activeBadge', '✓ Active')}"
            item = QListWidgetItem(txt)
            item.setData(Qt.ItemDataRole.UserRole, p.id)
            if p.id == active_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._profile_list.addItem(item)

    # ── Navigation ────────────────────────────────────────────────

    def _on_nav_clicked(self, item: QListWidgetItem) -> None:
        key = item.data(Qt.ItemDataRole.UserRole)
        mapping = {"general": 0, "detection": 1, "alerts": 2, "data": 3, "about": 4}
        idx = mapping.get(key, 0)
        self._stack.setCurrentIndex(idx)

        for i, (k, label, it) in enumerate(self._nav_items):
            if k == key:
                self._set_active_nav(i)
                break

    def _rebuild_content(self) -> None:
        """Rebuild dialog content after language change."""
        # Update window title
        self.setWindowTitle(t("settings.title", "Settings"))

        # Update nav items
        nav_keys_map = {
            "general": "settings.general",
            "detection": "settings.detection",
            "alerts": "settings.alerts",
            "data": "settings.dataManagement",
            "about": "settings.about",
        }
        for key, label, item in self._nav_items:
            i18n_key = nav_keys_map.get(key, key)
            item.setText(f"  {t(i18n_key, key.title())}")

        # Save current page
        current = self._stack.currentIndex()

        # Remove all pages from stack
        old_pages = []
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            old_pages.append(w)

        # Rebuild pages (creates new combo boxes)
        self._stack.addWidget(self._page_general())
        self._stack.addWidget(self._page_detection())
        self._stack.addWidget(self._page_alerts())
        self._stack.addWidget(self._page_data_management())
        self._stack.addWidget(self._page_about())

        # Restore page
        self._stack.setCurrentIndex(current)

        # Reload settings into new widgets
        self._load()

        # Delete old pages
        for w in old_pages:
            w.deleteLater()

        # Re-set active nav
        self._set_active_nav(current)

        # Re-apply M3 style
        self._m3 = M3_DARK if self._theme_manager.effective == "dark" else M3_LIGHT
        self._apply_m3_style()

    # ── Handlers ──────────────────────────────────────────────────

    def _on_lang(self) -> None:
        lang = self._lang_combo.currentData()
        if lang and lang != self._settings.lang:
            self._settings.lang = lang
            self._save()
            self._i18n.set_language(lang)
            self._rebuild_content()
            self.language_changed.emit(lang)

    def _on_theme(self) -> None:
        mode = self._theme_combo.currentData()
        if mode and mode != self._settings.theme:
            self._settings.theme = mode
            self._save()
            self._theme_manager.mode = mode
            self._m3 = M3_DARK if self._theme_manager.effective == "dark" else M3_LIGHT
            self._apply_m3_style()
            self.theme_changed.emit(mode)

    def _on_sensitivity(self) -> None:
        level = self._sens_combo.currentData()
        if level and level != self._settings.sensitivity:
            self._settings.sensitivity = level
            self._save()

    def _on_bool_setting(self, name: str, checked: bool) -> None:
        setattr(self._settings, name, checked)
        self._save()

    def _on_profile_click(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid and pid != self._profile_manager.active_id:
            self._profile_manager.activate(pid)
            self._refresh_profiles()

    def _add_profile(self) -> None:
        name = self._profile_input.text().strip()
        if not name:
            return
        self._profile_manager.create(name)
        self._profile_input.clear()
        self._refresh_profiles()

    def _export_logs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("settings.exportLogs", "Export Logs"),
            "fallguard_logs.zip",
            "FallGuard Archive (*.zip);;JSON (*.json);;All (*)",
        )
        if not path:
            return
        try:
            if self._repos is not None:
                from ..data_services import ExportService

                result = ExportService(self._repos.db).export(Path(path))
                QMessageBox.information(
                    self,
                    t("settings.exportLogs", "Export Logs"),
                    f"Exported to {result.path}",
                )
            else:
                data = self._profile_manager.snapshot()
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, t("settings.exportLogs", "Export Logs"), str(exc))

    def _clear_history(self) -> None:
        answer = QMessageBox.question(
            self,
            t("settings.clearHistory", "Clear History"),
            t("settings.clearHistoryDesc", "Remove local event history for demo cleanup."),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            if self._repos is not None and self._media_root is not None:
                from ..data_services import HistoryService

                warnings = HistoryService(self._repos.db, self._media_root).clear()
                if warnings:
                    QMessageBox.warning(
                        self,
                        t("settings.clearHistory", "Clear History"),
                        "History was cleared, but some media files could not be removed:\n"
                        + "\n".join(warnings[:5]),
                    )
            for profile in self._profile_manager.list_all():
                profile.fall_events.clear()
            save = getattr(self._profile_manager, "_save", None)
            if callable(save):
                save()
            self._refresh_profiles()
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, t("settings.clearHistory", "Clear History"), str(exc))

    def _open_dataset_folder(self) -> None:
        if self._repos is not None and self._media_root is not None:
            from .dataset_dialog import DatasetDialog

            DatasetDialog(self._repos.media, self._media_root, self).exec()
            return
        try:
            from ..web_app import find_app_root, writable_output_root
        except ImportError:
            from fall_prediction_desktop.web_app import find_app_root, writable_output_root  # type: ignore[no-redef]
        folder = writable_output_root(find_app_root())
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _save(self) -> None:
        try:
            from ..web_app import save_settings, find_app_root
        except ImportError:
            from fall_prediction_desktop.web_app import save_settings, find_app_root  # type: ignore[no-redef]
        save_settings(find_app_root(), self._settings)
