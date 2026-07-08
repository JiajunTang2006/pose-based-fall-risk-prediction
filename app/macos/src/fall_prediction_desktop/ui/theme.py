"""
Google Material Design 3 token system for FallGuard.
Surface layering: bg → surface → surface-container → card.
8px spacing grid throughout.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QApplication

# ── M3 Color Tokens ──────────────────────────────────────────────

LIGHT = {
    # Primary (actions, focus, active states)
    "primary":           "#22C55E",
    "on_primary":        "#FFFFFF",
    "primary_container": "#EAF7EF",
    "on_primary_container": "#14532D",

    # Secondary
    "secondary":         "#16A34A",
    "secondary_container": "#DCFCE7",

    # Tertiary / accent
    "tertiary":          "#16A34A",

    # Error
    "error":             "#EA4335",
    "error_container":   "#FCE8E6",
    "on_error_container": "#C5221F",

    # Warning
    "warning":           "#F59E0B",
    "warning_container": "#FEF3C7",

    # Surface layering (light)
    "surface_dim":       "#F8FAFC",
    "surface":           "#F8FAFC",
    "surface_bright":    "#FFFFFF",
    "surface_container_lowest": "#FFFFFF",
    "surface_container_low":    "#F3F6FB",
    "surface_container":        "#EDEFF2",
    "surface_container_high":   "#E8EAED",
    "surface_container_highest":"#E2E4E8",

    # On-surface (text)
    "on_surface":        "#1F2937",
    "on_surface_variant":"#4B5563",
    "on_surface_secondary": "#6B7280",
    "outline":           "#9CA3AF",
    "outline_variant":   "#E5E7EB",

    # Shadows / elevation
    "shadow_1": "0 1px 2px rgba(16, 24, 40, 0.04)",
    "shadow_2": "0 1px 3px rgba(16, 24, 40, 0.06), 0 1px 2px rgba(16, 24, 40, 0.04)",
    "shadow_3": "0 2px 8px rgba(16, 24, 40, 0.06), 0 1px 4px rgba(16, 24, 40, 0.04)",

    # Monitor / video
    "monitor_bg":        "#111827",
    "monitor_text":      "rgba(209, 213, 219, 1)",

    # Semantic backgrounds
    "success_bg":        "#EAF7EF",
    "success_fg":        "#16A34A",
    "idle_bg":           "#F1F3F4",
    "idle_fg":           "#5F6368",
    "warning_bg":        "#FEF3C7",
    "warning_fg":        "#B45309",
    "danger_bg":         "#FEE2E2",
    "danger_fg":         "#EF4444",

    # Button states
    "primary_hover":     "#16A34A",
    "primary_pressed":   "#15803D",
    "disabled_bg":       "#DADCE0",
    "disabled_fg":       "#9AA0A6",

    # Chart
    "chart_grid": "#E5E7EB",
    "chart_text": "#6B7280",
    "ring_track": "#DFF3E7",
}

DARK = {
    # Primary
    "primary":           "#4ADE80",
    "on_primary":        "#052E16",
    "primary_container": "rgba(34, 197, 94, 0.18)",
    "on_primary_container": "#86EFAC",

    # Secondary
    "secondary":         "#86EFAC",
    "secondary_container": "rgba(34, 197, 94, 0.18)",

    # Tertiary
    "tertiary":          "#86EFAC",

    # Error
    "error":             "#F28B82",
    "error_container":   "rgba(242, 139, 130, 0.18)",
    "on_error_container": "#F28B82",

    # Warning
    "warning":           "#FBBF24",
    "warning_container": "rgba(245, 158, 11, 0.18)",

    # Surface layering (dark) — deep navy, not black
    "surface_dim":       "#0B1220",
    "surface":           "#0B1220",
    "surface_bright":    "#101828",
    "surface_container_lowest": "#09101C",
    "surface_container_low":    "#0B1220",
    "surface_container":        "#111827",
    "surface_container_high":   "#172033",
    "surface_container_highest":"#1A2740",

    # On-surface (text)
    "on_surface":        "#F9FAFB",
    "on_surface_variant":"#CBD5E1",
    "on_surface_secondary":"#94A3B8",
    "outline":           "#64748B",
    "outline_variant":   "#263244",

    # Shadows (subtler in dark)
    "shadow_1": "0 1px 2px rgba(0,0,0,0.25)",
    "shadow_2": "0 1px 3px rgba(0,0,0,0.35)",
    "shadow_3": "0 2px 8px rgba(0,0,0,0.40)",

    # Monitor / video
    "monitor_bg":        "#0A0D12",
    "monitor_text":      "rgba(209, 213, 219, 0.85)",

    # Semantic backgrounds
    "success_bg":        "rgba(34, 197, 94, 0.18)",
    "success_fg":        "#86EFAC",
    "idle_bg":           "#1E293B",
    "idle_fg":           "#94A3B8",
    "warning_bg":        "rgba(253, 214, 99, 0.18)",
    "warning_fg":        "#FBBF24",
    "danger_bg":         "rgba(242, 139, 130, 0.18)",
    "danger_fg":         "#F28B82",

    # Button states
    "primary_hover":     "#A2C9FA",
    "primary_pressed":   "#B8D4FB",
    "disabled_bg":       "#3C4043",
    "disabled_fg":       "#9AA0A6",

    # Chart
    "chart_grid": "#263244",
    "chart_text": "#94A3B8",
    "ring_track": "#1F3A2A",
}


class ThemeManager(QObject):
    """Manages light/dark/system theme, emits signal on change."""

    theme_changed = Signal(str)

    def __init__(self, mode: str = "system") -> None:
        super().__init__()
        self._mode = mode
        self._effective = "light"
        self._resolve_effective()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("light", "dark", "system"):
            return
        self._mode = value
        old = self._effective
        self._resolve_effective()
        if self._effective != old:
            self.theme_changed.emit(self._effective)

    @property
    def effective(self) -> str:
        return self._effective

    @property
    def colors(self) -> dict[str, str]:
        return DARK if self._effective == "dark" else LIGHT

    def _resolve_effective(self) -> None:
        if self._mode == "system":
            try:
                app = QApplication.instance()
                if app:
                    scheme = app.styleHints().colorScheme()
                    self._effective = "dark" if scheme == Qt.ColorScheme.Dark else "light"
                    return
            except Exception:
                pass
        self._effective = self._mode if self._mode != "system" else "light"


def build_stylesheet(c: dict[str, str]) -> str:
    """Build M3 stylesheet using Qt property selectors."""

    return f"""
    /* ═══════════ FOUNDATION ═══════════ */
    QWidget {{
        font-family: "Inter", ".AppleSystemUIFont", "Helvetica Neue", sans-serif;
        font-size: 13px;
        color: {c["on_surface"]};
        background: transparent;
    }}

    /* ═══════════ SCROLLBARS (thin, minimal) ═══════════ */
    QScrollBar:vertical {{
        background: transparent; width: 6px; margin: 4px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c["outline_variant"]}; min-height: 32px; border-radius: 3px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c["outline"]}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: none; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

    QScrollBar:horizontal {{
        background: transparent; height: 6px; margin: 0 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c["outline_variant"]}; min-width: 32px; border-radius: 3px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {c["outline"]}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: none; }}

    /* ═══════════ SCROLL AREA ═══════════ */
    QScrollArea {{ border: none; background: transparent; }}

    /* ═══════════ CARDS ═══════════ */
    QFrame[class="Card"] {{
        background: {c["surface_bright"]};
        border: 1px solid {c["outline_variant"]};
        border-radius: 20px;
    }}

    /* ═══════════ LINE EDIT ═══════════ */
    QLineEdit {{
        background: {c["surface_bright"]}; color: {c["on_surface"]};
        border: 1px solid {c["outline_variant"]}; border-radius: 14px;
        padding: 10px 16px; font-size: 14px;
        selection-background-color: {c["primary_container"]};
    }}
    QLineEdit:focus {{
        border-color: {c["primary"]}; border-width: 2px; padding: 9px 15px;
    }}
    QLineEdit:disabled {{
        color: {c["outline"]}; background: {c["surface_container"]};
    }}

    /* ═══════════ COMBOBOX ═══════════ */
    QComboBox {{
        background: {c["surface_bright"]}; color: {c["on_surface"]};
        border: 1px solid {c["outline_variant"]}; border-radius: 12px;
        padding: 8px 32px 8px 12px; min-width: 120px; font-size: 13px;
    }}
    QComboBox:hover {{ border-color: {c["outline"]}; }}
    QComboBox::drop-down {{
        subcontrol-origin: padding; subcontrol-position: top right;
        width: 28px; border: none;
    }}
    QComboBox QAbstractItemView {{
        background: {c["surface_bright"]}; border: 1px solid {c["outline_variant"]};
        border-radius: 10px; padding: 4px; outline: none;
        selection-background-color: {c["primary_container"]};
        selection-color: {c["on_surface"]};
    }}
    QComboBox QAbstractItemView::item {{
        padding: 8px 12px; border-radius: 6px; min-height: 28px;
    }}

    /* ═══════════ TAB WIDGET ═══════════ */
    QTabWidget::pane {{ border: none; background: {c["surface_bright"]}; }}
    QTabBar::tab {{
        background: transparent; color: {c["on_surface_variant"]};
        border: none; border-bottom: 2px solid transparent;
        padding: 10px 20px; font-size: 13px; font-weight: 500; min-width: 80px;
    }}
    QTabBar::tab:hover {{ color: {c["on_surface"]}; background: {c["surface_container"]}; }}
    QTabBar::tab:selected {{
        color: {c["primary"]}; border-bottom: 2px solid {c["primary"]}; font-weight: 600;
    }}

    /* ═══════════ LIST WIDGET ═══════════ */
    QListWidget {{
        background: {c["surface_bright"]}; border: 1px solid {c["outline_variant"]};
        border-radius: 12px; padding: 4px; outline: none;
    }}
    QListWidget::item {{ padding: 10px 14px; border-radius: 8px; margin: 1px 0; }}
    QListWidget::item:hover {{ background: {c["surface_container"]}; }}
    QListWidget::item:selected {{
        background: {c["primary_container"]}; color: {c["on_primary_container"]};
    }}

    /* ═══════════ TOOLTIP ═══════════ */
    QToolTip {{
        background: {c["surface_container_highest"]}; color: {c["on_surface"]};
        border: 1px solid {c["outline_variant"]}; border-radius: 8px;
        padding: 6px 12px; font-size: 12px;
    }}
    """
