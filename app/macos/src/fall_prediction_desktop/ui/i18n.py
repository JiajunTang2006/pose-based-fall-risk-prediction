from __future__ import annotations

import json
from pathlib import Path


class I18n:
    STORAGE_KEY = "fallguard_lang"

    def __init__(self, locales_dir: Path) -> None:
        self._locales_dir = locales_dir
        self._translations: dict[str, dict] = {}
        self._current_lang: str = "en"
        self._ready: bool = False


    def init(self) -> None:
        self._translations = {}
        for lang in ("en", "zh"):
            path = self._locales_dir / f"{lang}.json"
            try:
                self._translations[lang] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._translations[lang] = {}

        saved = self._load_saved()
        if saved in self._translations:
            self._current_lang = saved
        self._ready = True

    def t(self, key: str, fallback: str = "") -> str:
        if not self._ready:
            return fallback or key
        value = self._resolve(self._translations.get(self._current_lang, {}), key)
        if value is not None:
            return value
        value = self._resolve(self._translations.get("en", {}), key)
        if value is not None:
            return value
        return fallback or key

    def t_dynamic(self, english_string: str) -> str:
        if not english_string:
            return english_string
        if self._current_lang == "en":
            return english_string
        backend_map = self._translations.get("en", {}).get("backend", {})
        key = backend_map.get(english_string, "")
        if key:
            translated = self.t(key)
            if translated and translated != key:
                return translated
        return english_string

    def set_language(self, lang: str) -> None:
        if lang not in self._translations:
            return
        self._current_lang = lang
        self._save_lang(lang)

    @property
    def language(self) -> str:
        return self._current_lang

    @property
    def ready(self) -> bool:
        return self._ready


    @staticmethod
    def _resolve(lang_obj: dict, path: str) -> str | None:
        parts = path.split(".")
        current = lang_obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, str) else None

    def _load_saved(self) -> str:
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("FallGuard", "FallGuard")
            return settings.value(self.STORAGE_KEY, "en")
        except Exception:
            return "en"

    def _save_lang(self, lang: str) -> None:
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("FallGuard", "FallGuard")
            settings.setValue(self.STORAGE_KEY, lang)
        except Exception:
            pass


_i18n: I18n | None = None


def get_i18n() -> I18n:
    assert _i18n is not None, "Call init_i18n() first"
    return _i18n


def init_i18n(locales_dir: Path) -> I18n:
    global _i18n
    _i18n = I18n(locales_dir)
    _i18n.init()
    return _i18n


def t(key: str, fallback: str = "") -> str:
    return get_i18n().t(key, fallback)


def t_dynamic(english: str) -> str:
    return get_i18n().t_dynamic(english)
