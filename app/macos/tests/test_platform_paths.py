from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fall_prediction_desktop.paths import media_output_dir, user_data_dir


def test_windows_data_dir_uses_localappdata(tmp_path, monkeypatch):
    local_app_data = tmp_path / "LocalAppData"
    user_profile = tmp_path / "User"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("USERPROFILE", str(user_profile))
    monkeypatch.delenv("FALLGUARD_DATA_DIR", raising=False)

    with patch.object(sys, "platform", "win32"):
        assert user_data_dir() == local_app_data / "FallGuard"
        assert media_output_dir() == user_profile / "Videos" / "FallGuard"


def test_linux_data_dir_uses_xdg(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.delenv("FALLGUARD_DATA_DIR", raising=False)

    with patch.object(sys, "platform", "linux"):
        assert user_data_dir() == xdg / "FallGuard"
