# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the headless FallGuard AI service."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "src" / "fall_prediction_service" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ROOT / "models"), "models"),
        (str(ROOT / "configs"), "configs"),
        (
            str(ROOT / "src" / "fall_prediction_desktop" / "database" / "schema.sql"),
            "fall_prediction_desktop/database",
        ),
    ],
    hiddenimports=[
        "fall_prediction_service",
        "fall_prediction_desktop.web_app",
        "fall_prediction_desktop.database.init_db",
        "fall_prediction.video_app",
        "fall_prediction.ensemble_predictor",
        "fall_prediction.lying_adl_filter",
        "fall_prediction.deep_dataset",
        "fall_prediction.deep_model",
        "fall_prediction.fusion_model",
        "fall_prediction.skeleton_dataset",
        "torch",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        *collect_submodules("sklearn.ensemble._hist_gradient_boosting"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "PyQt5",
        "PyQt6",
        "webview",
        "rumps",
        "tkinter",
        "fall_prediction_desktop.gui",
        "fall_prediction_desktop.menubar",
        "fall_prediction_desktop.ui",
        "polars",
        "_polars_runtime_32",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fallguard-ai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="fallguard-ai",
)
