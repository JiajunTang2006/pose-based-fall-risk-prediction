# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/tangjiajun/Desktop/apps/macos/src/fall_prediction_desktop/__main__.py'],
    pathex=['/Users/tangjiajun/Desktop/apps/macos/src'],
    binaries=[],
    datas=[('/Users/tangjiajun/Desktop/apps/macos/models', 'models'), ('/Users/tangjiajun/Desktop/apps/macos/assets', 'assets')],
    hiddenimports=['matplotlib', 'objc', 'rumps', 'joblib', 'sklearn', 'sklearn.ensemble', 'sklearn.ensemble._hist_gradient_boosting', 'sklearn.preprocessing', 'sklearn.tree', 'sklearn.utils', 'sklearn.base', 'sklearn.metrics', 'sklearn.model_selection', 'PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtCore', 'shiboken6', 'fall_prediction', 'fall_prediction.camera', 'fall_prediction.runtime', 'fall_prediction.landmarks', 'fall_prediction.features', 'fall_prediction.risk', 'fall_prediction.predictor', 'fall_prediction.pose', 'fall_prediction.ml_features', 'fall_prediction.ml_predictor', 'fall_prediction.window_dataset', 'fall_prediction.config', 'fall_prediction.video_app', 'fall_prediction_desktop', 'fall_prediction_desktop.web_app', 'fall_prediction_desktop.menubar', 'fall_prediction_desktop.runner', 'fall_prediction_desktop.ui', 'fall_prediction_desktop.ui.main_window', 'fall_prediction_desktop.ui.settings_dialog', 'fall_prediction_desktop.ui.widgets', 'fall_prediction_desktop.ui.theme', 'fall_prediction_desktop.ui.i18n'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'notebook', 'tensorboard', 'pytest', 'mediapipe', 'tkinter', '_tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'wx', 'sphinx', 'docutils', 'pywebview', 'polars', '_polars_runtime_32', 'pyarrow', 'pandas', 'torch._inductor', 'torch._dynamo'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FallGuard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='/Users/tangjiajun/Desktop/apps/macos/entitlements.plist',
    icon=['/Users/tangjiajun/Desktop/apps/macos/assets/FallGuard.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FallGuard',
)
app = BUNDLE(
    coll,
    name='FallGuard.app',
    icon='/Users/tangjiajun/Desktop/apps/macos/assets/FallGuard.icns',
    bundle_identifier='com.fallguard.desktop',
)
