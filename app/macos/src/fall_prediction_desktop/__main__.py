"""
FallGuard entry point.

Default:   open the native PySide6 desktop window.
--menubar: run as a lightweight menu bar app instead.
--connect: open a monitor window connected to an already-running menu bar server.
"""

from __future__ import annotations

import argparse

# Relative imports work when running as `python -m fall_prediction_desktop`.
# Absolute imports work inside a PyInstaller bundle where relative imports fail.
try:
    from . import __version__
    from .web_app import main_native as run_native
    from .web_app import connect_and_show
    from .menubar import main as run_menubar
except ImportError:
    from fall_prediction_desktop import __version__  # type: ignore[no-redef]
    from fall_prediction_desktop.web_app import main_native as run_native  # type: ignore[no-redef]
    from fall_prediction_desktop.web_app import connect_and_show  # type: ignore[no-redef]
    from fall_prediction_desktop.menubar import main as run_menubar  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FallGuard — AI fall detection.")
    parser.add_argument("--version", action="version", version=f"FallGuard {__version__}")
    parser.add_argument(
        "--menubar",
        action="store_true",
        help="Run as a menu bar app (instead of opening the desktop window).",
    )
    parser.add_argument(
        "--connect",
        metavar="URL",
        help="Open a monitor window connected to an already-running FallGuard server.",
    )
    args = parser.parse_args(argv)

    if args.connect:
        connect_and_show(args.connect)
    elif args.menubar:
        run_menubar()
    else:
        run_native()


if __name__ == "__main__":
    main()
