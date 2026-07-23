"""
FallGuard AI Service — headless entry point.

Start the service::

    python -m fall_prediction_service --port 0 --data-dir /tmp/fallguard-dev

The service prints a single ``ready`` JSON line to stdout and then listens for
HTTP requests on ``127.0.0.1`` until it receives SIGTERM, SIGINT, or a
``POST /api/v1/shutdown`` request.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path

from fall_prediction_service import API_VERSION, __version__
from fall_prediction_service.app import create_service
from fall_prediction_service.lifecycle import ServiceLifecycle, install_signal_handlers

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FallGuard AI Service — headless prediction & monitoring API.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to listen on (0 = OS-assigned).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token (auto-generated if not provided).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Writable data directory (default: ~/Library/Application Support/FallGuard).",
    )
    parser.add_argument(
        "--resource-root",
        default=None,
        help="Read-only resource root with models/ and configs/.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="PID of the parent process. Service exits when this PID disappears.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"FallGuard AI Service {__version__}",
    )
    args = parser.parse_args(argv)

    # ── resolve paths ───────────────────────────────────────────────
    if args.resource_root:
        resource_root = Path(args.resource_root).expanduser().resolve()
    else:
        resource_root = _default_resource_root()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        # Preserve the legacy Movies/FallGuard database when present; new
        # installations use Application Support through default_data_dir().
        from fall_prediction_desktop.database.init_db import default_data_dir
        data_dir = default_data_dir(resource_root)

    # Set env so sub-modules also use the correct data dir
    os.environ.setdefault("FALLGUARD_DATA_DIR", str(data_dir))

    # ── token ───────────────────────────────────────────────────────
    token = args.token or secrets.token_urlsafe(32)

    # ── host validation ─────────────────────────────────────────────
    host = args.host
    if host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--host must be a loopback address (127.0.0.1, localhost, or ::1)")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

    # ── create service ──────────────────────────────────────────────
    service = create_service(
        host=host,
        port=args.port,
        token=token,
        data_dir=data_dir,
        resource_root=resource_root,
        parent_pid=args.parent_pid,
        debug=args.debug,
    )

    actual_port = service.server_address[1]

    # ── print ready line (MUST be the first stdout line) ────────────
    ready_msg = json.dumps({
        "event": "ready",
        "port": actual_port,
        "token": token,
        "api_version": API_VERSION,
        "pid": os.getpid(),
    })
    print(ready_msg, flush=True)

    # ── signal handlers ─────────────────────────────────────────────
    lifecycle = service.lifecycle
    install_signal_handlers(lifecycle)
    lifecycle.start_watchdog()

    logger.info("Listening on %s:%s (api=%s)", host, actual_port, API_VERSION)

    # ── serve ───────────────────────────────────────────────────────
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        lifecycle.shutdown()
        logger.info("FallGuard AI Service stopped.")


def _default_resource_root() -> Path:
    """Best-guess the resource root (models/, configs/, etc.)."""
    # Inside a PyInstaller bundle, _MEIPASS is the Resources dir.
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)

    # Running from source: walk up from this file.
    here = Path(__file__).resolve().parent  # fall_prediction_service/
    for candidate in (here.parent.parent, here.parent):  # macos/, src/
        if (candidate / "models").is_dir() and (candidate / "web").is_dir():
            return candidate
    return here.parent.parent  # fallback to macos/


if __name__ == "__main__":
    main()
