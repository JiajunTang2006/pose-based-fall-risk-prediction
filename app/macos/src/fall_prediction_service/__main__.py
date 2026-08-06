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

    if args.resource_root:
        resource_root = Path(args.resource_root).expanduser().resolve()
    else:
        resource_root = _default_resource_root()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        from fall_prediction_desktop.database.init_db import default_data_dir
        data_dir = default_data_dir(resource_root)

    os.environ.setdefault("FALLGUARD_DATA_DIR", str(data_dir))

    token = args.token or secrets.token_urlsafe(32)

    host = args.host
    if host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--host must be a loopback address (127.0.0.1, localhost, or ::1)")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

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

    ready_msg = json.dumps({
        "event": "ready",
        "port": actual_port,
        "token": token,
        "api_version": API_VERSION,
        "pid": os.getpid(),
    })
    print(ready_msg, flush=True)

    lifecycle = service.lifecycle
    install_signal_handlers(lifecycle)
    lifecycle.start_watchdog()

    logger.info("Listening on %s:%s (api=%s)", host, actual_port, API_VERSION)

    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        lifecycle.shutdown()
        logger.info("FallGuard AI Service stopped.")


def _default_resource_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)

    here = Path(__file__).resolve().parent
    for candidate in (here.parent.parent, here.parent):
        if (
            (candidate / "models").is_dir()
            and (candidate / "configs").is_dir()
            and (candidate / "src").is_dir()
        ):
            return candidate
    return here.parent.parent


if __name__ == "__main__":
    main()
