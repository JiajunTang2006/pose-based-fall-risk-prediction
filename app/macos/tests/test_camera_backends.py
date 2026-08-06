from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from fall_prediction.camera import _candidate_backends


def test_windows_backend_order_prefers_directshow_then_msmf():
    cv2 = SimpleNamespace(CAP_DSHOW=700, CAP_MSMF=1400, CAP_ANY=0)
    with patch.object(sys, "platform", "win32"):
        assert _candidate_backends(cv2) == [
            (700, "DirectShow"),
            (1400, "Media Foundation"),
            (0, "default"),
        ]
