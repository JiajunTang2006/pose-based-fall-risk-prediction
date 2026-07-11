"""Local alert delivery for risk-state transitions.

macOS desktop notifications are intentionally not implemented yet.  A future
notification adapter can be added beside ``SoundAlertService`` without changing
the prediction or event-persistence layers.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .risk_state_machine import RiskState, StateChangeEvent


class SoundAlertService:
    """Play a bounded local sound when monitoring enters an elevated state."""

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        player: Callable[[RiskState], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._clock = clock
        self._player = player or self._play_system_sound
        self._last_alert_at: float | None = None
        self._lock = threading.Lock()

    def on_state_change(self, event: StateChangeEvent) -> bool:
        """Deliver an escalation sound; return whether delivery was scheduled."""
        if not self.enabled or not event.is_escalation:
            return False
        if event.to_state not in {RiskState.WARNING, RiskState.FALL}:
            return False

        now = self._clock()
        with self._lock:
            if (
                self._last_alert_at is not None
                and now - self._last_alert_at < self._cooldown_seconds
                and event.to_state is not RiskState.FALL
            ):
                return False
            self._last_alert_at = now

        threading.Thread(
            target=self._play_safely,
            args=(event.to_state,),
            name="fallguard-sound-alert",
            daemon=True,
        ).start()
        return True

    def _play_safely(self, state: RiskState) -> None:
        try:
            self._player(state)
        except (OSError, subprocess.SubprocessError):
            # Alert delivery must never interrupt frame processing.
            pass

    @staticmethod
    def _play_system_sound(state: RiskState) -> None:
        if sys.platform != "darwin":
            return
        sound_name = "Sosumi.aiff" if state is RiskState.FALL else "Ping.aiff"
        sound_path = Path("/System/Library/Sounds") / sound_name
        if sound_path.is_file():
            subprocess.run(
                ["/usr/bin/afplay", str(sound_path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4.0,
            )
