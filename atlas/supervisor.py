from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from atlas.types import SupervisorState


@dataclass
class SupervisorConfig:
    hazard_guard_threshold: float
    oscillation_guard_threshold: float
    meta_timeout_ticks: int
    guard_dwell_ticks: int


class Supervisor:
    """Maintains mode state and gates advisory influence."""

    def __init__(self, config: SupervisorConfig) -> None:
        self._config = config
        self._mode = "REFLEX_ONLY"
        self._guard_counter = 0
        self._last_meta_tick: Optional[int] = None

    def observe_meta(self, t: int) -> None:
        self._last_meta_tick = t

    def update(self, t: int, hazard: float, oscillation: float, novelty: Optional[float]) -> SupervisorState:
        guard_requested = (
            hazard >= self._config.hazard_guard_threshold
            or oscillation >= self._config.oscillation_guard_threshold
            or (novelty is not None and novelty >= 0.7)
        )
        if guard_requested:
            self._guard_counter = self._config.guard_dwell_ticks
        else:
            self._guard_counter = max(0, self._guard_counter - 1)

        self._mode = "GUARDED" if self._guard_counter > 0 else "REFLEX_ONLY"
        meta_age = None
        if self._last_meta_tick is not None:
            meta_age = t - self._last_meta_tick
        meta_ok = meta_age is not None and meta_age <= self._config.meta_timeout_ticks
        guard_enabled = self._mode == "GUARDED"
        return SupervisorState(
            mode=self._mode,
            meta_ok=meta_ok,
            guard_enabled=guard_enabled,
            meta_age=meta_age,
        )
