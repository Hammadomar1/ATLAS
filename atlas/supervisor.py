from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from atlas.types import SupervisorState


@dataclass
class SupervisorConfig:
    hazard_high: float
    hazard_low: float
    oscillation_high: float
    novelty_high: float
    meta_timeout_ticks: int


class Supervisor:
    """Maintains mode state and gates advisory influence."""

    def __init__(self, config: SupervisorConfig) -> None:
        self._config = config
        self._mode = "Normal"
        self._meta_age: Optional[int] = 0

    def observe_meta(self, t: int) -> None:
        self._meta_age = 0

    def update(
        self,
        t: int,
        hazard: float,
        oscillation: float,
        novelty: Optional[float],
        safe_set_empty: bool,
        meta_arrived: bool,
    ) -> SupervisorState:
        if meta_arrived:
            self._meta_age = 0
        elif self._meta_age is not None:
            self._meta_age += 1

        hazard_active = hazard >= self._config.hazard_high
        guard_active = (
            oscillation >= self._config.oscillation_high
            or (novelty is not None and novelty >= self._config.novelty_high)
        )
        recover = hazard <= self._config.hazard_low and not safe_set_empty

        if self._mode == "Normal":
            if hazard_active or safe_set_empty:
                self._mode = "Emergency"
            elif guard_active:
                self._mode = "Guarded"
        elif self._mode == "Guarded":
            if hazard_active or safe_set_empty:
                self._mode = "Emergency"
            elif not guard_active:
                self._mode = "Normal"
        elif self._mode == "Emergency":
            if recover:
                self._mode = "Guarded"

        meta_age = self._meta_age
        meta_ok = meta_age is not None and meta_age <= self._config.meta_timeout_ticks
        accept_meta = self._mode != "Emergency" and meta_ok
        guard_enabled = self._mode == "Guarded"
        return SupervisorState(
            mode=self._mode,
            meta_ok=meta_ok,
            guard_enabled=guard_enabled,
            meta_age=meta_age,
            accept_meta=accept_meta,
        )
