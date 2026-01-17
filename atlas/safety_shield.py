from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from atlas.types import ShieldOutput


@dataclass(frozen=True)
class ShieldConfig:
    hazard_threshold: float
    halt_action: float
    candidate_actions: Sequence[float] | None = None
    tie_breaker: str = "smaller_magnitude"


class SafetyShield:
    """Deterministic action shield based on onboard reflex drive."""

    def __init__(self, config: ShieldConfig) -> None:
        self._config = config

    def apply(self, action: float, drive: Sequence[float]) -> ShieldOutput:
        safe_action, safe_set_empty, projected, _ = self.safe_action(action, drive)
        if safe_set_empty:
            return ShieldOutput(
                action=self._config.halt_action,
                used_fallback=True,
                safe_set_empty=True,
            )
        return ShieldOutput(
            action=safe_action,
            used_fallback=projected,
            safe_set_empty=False,
        )

    def safe_action(self, action: float, drive: Sequence[float]) -> tuple[float, bool, bool, int]:
        hazards = self._sanitize_drive(drive)
        if hazards is None or not hazards:
            return self._config.halt_action, True, True, 0
        candidates = list(self._candidate_actions(len(hazards)))
        if len(candidates) != len(hazards):
            candidates = list(self._default_actions(len(hazards)))
        safe_indices = [i for i, hazard in enumerate(hazards) if hazard <= self._config.hazard_threshold]
        if not safe_indices:
            return self._config.halt_action, True, True, 0
        selected_index = min(
            safe_indices,
            key=lambda i: (
                abs(candidates[i] - action),
                abs(candidates[i]) if self._config.tie_breaker == "smaller_magnitude" else i,
                i,
            ),
        )
        safe_action = candidates[selected_index]
        projected = safe_action != action
        return safe_action, False, projected, len(safe_indices)

    def _candidate_actions(self, size: int) -> Sequence[float]:
        if self._config.candidate_actions is not None:
            return self._config.candidate_actions
        return self._default_actions(size)

    @staticmethod
    def _default_actions(size: int) -> Sequence[float]:
        if size <= 1:
            return [0.0]
        return [index / (size - 1) * 2.0 - 1.0 for index in range(size)]

    @staticmethod
    def _sanitize_drive(drive: Sequence[float]) -> Sequence[float] | None:
        hazards = []
        for hazard in drive:
            if not isfinite(hazard):
                return None
            hazards.append(float(hazard))
        return hazards
