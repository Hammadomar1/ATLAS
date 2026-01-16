from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from atlas.types import ShieldOutput


@dataclass(frozen=True)
class ShieldConfig:
    hazard_threshold: float
    halt_action: float


class SafetyShield:
    """Deterministic action shield based on onboard reflex drive."""

    def __init__(self, config: ShieldConfig) -> None:
        self._config = config

    def apply(self, action: float, drive: Sequence[float]) -> ShieldOutput:
        safe_indices = [i for i, hazard in enumerate(drive) if hazard < self._config.hazard_threshold]
        if not safe_indices:
            return ShieldOutput(
                action=self._config.halt_action,
                used_fallback=True,
                safe_set_empty=True,
            )
        selected_index = min(safe_indices, key=lambda i: abs(i - self._bearing_index_from_action(action)))
        safe_action = self._action_from_bearing_index(selected_index, len(drive))
        return ShieldOutput(
            action=safe_action,
            used_fallback=False,
            safe_set_empty=False,
        )

    @staticmethod
    def _bearing_index_from_action(action: float) -> int:
        clipped = max(-1.0, min(1.0, action))
        return int(round((clipped + 1.0) * 0.5 * 10))

    @staticmethod
    def _action_from_bearing_index(index: int, size: int) -> float:
        if size <= 1:
            return 0.0
        ratio = index / (size - 1)
        return ratio * 2.0 - 1.0
