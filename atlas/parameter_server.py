from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class ParameterBounds:
    minimum: Sequence[float]
    maximum: Sequence[float]


class ParameterServer:
    """Owns tunable parameters and applies safe projection."""

    def __init__(self, initial_theta: Sequence[float], bounds: ParameterBounds) -> None:
        if len(initial_theta) != len(bounds.minimum) or len(initial_theta) != len(bounds.maximum):
            raise ValueError("Theta and bounds dimension mismatch")
        self._theta = list(initial_theta)
        self._bounds = bounds

    @property
    def theta(self) -> List[float]:
        return list(self._theta)

    def apply_delta(self, delta_theta: Sequence[float]) -> List[float]:
        if len(delta_theta) != len(self._theta):
            raise ValueError("Delta theta dimension mismatch")
        projected = []
        for value, delta, min_value, max_value in zip(
            self._theta, delta_theta, self._bounds.minimum, self._bounds.maximum
        ):
            updated = value + delta
            if updated < min_value:
                projected.append(min_value)
            elif updated > max_value:
                projected.append(max_value)
            else:
                projected.append(updated)
        self._theta = projected
        return list(self._theta)
