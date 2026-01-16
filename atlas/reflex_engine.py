from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from atlas.types import DNFState, ProtoEvent, ReflexDrive, ReflexInput, ReflexOutput


@dataclass
class ReflexConfig:
    bearings: int
    rpe_gain: float
    composer_gain: float
    dnf_decay: float
    dnf_gain: float
    controller_gain: float


class ReflexEngine:
    """Serial reflex path: RPE -> Composer -> DNF -> Controller."""

    def __init__(self, config: ReflexConfig) -> None:
        self._config = config
        self._dnf_state = [0.0 for _ in range(config.bearings)]
        self._last_winner = 0

    def step(self, reflex_input: ReflexInput) -> ReflexOutput:
        primitives = self._rpe(reflex_input.frame)
        events, drive = self._composer(primitives)
        dnf_state = self._dnf(drive)
        action = self._controller(dnf_state.winner_index)
        return ReflexOutput(
            t=reflex_input.t,
            nominal_action=action,
            drive=ReflexDrive(drive=drive),
            dnf_state=dnf_state,
            events=events,
        )

    def _rpe(self, frame: Sequence[Sequence[Sequence[float]]]) -> List[float]:
        row_sums = []
        for row in frame:
            total = sum(sum(pixel) for pixel in row)
            row_sums.append(total)
        if not row_sums:
            return [0.0 for _ in range(self._config.bearings)]
        band_size = max(1, len(row_sums) // self._config.bearings)
        primitives = []
        for i in range(self._config.bearings):
            start = i * band_size
            end = min(len(row_sums), (i + 1) * band_size)
            band_value = sum(row_sums[start:end]) / max(1, end - start)
            primitives.append(self._config.rpe_gain * band_value)
        return primitives

    def _composer(self, primitives: Sequence[float]) -> Tuple[List[ProtoEvent], List[float]]:
        events = []
        drive = []
        for idx, value in enumerate(primitives):
            strength = self._config.composer_gain * value
            drive.append(strength)
            if strength > 0.5:
                events.append(ProtoEvent(bearing_index=idx, strength=strength))
        return events, drive

    def _dnf(self, drive: Sequence[float]) -> DNFState:
        updated = []
        for idx, value in enumerate(drive):
            state = (1.0 - self._config.dnf_decay) * self._dnf_state[idx]
            state += self._config.dnf_gain * math.tanh(value)
            updated.append(state)
        self._dnf_state = updated
        winner_index = int(max(range(len(updated)), key=lambda i: updated[i])) if updated else 0
        self._last_winner = winner_index
        return DNFState(u=list(updated), winner_index=winner_index)

    def _controller(self, winner_index: int) -> float:
        if self._config.bearings <= 1:
            return 0.0
        ratio = winner_index / (self._config.bearings - 1)
        return self._config.controller_gain * (ratio * 2.0 - 1.0)
