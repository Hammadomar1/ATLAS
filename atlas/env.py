from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from atlas.scenarios import ScenarioConfig


@dataclass
class EnvState:
    position: float
    velocity: float
    tick: int


class SimpleCorridorEnv:
    """Minimal Gym-like environment producing RGB frames."""

    def __init__(self, width: int = 32, height: int = 24) -> None:
        self._width = width
        self._height = height
        self._rng = random.Random(0)
        self._scenario: ScenarioConfig | None = None
        self._state = EnvState(position=0.0, velocity=0.0, tick=0)

    def reset(self, seed: int, scenario_cfg: ScenarioConfig) -> Sequence[Sequence[Sequence[int]]]:
        self._rng = random.Random(seed)
        self._scenario = scenario_cfg
        self._state = EnvState(position=0.0, velocity=0.0, tick=0)
        return self._render()

    def step(self, action: float) -> tuple[Sequence[Sequence[Sequence[int]]], float, bool, dict[str, Any]]:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before stepping.")
        position = self._state.position + action * 0.1
        velocity = action
        self._state = EnvState(position=position, velocity=velocity, tick=self._state.tick + 1)
        collision = abs(position) > self._scenario.corridor_width
        near_miss = max(0.0, abs(position) - self._scenario.corridor_width * 0.8)
        info = {
            "collision": collision,
            "near_miss": near_miss,
            "distance_to_obstacle": max(0.0, self._scenario.corridor_width - abs(position)),
            "progress": self._state.tick,
            "velocity": velocity,
        }
        reward = 1.0 - near_miss
        done = collision
        return self._render(), reward, done, info

    def downsample(self, frame: Sequence[Sequence[Sequence[int]]], factor: int = 2) -> Sequence[Sequence[Sequence[int]]]:
        return frame[::factor]

    def _render(self) -> Sequence[Sequence[Sequence[int]]]:
        scenario = self._scenario
        noise = scenario.rgb_noise if scenario else 0.0
        frame = []
        for _ in range(self._height):
            row = []
            for _ in range(self._width):
                base = int(128 + self._state.position * 10)
                pixel = [
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                ]
                row.append(pixel)
            frame.append(row)
        return frame

    @staticmethod
    def _clamp(value: int) -> int:
        return max(0, min(255, value))
