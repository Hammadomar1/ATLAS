from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ScenarioConfig:
    corridor_width: float
    obstacle_density: float
    moving_obstacles: bool
    texture_richness: float
    lighting: float
    rgb_noise: float
    blur: float
    wind: float


class ScenarioSampler:
    """Produces deterministic scenario configurations."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def fixed_set(self, count: int) -> List[ScenarioConfig]:
        return [self._make(i, deterministic=True) for i in range(count)]

    def sample(self) -> ScenarioConfig:
        return self._make(self._rng.randrange(0, 10_000), deterministic=False)

    def _make(self, index: int, deterministic: bool) -> ScenarioConfig:
        rng = random.Random(index) if deterministic else self._rng
        return ScenarioConfig(
            corridor_width=rng.uniform(1.0, 5.0),
            obstacle_density=rng.uniform(0.0, 1.0),
            moving_obstacles=rng.random() < 0.5,
            texture_richness=rng.uniform(0.0, 1.0),
            lighting=rng.uniform(0.5, 1.5),
            rgb_noise=rng.uniform(0.0, 0.2),
            blur=rng.uniform(0.0, 1.0),
            wind=rng.uniform(0.0, 0.5),
        )
