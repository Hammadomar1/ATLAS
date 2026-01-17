from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NoiseConfig:
    blur: float
    rgb_noise: float
    dropout_prob: float
    occlusion_prob: float
    occlusion_size: int


@dataclass(frozen=True)
class ScenarioConfig:
    corridor_width: float
    obstacle_density: float
    obstacle_distribution: str
    obstacle_motion: str
    texture_richness: float
    lighting: float
    noise: NoiseConfig
    wind: float
    regime: str


class ScenarioSampler:
    """Produces deterministic scenario configurations."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def fixed_set(self, count: int) -> List[ScenarioConfig]:
        return [self._make(i, deterministic=True) for i in range(count)]

    def sample(self) -> ScenarioConfig:
        return self._make(self._rng.randrange(0, 10_000), deterministic=False)

    def presets(self) -> Dict[str, ScenarioConfig]:
        return {
            "corridor": ScenarioConfig(
                corridor_width=3.5,
                obstacle_density=0.1,
                obstacle_distribution="uniform",
                obstacle_motion="static",
                texture_richness=0.2,
                lighting=1.0,
                noise=NoiseConfig(blur=0.05, rgb_noise=0.02, dropout_prob=0.0, occlusion_prob=0.0, occlusion_size=0),
                wind=0.0,
                regime="corridor",
            ),
            "clutter": ScenarioConfig(
                corridor_width=2.0,
                obstacle_density=0.6,
                obstacle_distribution="clustered",
                obstacle_motion="drift",
                texture_richness=0.6,
                lighting=0.9,
                noise=NoiseConfig(blur=0.2, rgb_noise=0.08, dropout_prob=0.01, occlusion_prob=0.05, occlusion_size=4),
                wind=0.1,
                regime="clutter",
            ),
            "dynamic_traps": ScenarioConfig(
                corridor_width=2.5,
                obstacle_density=0.4,
                obstacle_distribution="clustered",
                obstacle_motion="oscillate",
                texture_richness=0.5,
                lighting=1.2,
                noise=NoiseConfig(blur=0.15, rgb_noise=0.06, dropout_prob=0.02, occlusion_prob=0.1, occlusion_size=6),
                wind=0.2,
                regime="dynamic_traps",
            ),
            "u_shape": ScenarioConfig(
                corridor_width=2.2,
                obstacle_density=0.5,
                obstacle_distribution="u_shape",
                obstacle_motion="static",
                texture_richness=0.4,
                lighting=0.8,
                noise=NoiseConfig(blur=0.1, rgb_noise=0.04, dropout_prob=0.01, occlusion_prob=0.08, occlusion_size=5),
                wind=0.05,
                regime="u_shape",
            ),
        }

    def preset(self, name: str) -> ScenarioConfig:
        presets = self.presets()
        if name not in presets:
            raise ValueError(f"Unknown scenario preset '{name}'.")
        return presets[name]

    def _make(self, index: int, deterministic: bool) -> ScenarioConfig:
        rng = random.Random(index) if deterministic else self._rng
        density_mode = rng.choice(["low", "medium", "high"])
        if density_mode == "low":
            obstacle_density = rng.betavariate(2, 8)
        elif density_mode == "high":
            obstacle_density = rng.betavariate(8, 2)
        else:
            obstacle_density = rng.betavariate(3, 3)
        return ScenarioConfig(
            corridor_width=rng.uniform(1.0, 5.0),
            obstacle_density=obstacle_density,
            obstacle_distribution=rng.choice(["uniform", "clustered", "u_shape"]),
            obstacle_motion=rng.choice(["static", "drift", "oscillate"]),
            texture_richness=rng.uniform(0.0, 1.0),
            lighting=rng.uniform(0.5, 1.5),
            noise=NoiseConfig(
                blur=rng.uniform(0.0, 0.6),
                rgb_noise=rng.uniform(0.0, 0.2),
                dropout_prob=rng.uniform(0.0, 0.05),
                occlusion_prob=rng.uniform(0.0, 0.2),
                occlusion_size=rng.randint(2, 8),
            ),
            wind=rng.uniform(0.0, 0.5),
            regime=rng.choice(["corridor", "clutter", "dynamic_traps", "u_shape"]),
        )
