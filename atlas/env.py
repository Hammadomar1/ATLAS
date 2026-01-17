from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from atlas.scenarios import NoiseConfig, ScenarioConfig


@dataclass
class EnvState:
    position: float
    velocity: float
    tick: int
    obstacles: list["Obstacle"]


@dataclass
class Obstacle:
    position: float
    velocity: float
    phase: float
    motion: str


class SimpleCorridorEnv:
    """Minimal Gym-like environment producing RGB frames."""

    def __init__(self, width: int = 32, height: int = 24) -> None:
        self._width = width
        self._height = height
        self._rng = random.Random(0)
        self._scenario: ScenarioConfig | None = None
        self._state = EnvState(position=0.0, velocity=0.0, tick=0, obstacles=[])

    def reset(self, seed: int, scenario_cfg: ScenarioConfig) -> Sequence[Sequence[Sequence[int]]]:
        self._rng = random.Random(seed)
        self._scenario = scenario_cfg
        obstacles = self._spawn_obstacles(scenario_cfg)
        self._state = EnvState(position=0.0, velocity=0.0, tick=0, obstacles=obstacles)
        return self._render()

    def step(self, action: float) -> tuple[Sequence[Sequence[Sequence[int]]], float, bool, dict[str, Any]]:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before stepping.")
        position = self._state.position + action * 0.1
        velocity = action
        obstacles = self._update_obstacles(self._state.obstacles)
        self._state = EnvState(position=position, velocity=velocity, tick=self._state.tick + 1, obstacles=obstacles)
        collision = abs(position) > self._scenario.corridor_width
        collision = collision or any(abs(position - obs.position) < 0.1 for obs in obstacles)
        near_miss = max(0.0, abs(position) - self._scenario.corridor_width * 0.8)
        info = {
            "collision": collision,
            "near_miss": near_miss,
            "distance_to_obstacle": max(0.0, self._scenario.corridor_width - abs(position)),
            "progress": self._state.tick,
            "velocity": velocity,
            "obstacles": [obs.position for obs in obstacles],
        }
        reward = 1.0 - near_miss
        done = collision
        return self._render(), reward, done, info

    def downsample(self, frame: Sequence[Sequence[Sequence[int]]], factor: int = 2) -> Sequence[Sequence[Sequence[int]]]:
        return frame[::factor]

    def _render(self) -> Sequence[Sequence[Sequence[int]]]:
        scenario = self._scenario
        noise = scenario.noise.rgb_noise if scenario else 0.0
        lighting = scenario.lighting if scenario else 1.0
        texture = scenario.texture_richness if scenario else 0.0
        frame = []
        for _ in range(self._height):
            row = []
            for _ in range(self._width):
                base = int(128 + self._state.position * 10)
                base = int(base * lighting + texture * self._rng.uniform(-30, 30))
                pixel = [
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                    self._clamp(base + int(self._rng.uniform(-noise, noise) * 255)),
                ]
                row.append(pixel)
            frame.append(row)
        if scenario:
            frame = self._apply_occlusion(frame, scenario.noise)
            frame = self._apply_dropout(frame, scenario.noise)
            if scenario.noise.blur > 0.0:
                frame = self._blur(frame)
        return frame

    @staticmethod
    def _clamp(value: int) -> int:
        return max(0, min(255, value))

    def _spawn_obstacles(self, scenario: ScenarioConfig) -> list[Obstacle]:
        count = int(self._width * scenario.obstacle_density)
        if count <= 0:
            return []
        positions: list[float]
        if scenario.obstacle_distribution == "clustered":
            centers = [self._rng.uniform(-scenario.corridor_width, scenario.corridor_width) for _ in range(2)]
            positions = [
                self._rng.choice(centers) + self._rng.uniform(-0.3, 0.3) for _ in range(count)
            ]
        elif scenario.obstacle_distribution == "u_shape":
            edge = scenario.corridor_width * 0.7
            positions = [
                self._rng.choice([-edge, edge]) + self._rng.uniform(-0.2, 0.2) for _ in range(count)
            ]
        else:
            positions = [self._rng.uniform(-scenario.corridor_width, scenario.corridor_width) for _ in range(count)]
        return [
            Obstacle(
                position=pos,
                velocity=self._rng.uniform(-0.02, 0.02),
                phase=self._rng.uniform(0.0, 6.28),
                motion=scenario.obstacle_motion,
            )
            for pos in positions
        ]

    def _update_obstacles(self, obstacles: list[Obstacle]) -> list[Obstacle]:
        scenario = self._scenario
        if not scenario:
            return obstacles
        moved = []
        for obs in obstacles:
            if obs.motion == "static":
                moved.append(obs)
                continue
            if obs.motion == "oscillate":
                phase = obs.phase + 0.2
                position = obs.position + 0.05 * math.sin(phase)
                moved.append(Obstacle(position=position, velocity=obs.velocity, phase=phase, motion=obs.motion))
                continue
            position = obs.position + obs.velocity + self._rng.uniform(-0.02, 0.02)
            moved.append(Obstacle(position=position, velocity=obs.velocity, phase=obs.phase, motion=obs.motion))
        return moved

    def _blur(self, frame: Sequence[Sequence[Sequence[int]]]) -> Sequence[Sequence[Sequence[int]]]:
        blurred = []
        for row in frame:
            blurred_row = []
            for pixel in row:
                avg = int(sum(pixel) / 3)
                blurred_row.append([avg, avg, avg])
            blurred.append(blurred_row)
        return blurred

    def _apply_dropout(
        self,
        frame: Sequence[Sequence[Sequence[int]]],
        noise: NoiseConfig,
    ) -> Sequence[Sequence[Sequence[int]]]:
        dropout_prob = noise.dropout_prob
        if dropout_prob <= 0.0:
            return frame
        dropped = []
        for row in frame:
            new_row = []
            for pixel in row:
                if self._rng.random() < dropout_prob:
                    new_row.append([0, 0, 0])
                else:
                    new_row.append(pixel)
            dropped.append(new_row)
        return dropped

    def _apply_occlusion(
        self,
        frame: Sequence[Sequence[Sequence[int]]],
        noise: NoiseConfig,
    ) -> Sequence[Sequence[Sequence[int]]]:
        occlusion_prob = noise.occlusion_prob
        if occlusion_prob <= 0.0 or self._rng.random() > occlusion_prob:
            return frame
        size = max(1, noise.occlusion_size)
        start_x = self._rng.randrange(0, max(1, self._width - size))
        start_y = self._rng.randrange(0, max(1, self._height - size))
        masked = []
        for y, row in enumerate(frame):
            new_row = []
            for x, pixel in enumerate(row):
                if start_x <= x < start_x + size and start_y <= y < start_y + size:
                    new_row.append([0, 0, 0])
                else:
                    new_row.append(pixel)
            masked.append(new_row)
        return masked
