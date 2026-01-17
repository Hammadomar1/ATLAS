from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np

from atlas.env import SimpleCorridorEnv
from atlas.types import Telemetry


@dataclass
class RolloutSample:
    frame: Sequence[Sequence[Sequence[int]]]
    telemetry: Telemetry
    reward: float
    done: bool
    info: dict[str, Any]


class RolloutCollector:
    def __init__(self, env: SimpleCorridorEnv) -> None:
        self._env = env

    def collect(
        self,
        policy: Callable[[Sequence[Sequence[Sequence[int]]]], float],
        telemetry_provider: Callable[[], Telemetry],
        steps: int,
    ) -> List[RolloutSample]:
        samples: List[RolloutSample] = []
        frame = self._env.step(0.0)[0]
        for _ in range(steps):
            action = policy(frame)
            frame, reward, done, info = self._env.step(action)
            telemetry = telemetry_provider()
            samples.append(
                RolloutSample(
                    frame=frame,
                    telemetry=telemetry,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )
            if done:
                break
        return samples

    def export(self, samples: List[RolloutSample], path: str) -> None:
        payload = np.array(
            [
                {
                    "frame": sample.frame,
                    "telemetry": sample.telemetry,
                    "reward": sample.reward,
                    "done": sample.done,
                    "info": sample.info,
                }
                for sample in samples
            ],
            dtype=object,
        )
        np.savez_compressed(path, samples=payload)
