from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from atlas.env import SimpleCorridorEnv
from atlas.meta_queue import MetaPacketQueue
from atlas.types import MetaPacket


@dataclass(frozen=True)
class SchedulerConfig:
    reflex_period_s: float
    meta_period_ticks: int


class DualTimeScheduler:
    """Runs reflex ticks every env step and meta ticks at a slower cadence."""

    def __init__(
        self,
        env: SimpleCorridorEnv,
        system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
        meta_queue: MetaPacketQueue,
        meta_policy: Callable[[int], MetaPacket],
        config: SchedulerConfig,
    ) -> None:
        self._env = env
        self._system_step = system_step
        self._meta_queue = meta_queue
        self._meta_policy = meta_policy
        self._config = config

    def run(self, ticks: int) -> None:
        action = 0.0
        for t in range(ticks):
            start = time.perf_counter()
            if t % self._config.meta_period_ticks == 0:
                packet = self._meta_policy(t)
                self._meta_queue.enqueue(packet, t)
            arrivals = self._meta_queue.deliver_due(t)
            for arrival in arrivals:
                self._system_step(t, self._env.step(action)[0])
            frame, _, done, _ = self._env.step(action)
            action = self._system_step(t, frame)
            elapsed = time.perf_counter() - start
            if elapsed < self._config.reflex_period_s:
                time.sleep(self._config.reflex_period_s - elapsed)
            if done:
                break
