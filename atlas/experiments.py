from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from atlas.env import SimpleCorridorEnv
from atlas.meta_queue import MetaPacketQueue
from atlas.scheduler import DualTimeScheduler, SchedulerConfig
from atlas.types import MetaPacket


@dataclass(frozen=True)
class ExperimentResult:
    tick_times: Sequence[float]
    deadline_miss_rate: float
    max_tick_time: float
    mean_tick_time: float
    jitter: float


def run_latency_experiment(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    meta_queue: MetaPacketQueue,
    meta_policy: Callable[[int], MetaPacket],
    config: SchedulerConfig,
    ticks: int,
) -> ExperimentResult:
    tick_times = []
    action = 0.0
    for t in range(ticks):
        start = time.perf_counter()
        if t % config.meta_period_ticks == 0:
            meta_queue.enqueue(meta_policy(t), t)
        _ = meta_queue.deliver_due(t)
        frame, _, done, _ = env.step(action)
        action = system_step(t, frame)
        elapsed = time.perf_counter() - start
        tick_times.append(elapsed)
        if done:
            break
    max_tick = max(tick_times) if tick_times else 0.0
    mean_tick = statistics.mean(tick_times) if tick_times else 0.0
    jitter = statistics.pstdev(tick_times) if len(tick_times) > 1 else 0.0
    deadline_miss_rate = sum(1 for t in tick_times if t >= config.reflex_period_s) / max(1, len(tick_times))
    return ExperimentResult(
        tick_times=tick_times,
        deadline_miss_rate=deadline_miss_rate,
        max_tick_time=max_tick,
        mean_tick_time=mean_tick,
        jitter=jitter,
    )
