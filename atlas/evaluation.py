from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from atlas.env import SimpleCorridorEnv
from atlas.experiments import ExperimentResult, run_latency_experiment
from atlas.meta_queue import MetaPacketQueue
from atlas.scheduler import SchedulerConfig
from atlas.types import MetaPacket


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    use_hnsp: bool
    use_meta: bool
    use_shield: bool


@dataclass(frozen=True)
class EvaluationResult:
    baseline: BaselineConfig
    latency: ExperimentResult
    collisions: int
    steps: int


def run_e1_determinism_latency(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    meta_queue: MetaPacketQueue,
    meta_policy: Callable[[int], MetaPacket],
    config: SchedulerConfig,
    ticks: int,
) -> ExperimentResult:
    return run_latency_experiment(env, system_step, meta_queue, meta_policy, config, ticks)


def run_e2_non_blocking(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    ticks: int,
) -> int:
    action = 0.0
    for t in range(ticks):
        frame, _, done, _ = env.step(action)
        action = system_step(t, frame)
        if done:
            return t + 1
    return ticks


def run_e4_delay_dropout(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    meta_queue: MetaPacketQueue,
    meta_policy: Callable[[int], MetaPacket],
    config: SchedulerConfig,
    ticks: int,
) -> int:
    action = 0.0
    collisions = 0
    for t in range(ticks):
        if t % config.meta_period_ticks == 0:
            meta_queue.enqueue(meta_policy(t), t)
        _ = meta_queue.deliver_due(t)
        frame, _, done, info = env.step(action)
        action = system_step(t, frame)
        if info.get("collision"):
            collisions += 1
        if done:
            break
    return collisions


def run_e5_oscillation(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    ticks: int,
) -> int:
    action = 0.0
    flips = 0
    last_action = 0.0
    for t in range(ticks):
        frame, _, done, _ = env.step(action)
        action = system_step(t, frame)
        if t > 0 and (action >= 0) != (last_action >= 0):
            flips += 1
        last_action = action
        if done:
            break
    return flips


def run_e6_regime_shift(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    ticks: int,
) -> int:
    action = 0.0
    collisions = 0
    for t in range(ticks):
        frame, _, done, info = env.step(action)
        action = system_step(t, frame)
        if info.get("collision"):
            collisions += 1
        if done:
            break
    return collisions

def run_e3_shield_effectiveness(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    ticks: int,
) -> tuple[int, int]:
    action = 0.0
    collisions = 0
    for t in range(ticks):
        frame, _, done, info = env.step(action)
        action = system_step(t, frame)
        if info.get("collision"):
            collisions += 1
        if done:
            break
    return collisions, t + 1


def run_baseline_suite(
    baseline: BaselineConfig,
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    meta_queue: MetaPacketQueue,
    meta_policy: Callable[[int], MetaPacket],
    config: SchedulerConfig,
    ticks: int,
) -> EvaluationResult:
    latency = run_e1_determinism_latency(env, system_step, meta_queue, meta_policy, config, ticks)
    collisions, steps = run_e3_shield_effectiveness(env, system_step, ticks)
    return EvaluationResult(baseline=baseline, latency=latency, collisions=collisions, steps=steps)
