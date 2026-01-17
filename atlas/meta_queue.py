from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from atlas.types import MetaArrival, MetaPacket


@dataclass(frozen=True)
class DelayDropoutConfig:
    delay_distribution: str
    delay_max_ticks: int
    dropout_prob: float
    burst_length: int = 0
    policy: str = "latest"


class MetaPacketQueue:
    """Deterministic queue of meta packets with delay/dropout."""

    def __init__(self, config: DelayDropoutConfig, seed: int) -> None:
        self._config = config
        self._rng = random.Random(seed)
        self._queue: List[tuple[int, MetaPacket]] = []
        self._dropout_remaining = 0

    def enqueue(self, packet: MetaPacket, now: int) -> None:
        if self._should_drop():
            return
        delay = self._sample_delay()
        arrival = MetaArrival(
            t=now + delay,
            delta_theta=packet.delta_theta,
            k=packet.k,
            created_at_t=packet.created_at_t,
        )
        heapq.heappush(self._queue, (arrival.t, packet))

    def deliver_due(self, t: int) -> List[MetaArrival]:
        due: List[MetaArrival] = []
        while self._queue and self._queue[0][0] <= t:
            _, packet = heapq.heappop(self._queue)
            due.append(
                MetaArrival(
                    t=t,
                    delta_theta=packet.delta_theta,
                    k=packet.k,
                    created_at_t=packet.created_at_t,
                )
            )
        if self._config.policy == "latest" and len(due) > 1:
            return [max(due, key=lambda arrival: arrival.created_at_t)]
        return due

    def _sample_delay(self) -> int:
        if self._config.delay_distribution == "fixed":
            return max(0, self._config.delay_max_ticks)
        if self._config.delay_distribution == "uniform":
            return self._rng.randint(0, max(0, self._config.delay_max_ticks))
        return 0

    def _should_drop(self) -> bool:
        if self._dropout_remaining > 0:
            self._dropout_remaining -= 1
            return True
        if self._config.burst_length > 0 and self._rng.random() < self._config.dropout_prob:
            self._dropout_remaining = self._config.burst_length
            return True
        return self._rng.random() < self._config.dropout_prob
