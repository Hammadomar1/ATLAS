from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

from atlas.types import ObserverOutput, ProtoEvent


@dataclass
class HNSPConfig:
    memory_dim: int
    novelty_threshold: float


class HNSPObserver(threading.Thread):
    """Asynchronous observer with hold-last semantics."""

    def __init__(self, config: HNSPConfig) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._events: "queue.Queue[List[ProtoEvent]]" = queue.Queue()
        self._latest = ObserverOutput(memory=[0.0] * config.memory_dim, novelty=0.0, soft_veto=0.0)
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                events = self._events.get(timeout=0.01)
            except queue.Empty:
                continue
            novelty = 1.0 if any(event.strength > self._config.novelty_threshold for event in events) else 0.0
            memory = [float(len(events)) % 1.0 for _ in range(self._config.memory_dim)]
            soft_veto = min(1.0, novelty)
            self._latest = ObserverOutput(memory=memory, novelty=novelty, soft_veto=soft_veto)

    def submit_events(self, events: List[ProtoEvent]) -> None:
        try:
            self._events.put_nowait(events)
        except queue.Full:
            return

    def latest(self) -> ObserverOutput:
        return self._latest

    def stop(self) -> None:
        self._stop_event.set()
