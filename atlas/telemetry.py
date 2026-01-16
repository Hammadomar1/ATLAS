from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import List

from atlas.types import MetricsRecord, Telemetry


class TelemetryPublisher:
    """Non-blocking telemetry publisher."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Telemetry]" = queue.Queue()

    def publish(self, telemetry: Telemetry) -> None:
        try:
            self._queue.put_nowait(telemetry)
        except queue.Full:
            return

    def drain(self) -> List[Telemetry]:
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items


class MetricsLogger:
    """Non-blocking metrics collector."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[MetricsRecord]" = queue.Queue()

    def record(self, record: MetricsRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            return

    def drain(self) -> List[MetricsRecord]:
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items
