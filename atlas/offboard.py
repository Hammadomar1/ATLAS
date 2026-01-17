from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from atlas.meta_cognition import MetaCognitionModel, MetaPolicyModel
from atlas.types import MetaArrival, MetaPacket


@dataclass
class MetaConfig:
    theta_dim: int
    delay_max_ticks: int
    dropout_prob: float
    update_scale: float
    seed: int = 0
    policy_scale: float = 1e-6
    weights_path: Optional[Path] = None
    use_real_vit: bool = False
    vit_model_name: str = "google/vit-base-patch16-224"


class MetaCognitionServer(threading.Thread):
    """Asynchronous meta-cognition that emits delayed/dropped updates."""

    def __init__(self, config: MetaConfig) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._rng = random.Random(config.seed)
        self._model: MetaPolicyModel
        if config.use_real_vit:
            from atlas.vit_policy import ViTMetaPolicyModel

            self._model = ViTMetaPolicyModel(model_name=config.vit_model_name, theta_dim=config.theta_dim)
        else:
            self._model = MetaCognitionModel(scale=config.policy_scale, theta_dim=config.theta_dim)
        if config.weights_path is not None:
            self._model.load_weights(config.weights_path)
        self._latest_frame: Optional[Sequence[Sequence[Sequence[int]]]] = None
        self._latest_telemetry: Mapping[str, float] = {}
        self._inbound: "queue.Queue[MetaPacket]" = queue.Queue()
        self._outbound: "queue.Queue[MetaArrival]" = queue.Queue()
        self._stop_event = threading.Event()

    def submit(self, packet: MetaPacket) -> None:
        try:
            self._inbound.put_nowait(packet)
        except queue.Full:
            return

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self._inbound.get(timeout=0.01)
            except queue.Empty:
                continue
            if random.random() < self._config.dropout_prob:
                continue
            delay = random.randint(0, self._config.delay_max_ticks)
            arrival_time = packet.created_at_t + delay
            self._outbound.put(
                MetaArrival(
                    t=arrival_time,
                    delta_theta=packet.delta_theta,
                    k=packet.k,
                    created_at_t=packet.created_at_t,
                )
            )

    def poll_arrivals(self, t: int) -> list[MetaArrival]:
        arrivals = []
        while True:
            try:
                arrival = self._outbound.get_nowait()
            except queue.Empty:
                break
            if arrival.t <= t:
                arrivals.append(arrival)
            else:
                self._outbound.put(arrival)
                break
        return arrivals

    def stop(self) -> None:
        self._stop_event.set()

    def observe_frame(self, frame: Sequence[Sequence[Sequence[int]]], telemetry: Mapping[str, float]) -> None:
        self._latest_frame = frame
        self._latest_telemetry = telemetry

    def synthesize_update(self, k: int, created_at_t: int) -> MetaPacket:
        if self._latest_frame is not None:
            delta = list(self._model.infer_delta(self._latest_frame, self._latest_telemetry))
        else:
            delta = [
                self._rng.uniform(-self._config.update_scale, self._config.update_scale)
                for _ in range(self._config.theta_dim)
            ]
        return MetaPacket(k=k, delta_theta=delta, created_at_t=created_at_t)
