from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence
import random

from atlas.types import ObserverOutput, ProtoEvent


@dataclass(frozen=True)
class QuantizerConfig:
    bins: int
    min_value: float
    max_value: float


@dataclass(frozen=True)
class HNSPConfig:
    memory_dim: int
    binary_mode: bool
    seed: int
    trajectory_length: int
    tm_length: int
    prototype_count: int
    x_quantizer: QuantizerConfig
    w_quantizer: QuantizerConfig
    s_quantizer: QuantizerConfig
    x_dot_quantizer: QuantizerConfig
    rho_quantizer: QuantizerConfig
    novelty_threshold_low: float
    novelty_threshold_high: float
    prototype_update_threshold: float


class HNSPObserver(threading.Thread):
    """Asynchronous observer that encodes events into a hypervector trajectory."""

    def __init__(self, config: HNSPConfig) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._events: "queue.Queue[List[ProtoEvent]]" = queue.Queue()
        self._encoder = _HNSPEncoder(config)
        self._latest = ObserverOutput(
            memory=[0.0 for _ in range(config.memory_dim)],
            novelty=0.0,
            soft_veto=0.0,
        )
        self._stop_event = threading.Event()
        self._seq = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                events = self._events.get(timeout=0.01)
            except queue.Empty:
                continue
            self._latest = self._encoder.step(events)
            self._seq += 1

    def submit_events(self, events: List[ProtoEvent]) -> None:
        try:
            self._events.put_nowait(events)
        except queue.Full:
            return

    def latest(self) -> ObserverOutput:
        return self._latest

    def latest_with_seq(self) -> tuple[ObserverOutput, int]:
        return self._latest, self._seq

    def stop(self) -> None:
        self._stop_event.set()


class _HNSPEncoder:
    def __init__(self, config: HNSPConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._tie_breaker = self._random_vector()
        self._roles = {
            "pos": self._random_vector(),
            "wid": self._random_vector(),
            "str": self._random_vector(),
            "vel": self._random_vector(),
            "vis": self._random_vector(),
        }
        self._codebooks = {
            "x": self._build_codebook(config.x_quantizer),
            "w": self._build_codebook(config.w_quantizer),
            "s": self._build_codebook(config.s_quantizer),
            "x_dot": self._build_codebook(config.x_dot_quantizer),
            "rho": self._build_codebook(config.rho_quantizer),
        }
        self._perm_base = self._build_permutation()
        self._perm_powers = self._build_permutation_powers(config.trajectory_length)
        self._frame_buffer: Deque[List[float]] = deque(maxlen=config.trajectory_length)
        self._tm: Deque[List[float]] = deque(maxlen=config.tm_length)
        self._prototypes: List[List[float]] = []

    def step(self, events: Sequence[ProtoEvent]) -> ObserverOutput:
        frame_vector = self._encode_frame(events)
        self._frame_buffer.appendleft(frame_vector)
        trajectory = self._encode_trajectory()
        self._tm.append(trajectory)
        novelty = self._update_prototypes(trajectory)
        soft_veto = self._soft_veto(novelty)
        return ObserverOutput(memory=list(trajectory), novelty=novelty, soft_veto=soft_veto)

    def _encode_frame(self, events: Sequence[ProtoEvent]) -> List[float]:
        if not events:
            return self._zero_vector()
        event_vectors = [self._encode_event(event) for event in events]
        return self._bundle(event_vectors)

    def _encode_event(self, event: ProtoEvent) -> List[float]:
        fillers = {
            "pos": self._codebooks["x"][self._quantize(event.bearing_index, self._config.x_quantizer)],
            "wid": self._codebooks["w"][self._quantize(event.width, self._config.w_quantizer)],
            "str": self._codebooks["s"][self._quantize(event.strength, self._config.s_quantizer)],
            "vel": self._codebooks["x_dot"][self._quantize(event.velocity, self._config.x_dot_quantizer)],
            "vis": self._codebooks["rho"][self._quantize(event.visibility, self._config.rho_quantizer)],
        }
        bound = [self._bind(self._roles[key], fillers[key]) for key in fillers]
        return self._bundle(bound)

    def _encode_trajectory(self) -> List[float]:
        if not self._frame_buffer:
            return self._zero_vector()
        permuted = []
        for j, frame in enumerate(self._frame_buffer):
            permuted.append(self._permute(frame, self._perm_powers[j]))
        return self._bundle(permuted)

    def _update_prototypes(self, trajectory: List[float]) -> float:
        if not self._prototypes:
            self._prototypes.append(trajectory)
            return 1.0
        sims = [self._similarity(trajectory, proto) for proto in self._prototypes]
        max_sim = max(sims)
        best_idx = sims.index(max_sim)
        if max_sim >= self._config.prototype_update_threshold:
            self._prototypes[best_idx] = self._bundle([self._prototypes[best_idx], trajectory])
        elif len(self._prototypes) < self._config.prototype_count:
            self._prototypes.append(trajectory)
        else:
            worst_idx = sims.index(min(sims))
            self._prototypes[worst_idx] = trajectory
        novelty = 1.0 - max_sim
        return max(0.0, min(1.0, novelty))

    def _soft_veto(self, novelty: float) -> float:
        if novelty <= self._config.novelty_threshold_low:
            return 0.0
        if novelty >= self._config.novelty_threshold_high:
            return 1.0
        span = self._config.novelty_threshold_high - self._config.novelty_threshold_low
        if span <= 0:
            return 1.0
        return (novelty - self._config.novelty_threshold_low) / span

    def _build_codebook(self, quantizer: QuantizerConfig) -> List[List[float]]:
        return [self._random_vector() for _ in range(quantizer.bins)]

    def _build_permutation(self) -> List[int]:
        indices = list(range(self._config.memory_dim))
        self._rng.shuffle(indices)
        return indices

    def _build_permutation_powers(self, length: int) -> List[List[int]]:
        powers = [list(range(self._config.memory_dim))]
        for _ in range(1, max(1, length)):
            prev = powers[-1]
            powers.append([self._perm_base[i] for i in prev])
        return powers

    def _permute(self, vector: Sequence[float], indices: Sequence[int]) -> List[float]:
        return [vector[i] for i in indices]

    def _quantize(self, value: float, config: QuantizerConfig) -> int:
        if config.bins <= 1:
            return 0
        min_value = min(config.min_value, config.max_value)
        max_value = max(config.min_value, config.max_value)
        clamped = max(min_value, min(max_value, float(value)))
        if max_value == min_value:
            return 0
        ratio = (clamped - min_value) / (max_value - min_value)
        index = int(ratio * config.bins)
        return max(0, min(config.bins - 1, index))

    def _random_vector(self) -> List[float]:
        if self._config.binary_mode:
            return [1.0 if self._rng.random() >= 0.5 else 0.0 for _ in range(self._config.memory_dim)]
        return [1.0 if self._rng.random() >= 0.5 else -1.0 for _ in range(self._config.memory_dim)]

    def _zero_vector(self) -> List[float]:
        return [0.0 for _ in range(self._config.memory_dim)]

    def _bind(self, left: Sequence[float], right: Sequence[float]) -> List[float]:
        if self._config.binary_mode:
            return [1.0 if int(l) ^ int(r) else 0.0 for l, r in zip(left, right)]
        return [l * r for l, r in zip(left, right)]

    def _bundle(self, vectors: Iterable[Sequence[float]]) -> List[float]:
        vectors_list = list(vectors)
        if not vectors_list:
            return self._zero_vector()
        dim = self._config.memory_dim
        sums = [0.0 for _ in range(dim)]
        for vector in vectors_list:
            for idx, value in enumerate(vector):
                sums[idx] += value
        bundled = []
        if self._config.binary_mode:
            half = len(vectors_list) / 2
            for idx, total in enumerate(sums):
                if total > half:
                    bundled.append(1.0)
                elif total < half:
                    bundled.append(0.0)
                else:
                    bundled.append(self._tie_breaker[idx])
        else:
            for idx, total in enumerate(sums):
                if total > 0:
                    bundled.append(1.0)
                elif total < 0:
                    bundled.append(-1.0)
                else:
                    bundled.append(self._tie_breaker[idx])
        return bundled

    def _similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        if self._config.binary_mode:
            matches = sum(1 for l, r in zip(left, right) if int(l) == int(r))
            return matches / self._config.memory_dim
        dot = sum(l * r for l, r in zip(left, right))
        return (dot / self._config.memory_dim + 1.0) / 2.0
