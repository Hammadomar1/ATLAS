from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import json


@dataclass(frozen=True)
class MetaEmbedding:
    values: Sequence[float]


class MetaPolicyModel(Protocol):
    def infer_delta(
        self, frame: Sequence[Sequence[Sequence[int]]], telemetry: Mapping[str, float]
    ) -> Sequence[float]:
        ...

    def load_weights(self, path: Path) -> None:
        ...


class ViTStub:
    def embed(self, frame: Sequence[Sequence[Sequence[int]]]) -> MetaEmbedding:
        total = sum(sum(sum(pixel) for pixel in row) for row in frame) if frame else 0.0
        mean = total / max(1.0, sum(len(row) for row in frame)) if frame else 0.0
        return MetaEmbedding(values=[total, mean])


class RouterStub:
    def route(self, embedding: MetaEmbedding) -> Sequence[float]:
        base = embedding.values[1] if len(embedding.values) > 1 else 0.0
        return [1.0, min(1.0, base / 255.0), 0.5, 0.5]


class ExpertStub:
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def forward(self, embedding: MetaEmbedding) -> float:
        return self._scale * (embedding.values[0] if embedding.values else 0.0)


class ModulationPolicy:
    def __init__(self, scale: float, theta_dim: int) -> None:
        self._scale = scale
        self._theta_dim = theta_dim

    def compute_delta(self, expert_outputs: Sequence[float]) -> Sequence[float]:
        base = self._scale * sum(expert_outputs)
        return [base for _ in range(self._theta_dim)]


class MetaCognitionModel:
    def __init__(self, scale: float, theta_dim: int) -> None:
        self._vit = ViTStub()
        self._router = RouterStub()
        self._expert_scales = [1e-6, 1e-6, 1e-6, 1e-6]
        self._experts = [ExpertStub(scale=scale) for scale in self._expert_scales]
        self._policy = ModulationPolicy(scale=scale, theta_dim=theta_dim)

    def infer_delta(
        self, frame: Sequence[Sequence[Sequence[int]]], telemetry: Mapping[str, float]
    ) -> Sequence[float]:
        embedding = self._vit.embed(frame)
        weights = self._router.route(embedding)
        outputs = []
        for weight, expert in zip(weights, self._experts):
            outputs.append(weight * expert.forward(embedding))
        telemetry_bias = telemetry.get("novelty", 0.0)
        outputs.append(telemetry_bias)
        return self._policy.compute_delta(outputs)

    def load_weights(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        scales = payload.get("expert_scales", self._expert_scales)
        if isinstance(scales, list) and len(scales) == len(self._experts):
            self._experts = [ExpertStub(scale=float(scale)) for scale in scales]

