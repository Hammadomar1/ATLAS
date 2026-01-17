from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MetaEmbedding:
    values: Sequence[float]


class ViTStub:
    def embed(self, frame: Sequence[Sequence[Sequence[int]]]) -> MetaEmbedding:
        total = sum(sum(sum(pixel) for pixel in row) for row in frame) if frame else 0.0
        return MetaEmbedding(values=[total])


class RouterStub:
    def route(self, embedding: MetaEmbedding) -> Sequence[float]:
        return [1.0]


class ExpertStub:
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def forward(self, embedding: MetaEmbedding) -> float:
        return self._scale * (embedding.values[0] if embedding.values else 0.0)


class ModulationPolicy:
    def __init__(self, scale: float, theta_dim: int) -> None:
        self._scale = scale
        self._theta_dim = theta_dim

    def compute_delta(self, embedding: MetaEmbedding) -> Sequence[float]:
        base = self._scale * (embedding.values[0] if embedding.values else 0.0)
        return [base for _ in range(self._theta_dim)]
