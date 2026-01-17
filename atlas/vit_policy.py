from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@dataclass(frozen=True)
class ViTConfig:
    model_name: str
    theta_dim: int


class ViTMetaPolicyModel:
    def __init__(self, model_name: str, theta_dim: int) -> None:
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._theta_dim = theta_dim
        self._projection: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def infer_delta(
        self, frame: Sequence[Sequence[Sequence[int]]], telemetry: Mapping[str, float]
    ) -> Sequence[float]:
        image = Image.fromarray(np.array(frame, dtype=np.uint8))
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        if self._projection is not None:
            delta = self._projection.dot(embedding)
            if self._bias is not None:
                delta = delta + self._bias
        else:
            novelty = telemetry.get("novelty", 0.0)
            delta = np.full(
                (self._theta_dim,),
                float(embedding.mean()) * 1e-6 + novelty * 1e-3,
            )
        return delta.tolist()

    def load_weights(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        projection = payload.get("projection")
        bias = payload.get("bias")
        if projection is not None:
            self._projection = np.array(projection, dtype=float)
        if bias is not None:
            self._bias = np.array(bias, dtype=float)
