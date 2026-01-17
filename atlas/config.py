from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    seed: int
    settings: dict[str, Any]


def load_config(path: Path) -> RunConfig:
    payload = json.loads(path.read_text())
    seed = int(payload.get("seed", 0))
    settings = payload.get("settings", {})
    return RunConfig(seed=seed, settings=settings)


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
