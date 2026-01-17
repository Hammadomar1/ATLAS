from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from atlas.types import MetricsRecord, Telemetry


@dataclass
class RunManifest:
    run_id: str
    seed: int
    config: dict[str, Any]
    git_hash: str
    platform: dict[str, Any] = field(default_factory=dict)


class EpisodeLogger:
    """Collects telemetry/metrics and writes compressed episode logs."""

    def __init__(self, output_dir: Path, run_manifest: RunManifest) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_manifest = run_manifest
        self._telemetry: List[Telemetry] = []
        self._metrics: List[MetricsRecord] = []
        self._write_manifest()

    def record_telemetry(self, telemetry: Telemetry) -> None:
        self._telemetry.append(telemetry)

    def record_metrics(self, metrics: MetricsRecord) -> None:
        self._metrics.append(metrics)

    def flush_episode(self, episode_id: int) -> Path:
        path = self._output_dir / f"episode_{episode_id:04d}.npz"
        telemetry_payload = np.array([asdict(item) for item in self._telemetry], dtype=object)
        metrics_payload = np.array([asdict(item) for item in self._metrics], dtype=object)
        np.savez_compressed(path, telemetry=telemetry_payload, metrics=metrics_payload)
        self._telemetry.clear()
        self._metrics.clear()
        return path

    def _write_manifest(self) -> None:
        manifest_path = self._output_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(asdict(self._run_manifest), indent=2))


def resolve_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"

