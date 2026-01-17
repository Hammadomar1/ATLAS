from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from atlas.config import RunConfig, load_config, set_determinism
from atlas.logging import EpisodeLogger, RunManifest, resolve_git_hash
from atlas.observer import HNSPConfig, HNSPObserver, QuantizerConfig
from atlas.offboard import MetaCognitionServer, MetaConfig
from atlas.parameter_server import ParameterBounds, ParameterServer
from atlas.reflex_engine import ReflexConfig, ReflexEngine
from atlas.safety_shield import SafetyShield, ShieldConfig
from atlas.supervisor import Supervisor, SupervisorConfig
from atlas.system import AtlasSystem, SystemConfig
from pathlib import Path

from atlas.logging import EpisodeLogger, RunManifest, resolve_git_hash
from atlas.telemetry import MetricsLogger, TelemetryPublisher


def make_frame(width: int, height: int) -> Sequence[Sequence[Sequence[float]]]:
    return [[[random.random() for _ in range(3)] for _ in range(width)] for _ in range(height)]


def run_simulation(ticks: int = 50, run_config: RunConfig | None = None) -> None:
    if run_config is None:
        run_config = RunConfig(seed=1, settings={"ticks": ticks})
    set_determinism(run_config.seed)
    ticks = int(run_config.settings.get("ticks", ticks))
    reflex_engine = ReflexEngine(
        ReflexConfig(
            bearings=11,
            rpe_gain=0.1,
            composer_gain=1.0,
            dnf_decay=0.1,
            dnf_gain=0.5,
            controller_gain=1.0,
        )
    )
    observer = HNSPObserver(
        HNSPConfig(
            memory_dim=8,
            binary_mode=True,
            seed=1,
            trajectory_length=4,
            tm_length=8,
            prototype_count=4,
            x_quantizer=QuantizerConfig(bins=8, min_value=0.0, max_value=10.0),
            w_quantizer=QuantizerConfig(bins=4, min_value=0.0, max_value=1.0),
            s_quantizer=QuantizerConfig(bins=4, min_value=0.0, max_value=1.0),
            x_dot_quantizer=QuantizerConfig(bins=4, min_value=-1.0, max_value=1.0),
            rho_quantizer=QuantizerConfig(bins=4, min_value=0.0, max_value=1.0),
            novelty_threshold_low=0.2,
            novelty_threshold_high=0.8,
            prototype_update_threshold=0.7,
        )
    )
    supervisor = Supervisor(
        SupervisorConfig(
            hazard_high=1.0,
            hazard_low=0.5,
            oscillation_high=3.0,
            novelty_high=0.7,
            meta_timeout_ticks=10,
        )
    )
    safety_shield = SafetyShield(ShieldConfig(hazard_threshold=0.7, halt_action=0.0))
    parameter_server = ParameterServer(
        initial_theta=[0.5, 0.5, 0.5],
        bounds=ParameterBounds(minimum=[0.0, 0.0, 0.0], maximum=[1.0, 1.0, 1.0]),
    )
    meta_server = MetaCognitionServer(
        MetaConfig(theta_dim=3, delay_max_ticks=5, dropout_prob=0.3, update_scale=0.1)
    )
    telemetry = TelemetryPublisher()
    metrics = MetricsLogger()
    output_dir = Path(run_config.settings.get("output_dir", "runs"))
    run_id = run_config.settings.get("run_id", "sim")
    logger = EpisodeLogger(
        output_dir,
        RunManifest(
            run_id=run_id,
            seed=run_config.seed,
            config=run_config.settings,
            git_hash=resolve_git_hash(),
        ),
    )

    system = AtlasSystem(
        SystemConfig(reflex_period_s=0.02, meta_period_ticks=5),
        reflex_engine,
        observer,
        supervisor,
        safety_shield,
        parameter_server,
        meta_server,
        telemetry,
        metrics,
    )

    system.start()
    for t in range(ticks):
        frame = make_frame(16, 12)
        system.maybe_emit_meta(t)
        system.reflex_tick(t, frame)
        for item in telemetry.drain():
            logger.record_telemetry(item)
        for item in metrics.drain():
            logger.record_metrics(item)
    system.stop()
    logger.flush_episode(episode_id=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a short ATLAS simulation.")
    parser.add_argument("--config", type=Path, default=Path("configs/run_config.json"))
    args = parser.parse_args()
    run_simulation(run_config=load_config(args.config))
