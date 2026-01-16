from __future__ import annotations

import random
from typing import Sequence

from atlas.observer import HNSPConfig, HNSPObserver
from atlas.offboard import MetaCognitionServer, MetaConfig
from atlas.parameter_server import ParameterBounds, ParameterServer
from atlas.reflex_engine import ReflexConfig, ReflexEngine
from atlas.safety_shield import SafetyShield, ShieldConfig
from atlas.supervisor import Supervisor, SupervisorConfig
from atlas.system import AtlasSystem, SystemConfig
from atlas.telemetry import MetricsLogger, TelemetryPublisher


def make_frame(width: int, height: int) -> Sequence[Sequence[Sequence[float]]]:
    return [[[random.random() for _ in range(3)] for _ in range(width)] for _ in range(height)]


def run_simulation(ticks: int = 50) -> None:
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
    observer = HNSPObserver(HNSPConfig(memory_dim=8, novelty_threshold=0.8))
    supervisor = Supervisor(
        SupervisorConfig(
            hazard_guard_threshold=1.0,
            oscillation_guard_threshold=3.0,
            meta_timeout_ticks=10,
            guard_dwell_ticks=5,
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
    system.stop()


if __name__ == "__main__":
    run_simulation()
