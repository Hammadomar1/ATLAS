from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from atlas.observer import HNSPObserver
from atlas.offboard import MetaCognitionServer
from atlas.parameter_server import ParameterServer
from atlas.reflex_engine import ReflexEngine
from atlas.safety_shield import SafetyShield
from atlas.supervisor import Supervisor
from atlas.telemetry import MetricsLogger, TelemetryPublisher
from atlas.types import MetricsRecord, ReflexInput, Telemetry


@dataclass
class SystemConfig:
    reflex_period_s: float
    meta_period_ticks: int


class AtlasSystem:
    """Coordinates onboard reflex loop and asynchronous advisory branches."""

    def __init__(
        self,
        config: SystemConfig,
        reflex_engine: ReflexEngine,
        observer: HNSPObserver,
        supervisor: Supervisor,
        safety_shield: SafetyShield,
        parameter_server: ParameterServer,
        meta_server: MetaCognitionServer,
        telemetry: TelemetryPublisher,
        metrics: MetricsLogger,
    ) -> None:
        self._config = config
        self._reflex_engine = reflex_engine
        self._observer = observer
        self._supervisor = supervisor
        self._safety_shield = safety_shield
        self._parameter_server = parameter_server
        self._meta_server = meta_server
        self._telemetry = telemetry
        self._metrics = metrics
        self._last_winner = 0

    def start(self) -> None:
        self._observer.start()
        self._meta_server.start()

    def stop(self) -> None:
        self._observer.stop()
        self._meta_server.stop()

    def reflex_tick(self, t: int, frame: Sequence[Sequence[Sequence[float]]]) -> float:
        start = time.perf_counter()
        reflex_input = ReflexInput(t=t, frame=frame)
        reflex_output = self._reflex_engine.step(reflex_input)
        self._observer.submit_events(reflex_output.events)

        observer_output = self._observer.latest()
        hazard = max(reflex_output.drive.drive) if reflex_output.drive.drive else 0.0
        oscillation = abs(reflex_output.dnf_state.winner_index - self._last_winner) / max(
            1e-6, self._config.reflex_period_s
        )
        self._last_winner = reflex_output.dnf_state.winner_index

        supervisor_state = self._supervisor.update(
            t=t,
            hazard=hazard,
            oscillation=oscillation,
            novelty=observer_output.novelty,
        )

        arrivals = self._meta_server.poll_arrivals(t)
        for arrival in arrivals:
            if supervisor_state.mode != "GUARDED" and (supervisor_state.meta_ok or supervisor_state.meta_age is None):
                self._parameter_server.apply_delta(arrival.delta_theta)
                self._supervisor.observe_meta(t)

        if supervisor_state.guard_enabled:
            soft_veto = observer_output.soft_veto
            safe_action = 0.0
            pre_action = (1.0 - soft_veto) * reflex_output.nominal_action + soft_veto * safe_action
        else:
            pre_action = reflex_output.nominal_action

        shield_output = self._safety_shield.apply(pre_action, reflex_output.drive.drive)

        downsampled = frame[::2]
        self._telemetry.publish(
            Telemetry(
                t=t,
                downsampled_frame=downsampled,
                nominal_action=reflex_output.nominal_action,
                shielded_action=shield_output.action,
                hazard=hazard,
                oscillation=oscillation,
                meta_age=supervisor_state.meta_age,
            )
        )

        duration = time.perf_counter() - start
        self._metrics.record(
            MetricsRecord(
                t=t,
                reflex_duration_s=duration,
                hazard=hazard,
                oscillation=oscillation,
                shield_used_fallback=shield_output.used_fallback,
                meta_age=supervisor_state.meta_age,
            )
        )
        return shield_output.action

    def maybe_emit_meta(self, t: int) -> None:
        if t % self._config.meta_period_ticks != 0:
            return
        packet = self._meta_server.synthesize_update(k=t // self._config.meta_period_ticks, created_at_t=t)
        self._meta_server.submit(packet)
