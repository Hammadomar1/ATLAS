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
        self._lambda_hold = 0.0
        self._observer_seq = 0

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

        observer_output, observer_seq = self._observer.latest_with_seq()
        hazard = max(reflex_output.drive.drive) if reflex_output.drive.drive else 0.0
        oscillation = abs(reflex_output.dnf_state.winner_index - self._last_winner) / max(
            1e-6, self._config.reflex_period_s
        )
        self._last_winner = reflex_output.dnf_state.winner_index
        safe_action, safe_set_empty, _, safe_set_size = self._safety_shield.safe_action(
            reflex_output.nominal_action, reflex_output.drive.drive
        )

        arrivals = self._meta_server.poll_arrivals(t)
        meta_arrived = bool(arrivals)
        supervisor_state = self._supervisor.update(
            t=t,
            hazard=hazard,
            oscillation=oscillation,
            novelty=observer_output.novelty,
            safe_set_empty=safe_set_empty,
            meta_arrived=meta_arrived,
        )
        for arrival in arrivals:
            if supervisor_state.accept_meta:
                self._parameter_server.apply_delta(arrival.delta_theta)
                self._supervisor.observe_meta(t)

        if observer_seq != self._observer_seq:
            self._lambda_hold = observer_output.soft_veto
            self._observer_seq = observer_seq

        if supervisor_state.guard_enabled:
            pre_action = (1.0 - self._lambda_hold) * reflex_output.nominal_action + self._lambda_hold * safe_action
        elif supervisor_state.mode == "Emergency":
            pre_action = self._safety_shield._config.halt_action
        else:
            pre_action = reflex_output.nominal_action

        shield_output = self._safety_shield.apply(pre_action, reflex_output.drive.drive)

        downsampled = frame[::2]
        self._meta_server.observe_frame(downsampled)
        self._telemetry.publish(
            Telemetry(
                t=t,
                downsampled_frame=downsampled,
                nominal_action=reflex_output.nominal_action,
                pre_action=pre_action,
                shielded_action=shield_output.action,
                hazard=hazard,
                kappa=self._safety_shield._config.hazard_threshold,
                safe_set_size=safe_set_size,
                oscillation=oscillation,
                mode=supervisor_state.mode,
                novelty=observer_output.novelty,
                soft_veto=observer_output.soft_veto,
                soft_veto_hold=self._lambda_hold,
                meta_arrived=meta_arrived,
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
