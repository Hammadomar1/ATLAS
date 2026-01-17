import unittest

from atlas.observer import HNSPConfig, HNSPObserver, QuantizerConfig
from atlas.offboard import MetaCognitionServer, MetaConfig
from atlas.parameter_server import ParameterBounds, ParameterServer
from atlas.safety_shield import SafetyShield, ShieldConfig
from atlas.supervisor import Supervisor, SupervisorConfig
from atlas.system import AtlasSystem, SystemConfig
from atlas.telemetry import MetricsLogger, TelemetryPublisher
from atlas.types import DNFState, ProtoEvent, ReflexDrive, ReflexOutput


class FakeReflexEngine:
    def __init__(self, drive):
        self._drive = drive
        self._t = 0

    def step(self, _):
        output = ReflexOutput(
            t=self._t,
            nominal_action=0.5,
            drive=ReflexDrive(drive=self._drive),
            dnf_state=DNFState(u=[0.0 for _ in self._drive], winner_index=0),
            events=[ProtoEvent(bearing_index=0, strength=0.0)],
        )
        self._t += 1
        return output


class SystemSafetyTests(unittest.TestCase):
    def _build_system(self, drive: list[float], meta_config: MetaConfig) -> AtlasSystem:
        reflex = FakeReflexEngine(drive)
        observer = HNSPObserver(
            HNSPConfig(
                memory_dim=8,
                binary_mode=True,
                seed=1,
                trajectory_length=2,
                tm_length=4,
                prototype_count=2,
                x_quantizer=QuantizerConfig(bins=4, min_value=0.0, max_value=10.0),
                w_quantizer=QuantizerConfig(bins=2, min_value=0.0, max_value=1.0),
                s_quantizer=QuantizerConfig(bins=2, min_value=0.0, max_value=1.0),
                x_dot_quantizer=QuantizerConfig(bins=2, min_value=-1.0, max_value=1.0),
                rho_quantizer=QuantizerConfig(bins=2, min_value=0.0, max_value=1.0),
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
                meta_timeout_ticks=5,
            )
        )
        shield = SafetyShield(ShieldConfig(hazard_threshold=0.5, halt_action=-0.3))
        telemetry = TelemetryPublisher()
        metrics = MetricsLogger()
        parameter_server = ParameterServer(
            initial_theta=[0.5],
            bounds=ParameterBounds(minimum=[0.0], maximum=[1.0]),
        )
        return AtlasSystem(
            SystemConfig(reflex_period_s=0.01, meta_period_ticks=10),
            reflex,
            observer,
            supervisor,
            shield,
            parameter_server,
            meta_server=MetaCognitionServer(meta_config),
            telemetry=telemetry,
            metrics=metrics,
        )

    def test_action_is_safe_or_halt(self) -> None:
        drive = [0.2, 0.9, 0.4]
        system = self._build_system(
            drive,
            MetaConfig(theta_dim=1, delay_max_ticks=0, dropout_prob=0.0, update_scale=0.1),
        )
        system.start()
        action = system.reflex_tick(0, [[[0.0]]])
        safe_action, safe_set_empty, _, _ = system._safety_shield.safe_action(action, drive)
        if safe_set_empty:
            self.assertEqual(action, system._safety_shield._config.halt_action)
        else:
            self.assertEqual(action, safe_action)
        system.stop()

    def test_action_is_safe_or_halt_over_many_ticks(self) -> None:
        drive = [0.1, 0.2, 0.9, 0.4, 0.3]
        system = self._build_system(
            drive,
            MetaConfig(theta_dim=1, delay_max_ticks=0, dropout_prob=1.0, update_scale=0.1),
        )
        system.start()
        for t in range(60):
            action = system.reflex_tick(t, [[[0.0]]])
            telemetry = system._telemetry.drain()[-1]
            safe_action, safe_set_empty, _, _ = system._safety_shield.safe_action(
                telemetry.pre_action,
                drive,
            )
            if safe_set_empty:
                self.assertEqual(action, system._safety_shield._config.halt_action)
            else:
                self.assertEqual(action, safe_action)
        system.stop()

    def test_emergency_recovery_hysteresis(self) -> None:
        supervisor = Supervisor(
            SupervisorConfig(
                hazard_high=0.8,
                hazard_low=0.3,
                oscillation_high=3.0,
                novelty_high=0.7,
                meta_timeout_ticks=5,
            )
        )
        state = supervisor.update(
            t=0,
            hazard=0.9,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=False,
        )
        self.assertEqual(state.mode, "Emergency")
        state = supervisor.update(
            t=1,
            hazard=0.2,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=False,
        )
        self.assertEqual(state.mode, "Guarded")

    def test_meta_stalled_forever(self) -> None:
        drive = [0.2, 0.4, 0.1]
        system = self._build_system(
            drive,
            MetaConfig(theta_dim=1, delay_max_ticks=0, dropout_prob=1.0, update_scale=0.1),
        )
        system._supervisor = Supervisor(
            SupervisorConfig(
                hazard_high=1.0,
                hazard_low=0.5,
                oscillation_high=3.0,
                novelty_high=0.7,
                meta_timeout_ticks=2,
            )
        )
        system.start()
        for t in range(6):
            system.maybe_emit_meta(t)
            system.reflex_tick(t, [[[0.0]]])
        telemetry = system._telemetry.drain()[-1]
        self.assertFalse(telemetry.meta_arrived)
        self.assertIsNotNone(telemetry.meta_age)
        self.assertGreater(telemetry.meta_age, 2)
        system.stop()


if __name__ == "__main__":
    unittest.main()
