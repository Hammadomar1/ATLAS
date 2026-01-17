import time
import unittest

from atlas.offboard import MetaCognitionServer, MetaConfig
from atlas.parameter_server import ParameterBounds, ParameterServer
from atlas.safety_shield import SafetyShield, ShieldConfig


class ParameterServerTests(unittest.TestCase):
    def test_projection_clamps_bounds(self) -> None:
        server = ParameterServer(
            initial_theta=[0.5, 0.5],
            bounds=ParameterBounds(minimum=[0.0, 0.0], maximum=[1.0, 1.0]),
        )
        updated = server.apply_delta([1.0, -1.0])
        self.assertEqual(updated, [1.0, 0.0])


class SafetyShieldTests(unittest.TestCase):
    def test_fallback_when_no_safe_actions(self) -> None:
        shield = SafetyShield(ShieldConfig(hazard_threshold=0.1, halt_action=-0.2))
        output = shield.apply(0.5, [0.5, 0.6])
        self.assertTrue(output.used_fallback)
        self.assertEqual(output.action, -0.2)

    def test_selects_safe_action_near_nominal(self) -> None:
        shield = SafetyShield(ShieldConfig(hazard_threshold=0.5, halt_action=0.0))
        output = shield.apply(0.1, [0.4, 0.2, 0.6])
        self.assertFalse(output.safe_set_empty)
        self.assertGreaterEqual(output.action, -1.0)
        self.assertLessEqual(output.action, 1.0)


class MetaServerTests(unittest.TestCase):
    def test_delay_dropout_injector(self) -> None:
        server = MetaCognitionServer(
            MetaConfig(theta_dim=2, delay_max_ticks=0, dropout_prob=0.0, update_scale=0.1)
        )
        server.start()
        packet = server.synthesize_update(k=0, created_at_t=5)
        server.submit(packet)
        time.sleep(0.05)
        arrivals = server.poll_arrivals(5)
        server.stop()
        self.assertEqual(len(arrivals), 1)
        self.assertEqual(arrivals[0].t, 5)


if __name__ == "__main__":
    unittest.main()
