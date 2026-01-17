import math
import unittest

from atlas.safety_shield import SafetyShield, ShieldConfig


class SafetyInvariantTests(unittest.TestCase):
    def test_action_is_safe_or_halt(self) -> None:
        drive = [0.2, 0.9, 0.4]
        config = ShieldConfig(hazard_threshold=0.5, halt_action=-0.3)
        shield = SafetyShield(config)
        output = shield.apply(0.7, drive)
        if output.safe_set_empty:
            self.assertEqual(output.action, config.halt_action)
            return
        candidates = [-1.0, 0.0, 1.0]
        index = candidates.index(output.action)
        self.assertLessEqual(drive[index], config.hazard_threshold)

    def test_invalid_drive_forces_halt(self) -> None:
        shield = SafetyShield(ShieldConfig(hazard_threshold=0.5, halt_action=0.0))
        output = shield.apply(0.0, [0.1, math.nan, 0.2])
        self.assertTrue(output.safe_set_empty)
        self.assertEqual(output.action, 0.0)


if __name__ == "__main__":
    unittest.main()
