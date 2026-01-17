import unittest

from atlas.supervisor import Supervisor, SupervisorConfig


class SupervisorModeTests(unittest.TestCase):
    def test_emergency_and_recover(self) -> None:
        supervisor = Supervisor(
            SupervisorConfig(
                hazard_high=1.0,
                hazard_low=0.3,
                oscillation_high=2.0,
                novelty_high=0.8,
                meta_timeout_ticks=5,
            )
        )
        state = supervisor.update(
            t=0,
            hazard=1.2,
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

    def test_meta_acceptance_blocks_in_emergency(self) -> None:
        supervisor = Supervisor(
            SupervisorConfig(
                hazard_high=1.0,
                hazard_low=0.3,
                oscillation_high=2.0,
                novelty_high=0.8,
                meta_timeout_ticks=2,
            )
        )
        state = supervisor.update(
            t=0,
            hazard=1.2,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=True,
        )
        self.assertFalse(state.accept_meta)

    def test_meta_timeout_blocks_acceptance(self) -> None:
        supervisor = Supervisor(
            SupervisorConfig(
                hazard_high=1.0,
                hazard_low=0.3,
                oscillation_high=2.0,
                novelty_high=0.8,
                meta_timeout_ticks=1,
            )
        )
        supervisor.update(
            t=0,
            hazard=0.0,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=True,
        )
        supervisor.update(
            t=1,
            hazard=0.0,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=False,
        )
        state = supervisor.update(
            t=2,
            hazard=0.0,
            oscillation=0.0,
            novelty=0.0,
            safe_set_empty=False,
            meta_arrived=False,
        )
        self.assertFalse(state.accept_meta)


if __name__ == "__main__":
    unittest.main()
