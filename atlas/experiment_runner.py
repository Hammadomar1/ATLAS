from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import argparse
import csv

from atlas.config import load_config, set_determinism
from atlas.env import SimpleCorridorEnv
from atlas.evaluation import (
    BaselineConfig,
    run_e1_determinism_latency,
    run_e2_non_blocking,
    run_e3_shield_effectiveness,
    run_e4_delay_dropout,
    run_e5_oscillation,
    run_e6_regime_shift,
)
from atlas.meta_queue import DelayDropoutConfig, MetaPacketQueue
from atlas.logging import resolve_git_hash
from atlas.scenarios import ScenarioSampler
from atlas.scheduler import SchedulerConfig
from atlas.types import MetaPacket


@dataclass(frozen=True)
class SweepConfig:
    delays: Sequence[int]
    dropouts: Sequence[float]
    ticks: int
    output_dir: Path
    run_config: dict[str, object]


def run_sweep(
    env: SimpleCorridorEnv,
    system_step: Callable[[int, Sequence[Sequence[Sequence[int]]]], float],
    meta_policy: Callable[[int], MetaPacket],
    scheduler: SchedulerConfig,
    sweep: SweepConfig,
) -> None:
    results = []
    _write_manifest(sweep)
    for delay in sweep.delays:
        for dropout in sweep.dropouts:
            meta_queue = MetaPacketQueue(
                DelayDropoutConfig(
                    delay_distribution="uniform",
                    delay_max_ticks=delay,
                    dropout_prob=dropout,
                ),
                seed=1,
            )
            latency = run_e1_determinism_latency(env, system_step, meta_queue, meta_policy, scheduler, sweep.ticks)
            non_blocking = run_e2_non_blocking(env, system_step, sweep.ticks)
            collisions, steps = run_e3_shield_effectiveness(env, system_step, sweep.ticks)
            delay_collisions = run_e4_delay_dropout(env, system_step, meta_queue, meta_policy, scheduler, sweep.ticks)
            flips = run_e5_oscillation(env, system_step, sweep.ticks)
            shift_collisions = run_e6_regime_shift(env, system_step, sweep.ticks)
            results.append(
                {
                    "delay": delay,
                    "dropout": dropout,
                    "latency": asdict(latency),
                    "non_blocking_steps": non_blocking,
                    "collisions": collisions,
                    "steps": steps,
                    "delay_collisions": delay_collisions,
                    "oscillation_flips": flips,
                    "regime_shift_collisions": shift_collisions,
                }
            )
    sweep.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = sweep.output_dir / "sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    _write_summary_csv(results, sweep.output_dir / "sweep_summary.csv")


def default_baselines() -> Sequence[BaselineConfig]:
    return [
        BaselineConfig(name="reflex_only", use_hnsp=False, use_meta=False, use_shield=True),
        BaselineConfig(name="reflex_hnsp", use_hnsp=True, use_meta=False, use_shield=True),
        BaselineConfig(name="full_atlas", use_hnsp=True, use_meta=True, use_shield=True),
    ]


def _write_summary_csv(results: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "delay",
        "dropout",
        "deadline_miss_rate",
        "max_tick_time",
        "mean_tick_time",
        "jitter",
        "non_blocking_steps",
        "collisions",
        "steps",
        "delay_collisions",
        "oscillation_flips",
        "regime_shift_collisions",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            latency = row["latency"]
            writer.writerow(
                {
                    "delay": row["delay"],
                    "dropout": row["dropout"],
                    "deadline_miss_rate": latency["deadline_miss_rate"],
                    "max_tick_time": latency["max_tick_time"],
                    "mean_tick_time": latency["mean_tick_time"],
                    "jitter": latency["jitter"],
                    "non_blocking_steps": row["non_blocking_steps"],
                    "collisions": row["collisions"],
                    "steps": row["steps"],
                    "delay_collisions": row["delay_collisions"],
                    "oscillation_flips": row["oscillation_flips"],
                    "regime_shift_collisions": row["regime_shift_collisions"],
                }
            )


def _write_manifest(sweep: SweepConfig) -> None:
    payload = {
        "delays": list(sweep.delays),
        "dropouts": list(sweep.dropouts),
        "ticks": sweep.ticks,
        "git_hash": resolve_git_hash(),
        "run_config": sweep.run_config,
    }
    manifest_path = sweep.output_dir / "experiment_manifest.json"
    sweep.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATLAS experiment sweeps.")
    parser.add_argument("--config", type=Path, default=Path("configs/run_config.json"))
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 2, 5])
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/experiments"))
    args = parser.parse_args()

    run_config = load_config(args.config)
    set_determinism(run_config.seed)
    settings = run_config.settings

    env = SimpleCorridorEnv()
    sampler = ScenarioSampler(seed=run_config.seed)
    preset = settings.get("scenario_preset")
    scenario = sampler.preset(preset) if preset else sampler.sample()
    env.reset(seed=run_config.seed, scenario_cfg=scenario)
    scheduler = SchedulerConfig(
        reflex_period_s=float(settings.get("reflex_period_s", 0.02)),
        meta_period_ticks=int(settings.get("meta_period_ticks", 5)),
    )

    def policy_step(_t: int, _frame: Sequence[Sequence[Sequence[int]]]) -> float:
        return 0.0

    def meta_policy(tick: int) -> MetaPacket:
        return MetaPacket(k=tick, delta_theta=[0.0], created_at_t=tick)

    ticks = int(settings.get("ticks", args.ticks))
    delays = settings.get("delays", args.delays)
    dropouts = settings.get("dropouts", args.dropouts)
    output_dir = Path(settings.get("output_dir", args.output_dir))
    sweep = SweepConfig(
        delays=delays,
        dropouts=dropouts,
        ticks=ticks,
        output_dir=output_dir,
        run_config=settings,
    )
    run_sweep(env, policy_step, meta_policy, scheduler, sweep)


if __name__ == "__main__":
    main()
