"""Sweep jet activation time and plot landing distance."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sweeps" / "jet_timing"
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv


def make_fixed_case(env: AirJetSortingEnv, object_type: str, seed: int) -> Dict[str, Any]:
    env.reset(seed=seed, options={"object_type": object_type})
    if env.current_case is None:
        raise RuntimeError("Environment did not create a case on reset.")
    return deepcopy(env.current_case)


def run_sweep(
    t_on_values: np.ndarray,
    cases: List[Dict[str, Any]],
    jet_params: Dict[str, float],
    base_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    env = AirJetSortingEnv(config=base_config, seed=0)

    for t_on in t_on_values:
        landing_values = []
        success_values = []
        reward_values = []

        for case in cases:
            env.current_case = deepcopy(case)
            params = dict(jet_params)
            params["t_on"] = float(t_on)
            result = env._run_simulation(env.current_case, params)
            reward = env._compute_reward(result, params)
            landing_position = result["landing_position"]
            landing_x = float(landing_position[0]) if landing_position is not None else np.nan
            landing_values.append(landing_x)
            success_values.append(bool(result["success"]) if result["success"] is not None else False)
            reward_values.append(float(reward))

        rows.append(
            {
                "t_on": float(t_on),
                "mean_landing_x": float(np.nanmean(landing_values)),
                "success_rate": float(np.mean(success_values)),
                "mean_reward": float(np.mean(reward_values)),
                "case_count": len(cases),
            }
        )

    return rows


def save_plot(rows: List[Dict[str, Any]], best_distance: Dict[str, Any], best_success: Dict[str, Any], output_path: Path) -> None:
    t_on = np.array([row["t_on"] for row in rows], dtype=float)
    mean_landing_x = np.array([row["mean_landing_x"] for row in rows], dtype=float)
    success_rate = np.array([row["success_rate"] for row in rows], dtype=float)
    mean_reward = np.array([row["mean_reward"] for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)

    axes[0].plot(t_on, mean_landing_x, marker="o", linewidth=1.8)
    axes[0].axvline(best_distance["t_on"], color="tab:red", linestyle="--", label="max landing_x")
    axes[0].set_ylabel("Mean landing_x [m]")
    axes[0].set_title("Jet t_on Sweep")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t_on, success_rate, marker="o", color="tab:green", linewidth=1.8)
    axes[1].axvline(best_success["t_on"], color="tab:red", linestyle="--", label="max success")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(t_on, mean_reward, marker="o", color="tab:purple", linewidth=1.8)
    axes[2].set_xlabel("t_on [s from simulation start]")
    axes[2].set_ylabel("Mean reward")
    axes[2].grid(alpha=0.25)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep jet activation time for fixed cases.")
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--t-count", type=int, default=51)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--umax", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t_on_values = np.linspace(args.t_min, args.t_max, args.t_count)
    jet_params = {
        "umax": float(args.umax),
        "sigma": float(args.sigma),
        "t_on": 0.0,
        "duration": float(args.duration),
    }

    base_env = AirJetSortingEnv(seed=args.seed)
    base_config = deepcopy(base_env.config)
    cases = [
        make_fixed_case(base_env, object_type="plate", seed=args.seed),
        make_fixed_case(base_env, object_type="rod", seed=args.seed + 1),
        make_fixed_case(base_env, object_type="irregular", seed=args.seed + 2),
    ]

    rows = run_sweep(
        t_on_values=t_on_values,
        cases=cases,
        jet_params=jet_params,
        base_config=base_config,
    )
    best_distance = max(rows, key=lambda row: row["mean_landing_x"])
    best_success = max(rows, key=lambda row: (row["success_rate"], row["mean_reward"]))
    best_reward = max(rows, key=lambda row: row["mean_reward"])

    csv_path = output_dir / "jet_timing_sweep.csv"
    plot_path = output_dir / "jet_timing_sweep.png"
    write_csv(rows, csv_path)
    save_plot(rows, best_distance, best_success, plot_path)

    print(f"Cases: {[case['object_type'] for case in cases]}")
    print(f"Fixed jet params except t_on: {jet_params}")
    print(f"Best t_on by mean_landing_x: {best_distance['t_on']:.4f} s, mean_landing_x={best_distance['mean_landing_x']:.4f} m")
    print(f"Best t_on by success_rate: {best_success['t_on']:.4f} s, success_rate={best_success['success_rate']:.3f}")
    print(f"Best t_on by mean_reward: {best_reward['t_on']:.4f} s, mean_reward={best_reward['mean_reward']:.4f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
