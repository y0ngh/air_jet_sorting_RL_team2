"""Sweep fixed jet position and plot landing distance heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sweeps" / "jet_position"
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("env", {})


def make_fixed_case(
    env: AirJetSortingEnv,
    object_type: str,
    seed: int,
    preserve_target: bool = False,
) -> Dict[str, Any]:
    env.reset(seed=seed, options={"object_type": object_type})
    if env.current_case is None:
        raise RuntimeError("Environment did not create a case on reset.")
    case = deepcopy(env.current_case)
    if not preserve_target:
        center_x = 0.5 * (case["target_x_min"] + case["target_x_max"])
        case["target_x_min"] = center_x - 0.20
        case["target_x_max"] = center_x + 0.20
    return case


def run_sweep(
    x_values: np.ndarray,
    z_values: np.ndarray,
    cases: List[Dict[str, Any]],
    jet_params: Dict[str, float],
    base_config: Dict[str, Any],
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    mean_landing_x = np.full((len(z_values), len(x_values)), np.nan, dtype=float)
    rows: List[Dict[str, Any]] = []

    env = AirJetSortingEnv(config=base_config, seed=0)
    for zi, z_center in enumerate(z_values):
        for xi, x_center in enumerate(x_values):
            env.config["fixed_jet"]["x_center"] = float(x_center)
            env.config["fixed_jet"]["z_center"] = float(z_center)

            landing_values = []
            success_values = []
            for case in cases:
                env.current_case = deepcopy(case)
                result = env._run_simulation(env.current_case, jet_params)
                landing_position = result["landing_position"]
                landing_x = float(landing_position[0]) if landing_position is not None else np.nan
                landing_values.append(landing_x)
                success_values.append(bool(result["success"]) if result["success"] is not None else False)

            mean_x = float(np.nanmean(landing_values))
            success_rate = float(np.mean(success_values))
            mean_landing_x[zi, xi] = mean_x
            rows.append(
                {
                    "jet_x": float(x_center),
                    "jet_z": float(z_center),
                    "mean_landing_x": mean_x,
                    "success_rate": success_rate,
                    "case_count": len(cases),
                }
            )

    return mean_landing_x, rows


def save_heatmap(
    values: np.ndarray,
    x_values: np.ndarray,
    z_values: np.ndarray,
    best_row: Dict[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), z_values.min(), z_values.max()],
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Mean landing_x [m]")
    ax.scatter(best_row["jet_x"], best_row["jet_z"], color="red", s=70, marker="x", label="best")
    ax.set_title("Jet Position Sweep")
    ax.set_xlabel("jet x_center [m]")
    ax.set_ylabel("jet z_center [m]")
    ax.legend()
    ax.grid(color="white", alpha=0.18)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep jet x/z position for fixed cases.")
    parser.add_argument("--x-min", type=float, default=0.00)
    parser.add_argument("--x-max", type=float, default=0.50)
    parser.add_argument("--z-min", type=float, default=0.04)
    parser.add_argument("--z-max", type=float, default=0.30)
    parser.add_argument("--x-count", type=int, default=26)
    parser.add_argument("--z-count", type=int, default=21)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--umax", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--t-on", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=0.10)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preserve-target", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x_values = np.linspace(args.x_min, args.x_max, args.x_count)
    z_values = np.linspace(args.z_min, args.z_max, args.z_count)
    jet_params = {
        "umax": float(args.umax),
        "sigma": float(args.sigma),
        "t_on": float(args.t_on),
        "duration": float(args.duration),
    }

    env_config = load_config(args.config)
    base_env = AirJetSortingEnv(config=env_config, seed=args.seed)
    base_config = deepcopy(base_env.config)
    cases = [
        make_fixed_case(base_env, object_type="plate", seed=args.seed, preserve_target=args.preserve_target),
        make_fixed_case(base_env, object_type="rod", seed=args.seed + 1, preserve_target=args.preserve_target),
        make_fixed_case(base_env, object_type="irregular", seed=args.seed + 2, preserve_target=args.preserve_target),
    ]

    values, rows = run_sweep(
        x_values=x_values,
        z_values=z_values,
        cases=cases,
        jet_params=jet_params,
        base_config=base_config,
    )
    best_row = max(rows, key=lambda row: row["mean_landing_x"])

    csv_path = output_dir / "jet_position_sweep.csv"
    heatmap_path = output_dir / "jet_position_heatmap.png"
    write_csv(rows, csv_path)
    save_heatmap(values, x_values, z_values, best_row, heatmap_path)

    print(f"Cases: {[case['object_type'] for case in cases]}")
    print(f"Fixed jet params: {jet_params}")
    print(f"Best jet_x: {best_row['jet_x']:.4f} m")
    print(f"Best jet_z: {best_row['jet_z']:.4f} m")
    print(f"Best mean_landing_x: {best_row['mean_landing_x']:.4f} m")
    print(f"Best success_rate: {best_row['success_rate']:.3f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
