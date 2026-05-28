"""Fair three-way baseline comparison: no_jet vs fixed_jet vs PPO.

All three conditions use exactly the same deterministic test case per seed:
- same object type, size, mass, drag
- same initial position, velocity, orientation, angular velocity
- same target region
- same simulator internal seed (case["seed"])

Only the jet control method differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "fixed_y0_z020_vx100_target04_09_relaxed_reward_env8_100k.json"
)
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv

# ── no_jet: umax=0 disables the jet; remaining values are placeholders
NO_JET_PARAMS: Dict[str, float] = {
    "umax": 0.0,
    "sigma": 0.05,
    "t_on": 0.0,
    "duration": 0.01,
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_fixed_jet_params(case: Dict[str, Any], jet_x: float) -> Dict[str, float]:
    """Deterministic strong-heuristic jet centered on nominal object arrival time.

    t_on centers the duration window at the moment the object nominally
    reaches the nozzle x-position.  Uses per-case x0 and vx so the formula
    stays valid if those ranges are ever widened in a future config.
    """
    duration = 0.10
    x0 = float(case["x0"])
    vx = float(case["vx"])
    t_on = max(0.0, (jet_x - x0) / max(vx, 1.0e-8) - duration / 2.0)
    return {
        "umax": 30.0,
        "sigma": 0.05,
        "t_on": t_on,
        "duration": duration,
    }


def result_to_info(
    result: Dict[str, Any],
    jet_params: Dict[str, float],
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an info dict matching the shape returned by env.step()."""
    landing = result["landing_position"]
    return {
        "success": bool(result["success"]) if result["success"] is not None else False,
        "has_landed": bool(result["has_landed"]),
        "landing_x": float(landing[0]) if landing is not None else float("nan"),
        "landing_y": float(landing[1]) if landing is not None else float("nan"),
        "action_physical": jet_params,
        "object_type": case["object_type"],
        "target_x_min": float(case["target_x_min"]),
        "target_x_max": float(case["target_x_max"]),
    }


def build_row(
    condition: str,
    episode: int,
    seed: int,
    info: Dict[str, Any],
    reward: float,
) -> Dict[str, Any]:
    ap = info.get("action_physical") or {}
    return {
        "condition": condition,
        "episode": episode,
        "seed": seed,
        "object_type": info.get("object_type"),
        "success": bool(info.get("success", False)),
        "has_landed": bool(info.get("has_landed", False)),
        "landing_x": info.get("landing_x"),
        "landing_y": info.get("landing_y"),
        "reward": float(reward),
        "action_umax": ap.get("umax"),
        "action_sigma": ap.get("sigma"),
        "action_t_on": ap.get("t_on"),
        "action_duration": ap.get("duration"),
        "target_x_min": info.get("target_x_min"),
        "target_x_max": info.get("target_x_max"),
    }


def load_ppo(
    run_dir: Path, env_config: Dict[str, Any], seed: int
) -> tuple[PPO, VecNormalize]:
    """Load the final PPO model and its VecNormalize statistics."""
    model_path = run_dir / "models" / "final_model.zip"
    vecnormalize_path = run_dir / "models" / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnormalize_path}")

    dummy_env = DummyVecEnv(
        [lambda: Monitor(AirJetSortingEnv(config=env_config, seed=seed))]
    )
    vec = VecNormalize.load(vecnormalize_path, dummy_env)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(model_path, device="cpu")
    return model, vec


def compare_baselines(
    run_dir: Path,
    config_path: Optional[Path],
    output_dir: Path,
    episodes: int,
    seed_start: int,
) -> None:
    # ── Resolve config
    if config_path is None:
        candidate = run_dir / "config_used.json"
        config_path = candidate if candidate.exists() else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    env_config = config.get("env", {})
    jet_x = float(env_config.get("fixed_jet", {}).get("x_center", 0.105))

    print(f"Config:  {config_path}")
    print(f"Run dir: {run_dir}")
    print(f"Jet x_center for fixed_jet timing: {jet_x:.4f} m")

    # ── Load PPO once (VecNormalize statistics are shared across all seeds)
    model, vec = load_ppo(run_dir, env_config, seed=seed_start)

    # ── One env object reused across seeds via reset(seed=X)
    env = AirJetSortingEnv(config=env_config, seed=seed_start)

    rows: List[Dict[str, Any]] = []
    seeds = list(range(seed_start, seed_start + episodes))

    for i, seed in enumerate(seeds):
        episode = i + 1

        # Sample the deterministic test case.  reset(seed=X) reinitialises
        # self.rng and calls _sample_case(), so the case is identical every
        # time this seed is used with this config.
        obs, _ = env.reset(seed=seed)
        case = dict(env.current_case)  # frozen copy for direct simulator calls

        # ── Method 1: no_jet
        result_no = env._run_simulation(case, NO_JET_PARAMS)
        reward_no = env._compute_reward(result_no, NO_JET_PARAMS)
        rows.append(build_row("no_jet", episode, seed,
                               result_to_info(result_no, NO_JET_PARAMS, case),
                               reward_no))

        # ── Method 2: fixed_jet
        fixed_params = make_fixed_jet_params(case, jet_x)
        result_fixed = env._run_simulation(case, fixed_params)
        reward_fixed = env._compute_reward(result_fixed, fixed_params)
        rows.append(build_row("fixed_jet", episode, seed,
                               result_to_info(result_fixed, fixed_params, case),
                               reward_fixed))

        # ── Method 3: ppo
        # env.current_case is still the same case (step/reset not called since
        # the reset above), so env.step() runs the same initial condition.
        norm_obs = vec.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        action_1d = np.asarray(action, dtype=np.float32).reshape(-1)
        _, reward_ppo, _, _, info_ppo = env.step(action_1d)
        rows.append(build_row("ppo", episode, seed, info_ppo, float(reward_ppo)))

    # ── Write main CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    main_csv = output_dir / "baseline_comparison.csv"
    df.to_csv(main_csv, index=False)

    # ── Write summary CSV
    # mean_landing_x / std_landing_x are computed over episodes that landed;
    # NaN rows are dropped by pandas by default.  landing_rate shows what
    # fraction of episodes actually landed, so mean_landing_x should be
    # interpreted relative to that rate.
    summary_rows = []
    for cond in ["no_jet", "fixed_jet", "ppo"]:
        sub = df[df["condition"] == cond]
        summary_rows.append({
            "condition": cond,
            "success_rate": float(sub["success"].mean()),
            "landing_rate": float(sub["has_landed"].mean()),
            "mean_landing_x": float(sub["landing_x"].mean()),
            "std_landing_x": float(sub["landing_x"].std()),
            "mean_action_umax": float(sub["action_umax"].mean()),
            "mean_action_sigma": float(sub["action_sigma"].mean()),
            "mean_action_t_on": float(sub["action_t_on"].mean()),
            "mean_action_duration": float(sub["action_duration"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "baseline_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Baseline Comparison Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nTotal rows: {len(df)}  ({episodes} episodes × 3 conditions)")
    print(f"Main CSV:    {main_csv}")
    print(f"Summary CSV: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair three-way comparison: no_jet vs fixed_jet vs PPO."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Trained PPO run directory (must contain models/final_model.zip).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Config path. Default: <run-dir>/config_used.json if it exists, "
            "otherwise the project default config."
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of deterministic test seeds (default: 100).",
    )
    parser.add_argument(
        "--seed-start", type=int, default=1000,
        help="First test seed (default: 1000).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <run-dir>/baseline_comparison).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "baseline_comparison")
    compare_baselines(
        run_dir=args.run_dir.resolve(),
        config_path=args.config,
        output_dir=output_dir.resolve(),
        episodes=max(1, args.episodes),
        seed_start=args.seed_start,
    )
