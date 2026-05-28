"""Evaluate a trained PPO model on the air-jet sorting environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "air_jet_ppo.json"
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(model_path: Path, vecnormalize_path: Path, config_path: Path, episodes: int):
    config = load_config(config_path)
    seed = int(config.get("seed", 42)) + 100000

    env = DummyVecEnv([lambda: Monitor(AirJetSortingEnv(config=config.get("env", {}), seed=seed))])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    rows = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, done, infos = env.step(action)
        info = infos[0]
        row = {
            "episode": episode,
            "reward": float(reward[0]),
            "done": bool(done[0]),
            "success": bool(info.get("success", False)),
            "has_landed": bool(info.get("has_landed", False)),
            "landing_x": info.get("landing_x"),
            "landing_y": info.get("landing_y"),
            "landing_time": info.get("landing_time"),
            "max_angular_speed": info.get("max_angular_speed"),
            "jet_impulse_norm": info.get("jet_impulse_norm"),
            "object_type": info.get("object_type"),
            "target_x_min": info.get("target_x_min"),
            "target_x_max": info.get("target_x_max"),
        }
        action_physical = info.get("action_physical") or {}
        for key, value in action_physical.items():
            row[f"action_{key}"] = value
        rows.append(row)

    env.close()

    df = pd.DataFrame(rows)
    output_dir = model_path.parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.csv"
    summary_path = output_dir / "summary.json"

    df.to_csv(output_path, index=False)
    summary = {
        "episodes": int(episodes),
        "success_rate": float(df["success"].mean()) if len(df) else 0.0,
        "mean_reward": float(df["reward"].mean()) if len(df) else 0.0,
        "mean_landing_x": float(df["landing_x"].mean()) if len(df) else None,
        "mean_landing_y": float(df["landing_y"].mean()) if len(df) else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    print(f"Saved evaluation CSV: {output_path}")
    print(f"Saved evaluation summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--episodes", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        config_path=args.config,
        episodes=args.episodes,
    )
