"""Analyze whether a trained PPO policy changes actions from reward feedback.

This script complements plot_episode_log.py:
1) random policy vs trained policy performance,
2) checkpoint-by-checkpoint action drift on the same cases,
3) action dependence on observation features.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "rl"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "air_jet_ppo.json"
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv
from eval.plot_episode_log import plot_episode_log


ACTION_COLUMNS = ["action_umax", "action_sigma", "action_t_on", "action_duration"]
FEATURE_COLUMNS = ["mass", "size_x", "size_y", "size_z", "vx", "target_center", "target_width"]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    step: int
    model_path: Path
    vecnormalize_path: Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run(results_dir: Path) -> Path:
    candidates = sorted(
        [path for path in results_dir.iterdir() if path.is_dir() and (path / "models" / "final_model.zip").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No trained run found under {results_dir}")
    return candidates[0]


def parse_checkpoint_step(path: Path) -> Optional[int]:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else None


def discover_models(run_dir: Path, max_checkpoints: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    checkpoint_pairs: list[ModelSpec] = []
    for model_path in sorted((run_dir / "checkpoints").glob("ppo_air_jet_*_steps.zip")):
        step = parse_checkpoint_step(model_path)
        if step is None:
            continue
        vec_path = run_dir / "checkpoints" / f"ppo_air_jet_vecnormalize_{step}_steps.pkl"
        if vec_path.exists():
            checkpoint_pairs.append(ModelSpec(label=f"{step}", step=step, model_path=model_path, vecnormalize_path=vec_path))

    if checkpoint_pairs:
        checkpoint_pairs = sorted(checkpoint_pairs, key=lambda spec: spec.step)
        if max_checkpoints > 0 and len(checkpoint_pairs) > max_checkpoints:
            indices = np.linspace(0, len(checkpoint_pairs) - 1, max_checkpoints, dtype=int)
            checkpoint_pairs = [checkpoint_pairs[int(i)] for i in indices]
        specs.extend(checkpoint_pairs)

    final_model = run_dir / "models" / "final_model.zip"
    final_vec = run_dir / "models" / "vecnormalize.pkl"
    if final_model.exists() and final_vec.exists():
        final_step = checkpoint_pairs[-1].step if checkpoint_pairs else 0
        specs.append(ModelSpec(label="final", step=final_step + 1, model_path=final_model, vecnormalize_path=final_vec))

    if not specs:
        raise FileNotFoundError(f"No model/VecNormalize pair found under {run_dir}")
    return specs


def build_vecnormalize(config: dict, vecnormalize_path: Path, seed: int) -> VecNormalize:
    env = DummyVecEnv([lambda: Monitor(AirJetSortingEnv(config=config.get("env", {}), seed=seed))])
    vec = VecNormalize.load(vecnormalize_path, env)
    vec.training = False
    vec.norm_reward = False
    return vec


def case_features(env: AirJetSortingEnv) -> dict:
    if env.current_case is None:
        raise RuntimeError("Environment has no active case.")
    case = env.current_case
    target_center = 0.5 * (case["target_x_min"] + case["target_x_max"])
    target_width = case["target_x_max"] - case["target_x_min"]
    return {
        "object_type": case["object_type"],
        "mass": float(case["mass"]),
        "size_x": float(case["size_x"]),
        "size_y": float(case["size_y"]),
        "size_z": float(case["size_z"]),
        "vx": float(case["vx"]),
        "target_x_min": float(case["target_x_min"]),
        "target_x_max": float(case["target_x_max"]),
        "target_center": float(target_center),
        "target_width": float(target_width),
    }


def row_from_step(
    *,
    label: str,
    step: int,
    seed: int,
    env: AirJetSortingEnv,
    action: np.ndarray,
    reward: float,
    info: dict,
) -> dict:
    row = {
        "policy": label,
        "step": step,
        "seed": seed,
        "reward": float(reward),
        "success": bool(info.get("success", False)),
        "has_landed": bool(info.get("has_landed", False)),
        "landing_x": info.get("landing_x"),
        "landing_y": info.get("landing_y"),
        "landing_time": info.get("landing_time"),
        "max_angular_speed": info.get("max_angular_speed"),
        "raw_action_0": float(action[0]),
        "raw_action_1": float(action[1]),
        "raw_action_2": float(action[2]),
        "raw_action_3": float(action[3]),
    }
    row.update(case_features(env))
    for key, value in (info.get("action_physical") or {}).items():
        row[f"action_{key}"] = float(value)
    return row


def evaluate_random_policy(config: dict, seeds: Iterable[int], rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        env = AirJetSortingEnv(config=config.get("env", {}), seed=int(seed))
        env.reset(seed=int(seed))
        action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        _, reward, _, _, info = env.step(action)
        rows.append(row_from_step(label="random", step=-1, seed=int(seed), env=env, action=action, reward=reward, info=info))
    return rows


def evaluate_model(config: dict, spec: ModelSpec, seeds: Iterable[int]) -> list[dict]:
    vec = build_vecnormalize(config, spec.vecnormalize_path, seed=0)
    model = PPO.load(spec.model_path, device="cpu")
    rows: list[dict] = []

    for seed in seeds:
        env = AirJetSortingEnv(config=config.get("env", {}), seed=int(seed))
        obs, _ = env.reset(seed=int(seed))
        norm_obs = vec.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        action_1d = np.asarray(action, dtype=np.float32).reshape(-1)
        _, reward, _, _, info = env.step(action_1d)
        rows.append(
            row_from_step(
                label=spec.label,
                step=spec.step,
                seed=int(seed),
                env=env,
                action=action_1d,
                reward=reward,
                info=info,
            )
        )

    vec.close()
    return rows


def save_policy_performance_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        df.groupby(["policy", "step"], sort=False)
        .agg(
            mean_reward=("reward", "mean"),
            sem_reward=("reward", "sem"),
            success_rate=("success", "mean"),
            mean_norm_x_error=("normalized_x_error", "mean"),
        )
        .reset_index()
    )
    labels = summary["policy"].astype(str).tolist()
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(x, summary["mean_reward"], yerr=summary["sem_reward"].fillna(0.0), color="tab:blue", alpha=0.8)
    axes[0].set_title("Reward")
    axes[0].set_ylabel("Mean reward")

    axes[1].bar(x, summary["success_rate"], color="tab:green", alpha=0.8)
    axes[1].set_title("Success Rate")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(x, summary["mean_norm_x_error"], color="tab:orange", alpha=0.8)
    axes[2].axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_title("Normalized X Error")
    axes[2].set_ylabel("Mean |x error| / target half-width")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "policy_performance_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_checkpoint_action_plot(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    checkpoint_df = df[df["policy"] != "random"].copy()
    if checkpoint_df.empty:
        return None

    grouped = checkpoint_df.groupby(["policy", "step"], sort=False)[ACTION_COLUMNS].mean().reset_index()
    labels = grouped["policy"].astype(str).tolist()
    x = np.arange(len(grouped))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), ACTION_COLUMNS):
        ax.plot(x, grouped[col], marker="o", linewidth=2)
        ax.set_title(col)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(alpha=0.25)

    output_path = output_dir / "checkpoint_action_evolution.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_observation_action_plot(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    trained = df[df["policy"] == "final"].copy()
    if trained.empty:
        trained = df[df["policy"] != "random"].sort_values("step").groupby("seed", as_index=False).tail(1)
    if trained.empty:
        return None

    fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True)
    pairs = [
        ("mass", "action_umax"),
        ("mass", "action_t_on"),
        ("target_center", "action_t_on"),
        ("target_center", "action_umax"),
        ("target_width", "action_sigma"),
        ("vx", "action_t_on"),
        ("size_x", "action_sigma"),
        ("size_z", "action_umax"),
        ("size_y", "action_duration"),
        ("object_type", "action_umax"),
        ("object_type", "action_t_on"),
        ("object_type", "action_duration"),
    ]

    for ax, (feature, action_col) in zip(axes.ravel(), pairs):
        if feature == "object_type":
            categories = sorted(trained[feature].astype(str).unique())
            values = [trained.loc[trained[feature].astype(str) == name, action_col] for name in categories]
            ax.boxplot(values, tick_labels=categories, showfliers=False)
            ax.tick_params(axis="x", rotation=25)
        else:
            ax.scatter(trained[feature], trained[action_col], s=22, alpha=0.7)
        ax.set_xlabel(feature)
        ax.set_ylabel(action_col)
        ax.grid(alpha=0.25)

    output_path = output_dir / "observation_action_response.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_center"] = 0.5 * (df["target_x_min"] + df["target_x_max"])
    df["target_width"] = df["target_x_max"] - df["target_x_min"]
    df["x_error"] = df["landing_x"] - df["target_center"]
    df["abs_x_error"] = df["x_error"].abs()
    df["normalized_x_error"] = df["abs_x_error"] / (0.5 * df["target_width"]).clip(lower=1.0e-12)
    return df


def write_correlation_table(df: pd.DataFrame, output_dir: Path) -> Path:
    trained = df[df["policy"] == "final"].copy()
    if trained.empty:
        trained = df[df["policy"] != "random"].sort_values("step").groupby("seed", as_index=False).tail(1)

    rows = []
    for feature in FEATURE_COLUMNS:
        for action_col in ACTION_COLUMNS:
            rows.append(
                {
                    "feature": feature,
                    "action": action_col,
                    "pearson_corr": float(trained[feature].corr(trained[action_col])) if len(trained) > 1 else np.nan,
                }
            )
    corr_df = pd.DataFrame(rows)
    output_path = output_dir / "observation_action_correlations.csv"
    corr_df.to_csv(output_path, index=False)
    return output_path


def save_tables(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    policy_csv = output_dir / "policy_action_analysis.csv"
    df.to_csv(policy_csv, index=False)
    paths.append(policy_csv)

    summary = (
        df.groupby(["policy", "step"], sort=False)
        .agg(
            episodes=("seed", "size"),
            success_rate=("success", "mean"),
            mean_reward=("reward", "mean"),
            mean_normalized_x_error=("normalized_x_error", "mean"),
            mean_action_umax=("action_umax", "mean"),
            mean_action_sigma=("action_sigma", "mean"),
            mean_action_t_on=("action_t_on", "mean"),
            mean_action_duration=("action_duration", "mean"),
        )
        .reset_index()
    )
    summary_csv = output_dir / "policy_action_summary.csv"
    summary.to_csv(summary_csv, index=False)
    paths.append(summary_csv)

    summary_json = output_dir / "policy_action_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    paths.append(summary_json)
    paths.append(write_correlation_table(df, output_dir))
    return paths


def analyze_policy_actions(
    *,
    run_dir: Path,
    config_path: Optional[Path],
    output_dir: Optional[Path],
    episodes: int,
    seed_start: int,
    max_checkpoints: int,
    include_episode_plots: bool,
    window: int,
) -> list[Path]:
    run_dir = run_dir.resolve()
    config_path = config_path or (run_dir / "config_used.json")
    if not config_path.exists():
        config_path = DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    output_dir = output_dir or (run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    if include_episode_plots and (run_dir / "episode_log.csv").exists():
        saved_paths.extend(plot_episode_log(run_dir / "episode_log.csv", output_dir, window=max(1, window)))

    seeds = list(range(seed_start, seed_start + episodes))
    rng = np.random.default_rng(int(config.get("seed", 42)) + 777)
    specs = discover_models(run_dir, max_checkpoints=max_checkpoints)

    rows: list[dict] = []
    rows.extend(evaluate_random_policy(config, seeds, rng))
    for spec in specs:
        rows.extend(evaluate_model(config, spec, seeds))

    df = add_error_columns(pd.DataFrame(rows))
    saved_paths.extend(save_tables(df, output_dir))
    saved_paths.append(save_policy_performance_plot(df, output_dir))
    checkpoint_plot = save_checkpoint_action_plot(df, output_dir)
    if checkpoint_plot is not None:
        saved_paths.append(checkpoint_plot)
    observation_plot = save_observation_action_plot(df, output_dir)
    if observation_plot is not None:
        saved_paths.append(observation_plot)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trained PPO policy actions on fixed test cases.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory under results/rl")
    parser.add_argument("--config", type=Path, default=None, help="Config path. Defaults to run_dir/config_used.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV/plot outputs.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of fixed test seeds.")
    parser.add_argument("--seed-start", type=int, default=1000, help="First test seed.")
    parser.add_argument("--max-checkpoints", type=int, default=6, help="Evenly sample at most this many checkpoints.")
    parser.add_argument("--no-episode-plots", action="store_true", help="Skip existing episode_log plots.")
    parser.add_argument("--window", type=int, default=100, help="Rolling window for episode_log plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_run(DEFAULT_RESULTS_DIR)
    paths = analyze_policy_actions(
        run_dir=run_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        episodes=max(1, args.episodes),
        seed_start=args.seed_start,
        max_checkpoints=max(0, args.max_checkpoints),
        include_episode_plots=not args.no_episode_plots,
        window=args.window,
    )
    print(f"Analyzed run: {run_dir}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
