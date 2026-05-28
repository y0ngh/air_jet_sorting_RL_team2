"""Plot RL episode CSV logs.

Creates summary figures from train/train_ppo.py episode_log.csv or
eval/evaluate_ppo.py evaluation.csv outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "rl"


def find_latest_episode_csv(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.glob("*/episode_log.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No episode_log.csv found under {results_dir}")
    return candidates[0]


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "episode_reward" not in df.columns and "reward" in df.columns:
        df["episode_reward"] = df["reward"]
    if "timestep" not in df.columns:
        df["timestep"] = np.arange(1, len(df) + 1)

    required = {"episode_reward", "landing_x", "landing_y", "target_x_min", "target_x_max"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["target_center"] = 0.5 * (df["target_x_min"] + df["target_x_max"])
    df["target_width"] = df["target_x_max"] - df["target_x_min"]
    df["x_error"] = df["landing_x"] - df["target_center"]
    df["abs_x_error"] = df["x_error"].abs()
    df["abs_y_error"] = df["landing_y"].abs()
    df["normalized_x_error"] = df["abs_x_error"] / (0.5 * df["target_width"]).clip(lower=1.0e-12)

    if "success" in df.columns:
        df["success"] = coerce_bool(df["success"])
    else:
        df["success"] = df["landing_x"].between(df["target_x_min"], df["target_x_max"])

    if "has_landed" in df.columns:
        df["has_landed"] = coerce_bool(df["has_landed"])

    return df


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def save_training_summary(df: pd.DataFrame, output_dir: Path, window: int) -> Path:
    x = df["timestep"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, df["episode_reward"], alpha=0.28, linewidth=0.8, label="episode")
    ax.plot(x, rolling_mean(df["episode_reward"], window), linewidth=2, label=f"rolling {window}")
    ax.set_title("Reward")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode reward")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    success = df["success"].astype(float)
    ax.plot(x, rolling_mean(success, window), linewidth=2, color="tab:green")
    ax.set_title("Success Rate")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Rolling success ({window})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(x, df["abs_x_error"], alpha=0.35, linewidth=0.8, label="abs x error")
    ax.plot(x, rolling_mean(df["abs_x_error"], window), linewidth=2, label=f"rolling {window}")
    ax.set_title("Landing X Error")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("|landing_x - target_center| [m]")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(x, df["normalized_x_error"], alpha=0.35, linewidth=0.8, label="normalized")
    ax.plot(x, rolling_mean(df["normalized_x_error"], window), linewidth=2, label=f"rolling {window}")
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1, label="target edge")
    ax.set_title("Normalized X Error")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("|x error| / target half-width")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2, 0]
    ax.plot(x, df["abs_y_error"], alpha=0.35, linewidth=0.8, label="abs y error")
    ax.plot(x, rolling_mean(df["abs_y_error"], window), linewidth=2, label=f"rolling {window}")
    ax.set_title("Landing Y Error")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("|landing_y| [m]")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2, 1]
    if "max_angular_speed" in df.columns:
        ax.plot(x, df["max_angular_speed"], alpha=0.35, linewidth=0.8, label="episode")
        ax.plot(x, rolling_mean(df["max_angular_speed"], window), linewidth=2, label=f"rolling {window}")
        ax.set_ylabel("Max angular speed")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "max_angular_speed not available", ha="center", va="center")
    ax.set_title("Angular Speed")
    ax.set_xlabel("Timestep")
    ax.grid(alpha=0.25)

    output_path = output_dir / "training_summary.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_landing_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = np.where(df["success"].to_numpy(), "tab:green", "tab:red")
    ax.scatter(df["landing_x"], df["landing_y"], c=colors, s=22, alpha=0.65, edgecolors="none")

    target_min = float(df["target_x_min"].median())
    target_max = float(df["target_x_max"].median())
    ax.axvspan(target_min, target_max, color="tab:green", alpha=0.12, label="median target x range")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title("Landing Map")
    ax.set_xlabel("landing_x [m]")
    ax.set_ylabel("landing_y [m]")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "landing_map.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_action_plot(df: pd.DataFrame, output_dir: Path, window: int) -> Optional[Path]:
    action_cols = [col for col in ["action_umax", "action_sigma", "action_t_on", "action_duration"] if col in df.columns]
    if not action_cols:
        return None

    fig, axes = plt.subplots(len(action_cols), 1, figsize=(11, 2.5 * len(action_cols)), sharex=True, constrained_layout=True)
    if len(action_cols) == 1:
        axes = [axes]
    x = df["timestep"]
    for ax, col in zip(axes, action_cols):
        ax.plot(x, df[col], alpha=0.3, linewidth=0.8)
        ax.plot(x, rolling_mean(df[col], window), linewidth=2)
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Physical Actions", y=1.01)

    output_path = output_dir / "actions.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_summary(df: pd.DataFrame) -> dict:
    summary = {
        "episodes": int(len(df)),
        "success_rate": float(df["success"].mean()) if len(df) else 0.0,
        "mean_reward": float(df["episode_reward"].mean()) if len(df) else 0.0,
        "median_reward": float(df["episode_reward"].median()) if len(df) else 0.0,
        "mean_abs_x_error_m": float(df["abs_x_error"].mean()) if len(df) else 0.0,
        "median_abs_x_error_m": float(df["abs_x_error"].median()) if len(df) else 0.0,
        "mean_abs_y_error_m": float(df["abs_y_error"].mean()) if len(df) else 0.0,
        "median_abs_y_error_m": float(df["abs_y_error"].median()) if len(df) else 0.0,
        "mean_normalized_x_error": float(df["normalized_x_error"].mean()) if len(df) else 0.0,
    }
    if "has_landed" in df.columns:
        summary["landed_rate"] = float(df["has_landed"].mean()) if len(df) else 0.0
    return summary


def write_tables(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Optional[Path], Path]:
    augmented_path = output_dir / "episode_log_with_errors.csv"
    summary_path = output_dir / "summary.json"
    by_object_path: Optional[Path] = None

    df.to_csv(augmented_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(build_summary(df), f, indent=2)

    if "object_type" in df.columns:
        by_object = (
            df.groupby("object_type", dropna=False)
            .agg(
                episodes=("success", "size"),
                success_rate=("success", "mean"),
                mean_reward=("episode_reward", "mean"),
                mean_abs_x_error_m=("abs_x_error", "mean"),
                mean_abs_y_error_m=("abs_y_error", "mean"),
            )
            .reset_index()
        )
        by_object_path = output_dir / "summary_by_object.csv"
        by_object.to_csv(by_object_path, index=False)

    return augmented_path, by_object_path, summary_path


def plot_episode_log(csv_path: Path, output_dir: Path, window: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    df = prepare_dataframe(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = [
        save_training_summary(df, output_dir, window),
        save_landing_plot(df, output_dir),
    ]
    action_path = save_action_plot(df, output_dir, window)
    if action_path is not None:
        paths.append(action_path)

    table_paths = write_tables(df, output_dir)
    paths.extend(path for path in table_paths if path is not None)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot episode_log.csv metrics.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to episode_log.csv or evaluation.csv")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing episode_log.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plot outputs")
    parser.add_argument("--window", type=int, default=100, help="Rolling mean window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv is not None:
        csv_path = args.csv
    elif args.run_dir is not None:
        csv_path = args.run_dir / "episode_log.csv"
    else:
        csv_path = find_latest_episode_csv(DEFAULT_RESULTS_DIR)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_dir = args.output_dir or (csv_path.parent / "plots")
    paths = plot_episode_log(csv_path=csv_path, output_dir=output_dir, window=max(1, args.window))
    print(f"Read CSV: {csv_path}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
