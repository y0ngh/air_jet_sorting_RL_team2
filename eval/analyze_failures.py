"""Summarize success, short-fail, and long-fail cases from RL CSV outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "rl"
sys.path.insert(0, str(PROJECT_ROOT))

from eval.plot_episode_log import prepare_dataframe

FEATURE_COLS = [
    "mass",
    "size_x",
    "size_y",
    "size_z",
    "vx",
    "target_x_min",
    "target_x_max",
    "target_width",
]
ACTION_COLS = ["action_umax", "action_sigma", "action_t_on", "action_duration"]


def find_latest_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("*/episode_log.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No episode_log.csv found under {results_dir}")
    return candidates[0]


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_dataframe(df)
    df["short_fail"] = (~df["success"]) & (df["landing_x"] < df["target_x_min"])
    df["long_fail"] = (~df["success"]) & (df["landing_x"] > df["target_x_max"])
    df["outcome"] = "other_fail"
    df.loc[df["short_fail"], "outcome"] = "short_fail"
    df.loc[df["long_fail"], "outcome"] = "long_fail"
    df.loc[df["success"], "outcome"] = "success"
    return df


def write_tables(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    detailed_path = output_dir / "failure_cases_with_inputs.csv"
    df.to_csv(detailed_path, index=False)
    paths.append(detailed_path)

    available_features = [col for col in FEATURE_COLS if col in df.columns]
    available_actions = [col for col in ACTION_COLS if col in df.columns]
    summary_cols = available_features + ["landing_x", "normalized_x_error", "episode_reward"] + available_actions
    summary = df.groupby(["outcome", "object_type"], dropna=False)[summary_cols].agg(["count", "mean", "median", "std"])
    summary_path = output_dir / "failure_summary_by_outcome_object.csv"
    summary.to_csv(summary_path)
    paths.append(summary_path)

    counts = pd.crosstab(df["object_type"], df["outcome"], normalize="index")
    counts_path = output_dir / "failure_outcome_fraction_by_object.csv"
    counts.to_csv(counts_path)
    paths.append(counts_path)
    return paths


def save_feature_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    cols = [col for col in FEATURE_COLS + ["landing_x", "normalized_x_error"] if col in df.columns]
    outcomes = ["success", "short_fail", "long_fail", "other_fail"]
    present = [name for name in outcomes if name in set(df["outcome"])]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    for ax, col in zip(axes.ravel(), cols):
        values = [df.loc[df["outcome"] == outcome, col].dropna() for outcome in present]
        ax.boxplot(values, tick_labels=present, showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")

    output_path = output_dir / "failure_input_boxplots.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_landing_error_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    color_map = {"success": "tab:green", "short_fail": "tab:orange", "long_fail": "tab:red", "other_fail": "tab:gray"}
    colors = df["outcome"].map(color_map).fillna("tab:gray")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    ax.scatter(df["target_x_min"], df["landing_x"], c=colors, s=12, alpha=0.45, edgecolors="none")
    lims = [
        float(np.nanmin([df["target_x_min"].min(), df["landing_x"].min()])),
        float(np.nanmax([df["target_x_min"].max(), df["landing_x"].max()])),
    ]
    ax.plot(lims, lims, color="black", linewidth=1, alpha=0.5, label="landing_x = target_x_min")
    ax.set_xlabel("target_x_min [m]")
    ax.set_ylabel("landing_x [m]")
    ax.set_title("Short Fail Boundary")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    for outcome, group in df.groupby("outcome"):
        delta = (group["landing_x"] - group["target_x_min"]).dropna()
        if delta.empty:
            continue
        ax.hist(delta, bins=40, alpha=0.55, label=outcome)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("landing_x - target_x_min [m]")
    ax.set_ylabel("Episodes")
    ax.set_title("Distance From Target Entry")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "failure_landing_error.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def analyze_failures(csv_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = add_outcomes(pd.read_csv(csv_path))
    paths = write_tables(df, output_dir)
    paths.append(save_feature_boxplot(df, output_dir))
    paths.append(save_landing_error_plot(df, output_dir))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze short/long failure cases from RL episode CSV.")
    parser.add_argument("--csv", type=Path, default=None, help="episode_log.csv or policy_action_analysis.csv")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing episode_log.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv is not None:
        csv_path = args.csv
    elif args.run_dir is not None:
        csv_path = args.run_dir / "episode_log.csv"
    else:
        csv_path = find_latest_csv(DEFAULT_RESULTS_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    output_dir = args.output_dir or (csv_path.parent / "plots")
    paths = analyze_failures(csv_path, output_dir)
    print(f"Read CSV: {csv_path}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
