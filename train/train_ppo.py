"""Train PPO on the air-jet sorting environment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "air_jet_ppo.json"
sys.path.insert(0, str(PROJECT_ROOT))

from env.air_jet_env import AirJetSortingEnv


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class EpisodeCsvCallback(BaseCallback):
    """Collect episode summaries from Monitor and save them as CSV."""

    def __init__(self, output_path: Path, flush_every: int = 10):
        super().__init__()
        self.output_path = output_path
        self.flush_every = flush_every
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            row = {
                "timestep": self.num_timesteps,
                "episode_reward": episode_info["r"],
                "episode_length": episode_info["l"],
                "success": info.get("success"),
                "has_landed": info.get("has_landed"),
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
            self.rows.append(row)

        if self.rows and len(self.rows) % self.flush_every == 0:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.output_path, index=False)


def make_env(config: Dict[str, Any], seed: int, rank: int = 0):
    def _factory():
        env = AirJetSortingEnv(config=config.get("env", {}), seed=seed + rank)
        return Monitor(env)

    return _factory


def train(
    config_path: Path | None = None,
    smoke_test: bool = False,
    resume_run: Path | None = None,
    additional_timesteps: int | None = None,
    experiment_name_override: str | None = None,
    n_envs: int = 1,
) -> Path:
    if config_path is None:
        config_path = (resume_run / "config_used.json") if resume_run is not None else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    is_resume = resume_run is not None
    if is_resume:
        resume_run = resume_run.resolve()
        model_path = resume_run / "models" / "final_model.zip"
        vecnormalize_path = resume_run / "models" / "vecnormalize.pkl"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not vecnormalize_path.exists():
            raise FileNotFoundError(vecnormalize_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_name_override or config.get("experiment_name", "air_jet_ppo")
    run_dir = PROJECT_ROOT / "results" / "rl" / f"{experiment_name}_{run_id}"
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    n_envs = max(1, int(n_envs))
    if smoke_test:
        total_timesteps = 64
    elif additional_timesteps is not None:
        total_timesteps = int(additional_timesteps)
    else:
        total_timesteps = int(config.get("total_timesteps", 20000))

    if experiment_name_override is not None:
        config["experiment_name"] = experiment_name_override
    config["n_envs"] = n_envs
    if is_resume:
        config["continued_from"] = str(resume_run)
        config["additional_timesteps"] = total_timesteps

    with (run_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    try:
        import tensorboard  # noqa: F401

        tensorboard_log = str(log_dir)
    except ImportError:
        tensorboard_log = None
        print("TensorBoard is not installed. Continuing with CSV logs only.")

    ppo_cfg = dict(config.get("ppo", {}))
    raw_policy_kwargs = ppo_cfg.pop("policy_kwargs", {})

    env_fns = [make_env(config, seed, rank) for rank in range(n_envs)]
    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="fork")
    if is_resume:
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(
            model_path,
            env=env,
            device=ppo_cfg.get("device", "auto"),
        )
        model.verbose = 1
        model.tensorboard_log = tensorboard_log
        print(f"Continuing training from: {resume_run}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)
        policy_kwargs = {
            "activation_fn": th.nn.ReLU,
            "net_arch": raw_policy_kwargs.get("net_arch", {"pi": [128, 128], "vf": [128, 128]}),
        }

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            **ppo_cfg,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_air_jet",
        save_vecnormalize=True,
    )
    csv_callback = EpisodeCsvCallback(run_dir / "episode_log.csv")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, csv_callback],
        tb_log_name="ppo_resume" if is_resume else "ppo",
        reset_num_timesteps=not is_resume,
    )

    final_model_path = model_dir / "final_model.zip"
    final_vecnormalize_path = model_dir / "vecnormalize.pkl"
    model.save(final_model_path)
    env.save(final_vecnormalize_path)
    env.close()

    print(f"Saved final model: {final_model_path}")
    print(f"Saved VecNormalize stats: {final_vecnormalize_path}")
    print(f"Saved logs/results under: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-run", type=Path, default=None, help="Run directory with models/final_model.zip to continue.")
    parser.add_argument("--additional-timesteps", type=int, default=None, help="Timesteps to train after --resume-run.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override output run prefix.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        smoke_test=args.smoke_test,
        resume_run=args.resume_run,
        additional_timesteps=args.additional_timesteps,
        experiment_name_override=args.experiment_name,
        n_envs=args.n_envs,
    )
