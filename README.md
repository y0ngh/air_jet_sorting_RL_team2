# Air Jet Sorting RL

PPO reinforcement learning setup for the Project 3 Team 2 3D air-jet sorting simulator.

This repository contains only the files needed to inspect, modify, train, and evaluate
the current RL version. Generated training results, virtual environments, and model
checkpoints are intentionally excluded.

## Current Experiment

Active config:

```bash
configs/fixed_y0_z020_vx100_target04_09_relaxed_reward_env8_100k.json
```

Key settings:

- Initial position: `y0 = 0`, `z0 = 0.20 m`
- Initial velocity: `vx = 1.0 m/s`, `vy = 0`, `vz = 0`
- Target region: `x = 0.4 ~ 0.9 m`
- Fixed jet position: `x = 0.105 m`, `y = 0`, `z = 0.21 m`
- Jet angle: `60 deg`
- PPO action variables: `umax`, `sigma`, `t_on`, `duration`
- Current `umax` range: `0 ~ 30 m/s`

## Repository Layout

```text
source/      3D rigid-body simulator core
env/         Gymnasium RL environment wrapper
train/       PPO training script
eval/        evaluation, plotting, failure analysis, sweep tools
simulator/   Streamlit simulator UI
assets/      simulator icons
configs/     active RL config
```

## Setup

```bash
python3 -m venv rl_env
source rl_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

```bash
env PYTHONNOUSERSITE=1 rl_env/bin/python train/train_ppo.py \
  --config configs/fixed_y0_z020_vx100_target04_09_relaxed_reward_env8_100k.json \
  --n-envs 8
```

Training outputs are written under:

```text
results/rl/<experiment_name>_<YYYYMMDD_HHMMSS>/
```

## Evaluate

```bash
RUN=$(ls -td results/rl/fixed_y0_z020_vx100_target04_09_relaxed_reward_env8_100k_* | head -1)
echo "$RUN"

env PYTHONNOUSERSITE=1 rl_env/bin/python eval/analyze_policy_actions.py \
  --run-dir "$RUN" \
  --episodes 500 \
  --max-checkpoints 8 \
  --output-dir "$RUN/plots_eval500"

env PYTHONNOUSERSITE=1 rl_env/bin/python eval/analyze_failures.py \
  --csv "$RUN/plots_eval500/policy_action_analysis.csv" \
  --output-dir "$RUN/plots_eval500_failures"

env PYTHONNOUSERSITE=1 rl_env/bin/python eval/plot_episode_log.py \
  --csv "$RUN/episode_log.csv" \
  --output-dir "$RUN/plots"
```

## Run Simulator UI

```bash
env PYTHONNOUSERSITE=1 rl_env/bin/python -m streamlit run simulator/simulator_app.py
```

## Notes

- GPU acceleration is usually not the bottleneck here because each RL step runs
  a NumPy-based rigid-body simulation.
- `results/`, `rl_env/`, checkpoints, TensorBoard logs, and cache files are ignored.
- The `umax=0~40` experiment is not included here; this repo is the latest `umax=0~30`
  version requested for team editing.
