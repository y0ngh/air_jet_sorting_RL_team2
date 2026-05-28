# Modification Log

## 2026-05-21 00:04:28 KST

Request: read `/Users/yhseomac/Downloads/plan_weasy.pdf`, install `pypdf` in the
project virtual environment, then write code for reinforcement learning with the
Team 2 3D air-jet simulator.

Sources used:

- `/Users/yhseomac/Downloads/plan_weasy.pdf`
- `source/core_3d.py`
- `simulator/simulator_app.py`

Changes:

- Added `env/air_jet_env.py`, a Gymnasium-compatible one-step environment that
  maps a PPO action to `[Umax, sigma, t_on, duration]` and calls
  `simulate_rigid_body_3d`.
- Added `configs/air_jet_ppo.json` for experiment, PPO, and environment settings.
- Added `train/train_ppo.py` to train Stable-Baselines3 PPO with CSV logs,
  TensorBoard logs, checkpoints, final model, and VecNormalize stats.
- Added `eval/evaluate_ppo.py` to evaluate a trained model and save CSV/JSON
  results.
- Added package markers `env/__init__.py` and `utils/__init__.py`.
- Added backup manifest at `backups/rl_20260521_000428/BACKUP_MANIFEST.md`.

Backup:

- No existing code files were edited. The backup manifest records that all code
  files in this change are newly created.

Verification:

- `check_env` initially found an observation shape mismatch: the environment
  returned 27 values while `observation_space` declared 28.
- Backed up `env/air_jet_env.py` to
  `backups/rl_20260521_000428/air_jet_env.py.before_obs_shape_fix`.
- Fixed `observation_space` shape from `(28,)` to `(27,)`.
- Smoke training initially failed because `train/train_ppo.py` was executed from
  a subdirectory path and could not import the project-local `env` package.
- Backed up `train/train_ppo.py` and `eval/evaluate_ppo.py` before adding
  `PROJECT_ROOT` to `sys.path` in both scripts.
- Smoke training then failed because TensorBoard is not installed in the
  project `.venv`.
- Backed up `train/train_ppo.py` before making TensorBoard logging optional.
  CSV episode logging remains enabled.
- Final verification passed:
  - `python -m py_compile env/air_jet_env.py train/train_ppo.py eval/evaluate_ppo.py`
  - `stable_baselines3.common.env_checker.check_env(AirJetSortingEnv(...))`
  - `python train/train_ppo.py --smoke-test`
  - `python eval/evaluate_ppo.py --model results/rl/air_jet_ppo_baseline_20260521_000935/models/final_model.zip --vecnormalize results/rl/air_jet_ppo_baseline_20260521_000935/models/vecnormalize.pkl --episodes 3`
- Smoke run output directory:
  `results/rl/air_jet_ppo_baseline_20260521_000935`.

## 2026-05-21 00:11:54 KST

Request: install TensorBoard in the project `.venv` and make training use it.

Sources used:

- User request in the current thread.
- `train/train_ppo.py`, which already enables `tensorboard_log` when the
  `tensorboard` package can be imported.
- `pip install tensorboard` output.

Changes:

- Installed `tensorboard==2.20.0` into
  `/Users/yhseomac/Desktop/mcfl/release_week2_simulator_20260512_1657/Project3_Team2_0515/.venv`.
- Added `tensorboard` to `requirements.txt`.
- Added TensorBoard-related locked packages to `requirements_lock.txt`:
  `absl-py==2.3.1`, `grpcio==1.80.0`, `Markdown==3.9`,
  `tensorboard==2.20.0`, `tensorboard-data-server==0.7.2`,
  and `Werkzeug==3.1.8`.

Backup:

- Backed up dependency files and this log under
  `backups/tensorboard_20260521_001154/`.

Verification:

- Confirmed TensorBoard import in `.venv`: `tensorboard.__version__ == 2.20.0`.
- Ran `python -m py_compile train/train_ppo.py`.
- Ran `python train/train_ppo.py --smoke-test`.
- Smoke training printed TensorBoard log destination:
  `results/rl/air_jet_ppo_baseline_20260521_001337/logs/ppo_1`.
- Confirmed TensorBoard event file was created:
  `results/rl/air_jet_ppo_baseline_20260521_001337/logs/ppo_1/events.out.tfevents.1779290018.yonghyeon-uiui-MacBookAir.local.21779.0`.
