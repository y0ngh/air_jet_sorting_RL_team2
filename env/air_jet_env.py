"""Gymnasium wrapper for the 3D rigid-body air-jet simulator.

The environment treats one full simulator run as one RL step. The policy chooses
four continuous jet parameters: Umax, sigma, t_on, and duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from source.core_3d import (
    InitialCondition3D,
    Jet3D,
    Simulation3D,
    TargetRegion3D,
    create_object_3d,
    euler_degrees_to_quaternion,
    simulate_rigid_body_3d,
)


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def scale_from_unit_action(self, value: float) -> float:
        clipped = float(np.clip(value, -1.0, 1.0))
        return self.low + 0.5 * (clipped + 1.0) * (self.high - self.low)

    def normalize(self, value: float) -> float:
        if abs(self.high - self.low) < 1.0e-12:
            return 0.0
        return float(2.0 * (value - self.low) / (self.high - self.low) - 1.0)


DEFAULT_CONFIG: Dict[str, Any] = {
    "object_types": ["plate", "rod", "irregular"],
    "ranges": {
        "mass": [0.001, 0.010],
        "drag_coefficient": [0.6, 1.6],
        "size_x": [0.05, 0.18],
        "size_y": [0.04, 0.16],
        "size_z": [0.006, 0.04],
        "rod_length": [0.08, 0.22],
        "rod_radius": [0.008, 0.04],
        "x0": [0.0, 0.0],
        "y0": [-0.03, 0.03],
        "z0": [0.20, 0.20],
        "vx": [1.0, 1.0],
        "vy": [0.0, 0.0],
        "vz": [0.0, 0.0],
        "roll": [-20.0, 20.0],
        "pitch": [-20.0, 20.0],
        "yaw": [-30.0, 30.0],
        "omega_x": [-5.0, 5.0],
        "omega_y": [-5.0, 5.0],
        "omega_z": [-5.0, 5.0],
        "target_x_min": [0.30, 0.40],
        "target_width": [0.25, 0.45],
    },
    "fixed_jet": {
        "x_center": 0.16,
        "y_center": 0.00,
        "z_center": 0.209,
        "axial_decay": 0.35,
        "angle_deg": 60.0,
        "azimuth_deg": 0.0,
        "noise_std": 0.0,
    },
    "action_ranges": {
        "umax": [10.0, 30.0],
        "sigma": [0.01, 0.05],
        "t_on": [0.0, 0.5],
        "duration": [0.01, 0.10],
    },
    "simulation": {
        "dt": 0.001,
        "t_max": 2.0,
        "gravity": 9.81,
        "air_density": 1.225,
        "landing_z": 0.0,
        "conveyor_length": 0.15,
        "free_fall_start_offset": 0.03,
    },
    "reward": {
        "success_bonus": 10.0,
        "miss_penalty": -2.0,
        "not_landed_penalty": -5.0,
        "center_distance_weight": 1.0,
        "short_distance_weight": 8.0,
        "long_distance_weight": 3.0,
        "x_error_clip": 3.0,
        "jet_cost_weight": 0.4,
        "angular_speed_weight": 0.01,
    },
}


class AirJetSortingEnv(gym.Env):
    """One-step PPO environment backed by the project 3D air-jet simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = self._merge_config(DEFAULT_CONFIG, config or {})
        self.ranges = {
            key: Range(float(value[0]), float(value[1]))
            for key, value in self.config["ranges"].items()
        }
        self.action_ranges = {
            key: Range(float(value[0]), float(value[1]))
            for key, value in self.config["action_ranges"].items()
        }
        self.object_types = list(self.config["object_types"])
        self.rng = np.random.default_rng(seed)
        self.episode_index = 0
        self.current_case: Optional[Dict[str, Any]] = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(27,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_index += 1
        self.current_case = self._sample_case(options or {})
        observation = self._build_observation(self.current_case)
        info = {
            "episode_index": self.episode_index,
            "object_type": self.current_case["object_type"],
            "target_x_min": self.current_case["target_x_min"],
            "target_x_max": self.current_case["target_x_max"],
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_case is None:
            raise RuntimeError("reset() must be called before step().")

        jet_params = self._action_to_jet_params(action)
        result = self._run_simulation(self.current_case, jet_params)
        reward = self._compute_reward(result, jet_params)

        landing_position = result["landing_position"]
        info = {
            "success": bool(result["success"]) if result["success"] is not None else False,
            "has_landed": bool(result["has_landed"]),
            "landing_x": float(landing_position[0]) if landing_position is not None else np.nan,
            "landing_y": float(landing_position[1]) if landing_position is not None else np.nan,
            "landing_time": float(result["landing_time"]) if result["landing_time"] is not None else np.nan,
            "max_angular_speed": float(result["max_angular_speed"]),
            "jet_impulse_norm": float(np.linalg.norm(result["jet_impulse"])),
            "action_physical": jet_params,
            "object_type": self.current_case["object_type"],
            "target_x_min": self.current_case["target_x_min"],
            "target_x_max": self.current_case["target_x_max"],
            "simulation_failed": bool(result.get("simulation_failed", False)),
            "simulation_error": result.get("simulation_error", ""),
        }

        observation = self._build_observation(self.current_case)
        return observation, float(reward), True, False, info

    @staticmethod
    def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def _sample_case(self, options: Dict[str, Any]) -> Dict[str, Any]:
        object_type = str(options.get("object_type", self.rng.choice(self.object_types)))
        mass = self.ranges["mass"].sample(self.rng)
        drag = self.ranges["drag_coefficient"].sample(self.rng)

        if object_type == "rod":
            rod_length = self.ranges["rod_length"].sample(self.rng)
            rod_radius = self.ranges["rod_radius"].sample(self.rng)
            size_x = rod_length
            size_y = 2.0 * rod_radius
            size_z = 2.0 * rod_radius
        else:
            rod_length = None
            rod_radius = None
            size_x = self.ranges["size_x"].sample(self.rng)
            size_y = self.ranges["size_y"].sample(self.rng)
            size_z = self.ranges["size_z"].sample(self.rng)

        target_x_min = self.ranges["target_x_min"].sample(self.rng)
        target_width = self.ranges["target_width"].sample(self.rng)

        return {
            "object_type": object_type,
            "mass": mass,
            "drag_coefficient": drag,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "rod_length": rod_length,
            "rod_radius": rod_radius,
            "x0": self.ranges["x0"].sample(self.rng),
            "y0": self.ranges["y0"].sample(self.rng),
            "z0": self.ranges["z0"].sample(self.rng),
            "vx": self.ranges["vx"].sample(self.rng),
            "vy": self.ranges["vy"].sample(self.rng),
            "vz": self.ranges["vz"].sample(self.rng),
            "roll": self.ranges["roll"].sample(self.rng),
            "pitch": self.ranges["pitch"].sample(self.rng),
            "yaw": self.ranges["yaw"].sample(self.rng),
            "omega_x": self.ranges["omega_x"].sample(self.rng),
            "omega_y": self.ranges["omega_y"].sample(self.rng),
            "omega_z": self.ranges["omega_z"].sample(self.rng),
            "target_x_min": target_x_min,
            "target_x_max": target_x_min + target_width,
            "seed": int(self.rng.integers(0, 2**31 - 1)),
        }

    def _build_observation(self, case: Dict[str, Any]) -> np.ndarray:
        one_hot = [1.0 if case["object_type"] == name else 0.0 for name in self.object_types]
        quat = euler_degrees_to_quaternion(case["roll"], case["pitch"], case["yaw"])
        target_center = 0.5 * (case["target_x_min"] + case["target_x_max"])
        target_width = case["target_x_max"] - case["target_x_min"]

        values = [
            *one_hot,
            self.ranges["mass"].normalize(case["mass"]),
            self.ranges["drag_coefficient"].normalize(case["drag_coefficient"]),
            self.ranges["size_x"].normalize(case["size_x"]),
            self.ranges["size_y"].normalize(case["size_y"]),
            self.ranges["size_z"].normalize(case["size_z"]),
            self.ranges["x0"].normalize(case["x0"]),
            self.ranges["y0"].normalize(case["y0"]),
            self.ranges["z0"].normalize(case["z0"]),
            self.ranges["vx"].normalize(case["vx"]),
            self.ranges["vy"].normalize(case["vy"]),
            self.ranges["vz"].normalize(case["vz"]),
            self.ranges["omega_x"].normalize(case["omega_x"]),
            self.ranges["omega_y"].normalize(case["omega_y"]),
            self.ranges["omega_z"].normalize(case["omega_z"]),
            case["roll"] / 180.0,
            case["pitch"] / 180.0,
            case["yaw"] / 180.0,
            *quat,
            self.ranges["target_x_min"].normalize(case["target_x_min"]),
            self.ranges["target_width"].normalize(target_width),
            target_center,
        ]
        return np.asarray(values, dtype=np.float32)

    def _action_to_jet_params(self, action: np.ndarray) -> Dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return {
            "umax": self.action_ranges["umax"].scale_from_unit_action(action[0]),
            "sigma": self.action_ranges["sigma"].scale_from_unit_action(action[1]),
            "t_on": self.action_ranges["t_on"].scale_from_unit_action(action[2]),
            "duration": self.action_ranges["duration"].scale_from_unit_action(action[3]),
        }

    def _run_simulation(self, case: Dict[str, Any], jet_params: Dict[str, float]) -> Dict[str, Any]:
        obj = create_object_3d(
            object_type=case["object_type"],
            mass=case["mass"],
            size_x=case["size_x"],
            size_y=case["size_y"],
            size_z=case["size_z"],
            drag_coefficient=case["drag_coefficient"],
            rod_length=case["rod_length"],
            rod_radius=case["rod_radius"],
            seed=case["seed"],
        )

        fixed_jet = self.config["fixed_jet"]
        jet = Jet3D(
            umax=jet_params["umax"],
            x_center=float(fixed_jet["x_center"]),
            y_center=float(fixed_jet["y_center"]),
            z_center=float(fixed_jet["z_center"]),
            sigma=jet_params["sigma"],
            axial_decay=float(fixed_jet["axial_decay"]),
            angle_deg=float(fixed_jet["angle_deg"]),
            azimuth_deg=float(fixed_jet["azimuth_deg"]),
            t_on=jet_params["t_on"],
            duration=jet_params["duration"],
            noise_std=float(fixed_jet["noise_std"]),
        )

        sim_cfg = self.config["simulation"]
        sim = Simulation3D(
            dt=float(sim_cfg["dt"]),
            t_max=float(sim_cfg["t_max"]),
            gravity=float(sim_cfg["gravity"]),
            air_density=float(sim_cfg["air_density"]),
            landing_z=float(sim_cfg["landing_z"]),
            conveyor_length=float(sim_cfg["conveyor_length"]),
            free_fall_start_offset=float(sim_cfg["free_fall_start_offset"]),
        )

        initial = InitialCondition3D(
            position=(case["x0"], case["y0"], case["z0"]),
            velocity=(case["vx"], case["vy"], case["vz"]),
            quaternion=euler_degrees_to_quaternion(case["roll"], case["pitch"], case["yaw"]),
            angular_velocity=(case["omega_x"], case["omega_y"], case["omega_z"]),
        )

        target = TargetRegion3D(x_min=case["target_x_min"], x_max=case["target_x_max"])

        try:
            return simulate_rigid_body_3d(
                obj=obj,
                jet=jet,
                sim=sim,
                initial=initial,
                target=target,
                seed=case["seed"],
            )
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
            # Very strong jets can occasionally make the rigid-body integrator
            # numerically unstable. Treat that trial as a failed action instead
            # of aborting the whole PPO run.
            return {
                "position": np.empty((0, 3), dtype=float),
                "velocity": np.empty((0, 3), dtype=float),
                "quaternion": np.empty((0, 4), dtype=float),
                "angular_velocity": np.empty((0, 3), dtype=float),
                "time": np.empty((0,), dtype=float),
                "landing_position": None,
                "landing_time": None,
                "has_landed": False,
                "success": False,
                "target": target,
                "jet_impulse": np.zeros(3, dtype=float),
                "angular_impulse": np.zeros(3, dtype=float),
                "max_angular_speed": 0.0,
                "simulation_failed": True,
                "simulation_error": str(exc),
            }

    def _compute_reward(self, result: Dict[str, Any], jet_params: Dict[str, float]) -> float:
        reward_cfg = self.config["reward"]
        if not result["has_landed"] or result["landing_position"] is None:
            reward = float(reward_cfg["not_landed_penalty"])
            if "min_reward" in reward_cfg:
                reward = max(reward, float(reward_cfg["min_reward"]))
            return reward

        landing = result["landing_position"]
        target = result["target"]
        landing_x = float(landing[0])
        target_width = max(target.x_max - target.x_min, 1.0e-6)
        norm_width = max(float(reward_cfg.get("x_error_normalization_width", target_width)), 1.0e-6)
        target_center = 0.5 * (target.x_min + target.x_max)
        clip = float(reward_cfg.get("x_error_clip", 3.0))

        if target.x_min <= landing_x <= target.x_max:
            center_error = min(abs(landing_x - target_center) / (0.5 * norm_width), clip)
            reward = float(reward_cfg["success_bonus"])
            reward -= float(reward_cfg["center_distance_weight"]) * (center_error**2)
        elif landing_x < target.x_min:
            short_error = min((target.x_min - landing_x) / norm_width, clip)
            reward = float(reward_cfg["miss_penalty"])
            reward -= float(reward_cfg["short_distance_weight"]) * ((1.0 + short_error) ** 2)
        else:
            long_error = min((landing_x - target.x_max) / norm_width, clip)
            reward = float(reward_cfg["miss_penalty"])
            reward -= float(reward_cfg["long_distance_weight"]) * (long_error**2)

        jet_cost = float(reward_cfg["jet_cost_weight"]) * (
            jet_params["umax"] / max(self.action_ranges["umax"].high, 1.0e-6)
        ) * jet_params["duration"]
        angular_cost = float(reward_cfg["angular_speed_weight"]) * float(result["max_angular_speed"])
        reward -= jet_cost + angular_cost

        if "min_reward" in reward_cfg:
            reward = max(reward, float(reward_cfg["min_reward"]))

        return float(reward)
