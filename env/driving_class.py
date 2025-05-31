from __future__ import annotations

from abc import abstractmethod # For GoalEnv interface

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType # REMOVED Action import
from gymnasium import Env # For GoalEnv interface
from typing import Any # For more general type hinting if needed

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory, KinematicsGoalObservation
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env.vehicle.graphics import VehicleGraphics
from gymnasium.envs.registration import register

# User-provided GoalEnv interface
class GoalEnv(Env):
    """ Interface for A goal-based environment. (Docstring from user) """
    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. (Docstring from user)"""
        raise NotImplementedError

class DrivingClassEnv(AbstractEnv, GoalEnv):
    """
    Goal-conditioned Driving Class scenario with Lidar observation and sequential goals.
    Agent uses Lidar for local perception and kinematic goals for task completion.
    """
    MIN_LANE_LEN_FOR_EDGE_OBSTACLES = 0.5

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        self.goal_landmark: Landmark | None = None
        self._lane_ids: list[tuple[str, str, int]] = []
        self.goal_sequence: list[tuple[str, str, int]] = []
        self.current_goal_index: int = 0
        self.last_step_goal_met: bool = False
        super().__init__(config, render_mode)


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "TupleObservation",
                    "observation_configs": [
                        {
                            "type": "LidarObservation",
                            "angle_range": [-np.pi, np.pi], # Full 360-degree Lidar
                            "cells": 30,                     # Number of Lidar beams
                            "max_distance": 50.0,            # Max Lidar distance (meters)
                            "normalize": True,               # Normalize Lidar distances to [0, 1]
                            "see_behind": True,              # Allow Lidar to see behind
                        },
                        {
                            "type": "KinematicsGoal",
                            "features": ["x", "y", "cos_h", "sin_h", "vx", "vy"], # Fallback/default for 'observation' field
                            "scales": [100, 100, 1, 1, 10, 10],
                            "normalize": True # Normalization for the wrapper itself if it were to use its own kinematics for 'observation'
                        }
                    ],
                },
                "action": {
                    "type": "ContinuousAction", "longitudinal": True, "lateral": True,
                    "acceleration_range": [-5, 5], "steering_range": [-np.pi / 3, np.pi / 3],
                },
                "simulation_frequency": 15, "policy_frequency": 5,
                "duration": 600,
                "reward_weights": np.array([1.0, 1.0, 0.02, 0.02, 0, 0]),
                "collision_reward": -150.0,
                "action_penalty_weight": -0.02,
                "on_road_shaping_reward": 0.005,
                "lane_centering_shaping_reward": 0.005,
                "lane_centering_cost_factor": 4,
                "controlled_vehicles": 1, "other_vehicles": 0,
                "goal_sequence": [
                    {"lane_tuple": ("b", "c", 2), "target_heading_rad": None},          # Goal 1
                    {"lane_tuple": ("e", "f", 0), "target_heading_rad": np.pi / 2},       # Goal 2 (Reversing bay, face West)
                    {"lane_tuple": ("b", "c", 2), "target_heading_rad": None},          # Goal 3
                    {"lane_tuple": ("f", "g", 0), "target_heading_rad": 0}    # Goal 4 (Parking, face South)
                ],
                "intermediate_goal_reward": 75.0,
                "final_goal_completion_reward": 150.0,
                "goal_longitudinal_offset": 0.5, "goal_heading_noise_std": np.deg2rad(3),
                "goal_position_noise_std": 0.1, "success_distance_threshold": 0.03,
                "success_heading_threshold_rad": np.deg2rad(15),
                "screen_width": 1200, "screen_height": 900, "centering_position": [0.3, 0.6], "scaling": 3.5,
                "lane_width": 6.0, "show_trajectories": False, "offroad_terminal": False,
                "x_offset": 0, "y_offset": 0,
                "road_segment_size": 80, "road_segment_gap": 15,
                "road_extra_length": [20], "start_lane_index": 1,
                "manual_control": True,
                "real_time_rendering": True,

            }
        )
        return config

    def _set_destination(self) -> None:
        if not self.road or not self._lane_ids or self.vehicle is None:
            if hasattr(self, 'goal_landmark') and self.goal_landmark and self.road and self.goal_landmark in self.road.objects:
                self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = None
            if self.vehicle: self.vehicle.goal = None
            return

        if self.goal_landmark and self.road and self.goal_landmark in self.road.objects:
            self.road.objects.remove(self.goal_landmark)
        self.goal_landmark = None
        if self.vehicle: self.vehicle.goal = None

        if self.current_goal_index >= len(self.goal_sequence): return

        current_goal_config = self.goal_sequence[self.current_goal_index]
        current_goal_config_tuple = current_goal_config["lane_tuple"]
        target_heading_override_rad = current_goal_config.get("target_heading_rad")

        if current_goal_config_tuple not in self._lane_ids: return

        try:
            goal_lane_object = self.road.network.get_lane(current_goal_config_tuple)
            long_offset_factor = np.clip(self.config["goal_longitudinal_offset"], 0.05, 0.95)
            goal_pos = goal_lane_object.position(goal_lane_object.length * long_offset_factor, 0)
            if target_heading_override_rad is not None:
                goal_head = float(target_heading_override_rad)
            else:
                goal_head = goal_lane_object.heading_at(goal_lane_object.length * long_offset_factor)
            goal_pos[0] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_pos[1] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_head += self.np_random.normal(0, self.config["goal_heading_noise_std"])

            self.goal_landmark = Landmark(self.road, goal_pos, heading=goal_head)
            self.goal_landmark.color=VehicleGraphics.GREEN
            self.goal_landmark.length=self.config["lane_width"] * 0.7
            self.goal_landmark.width=self.config["lane_width"] * 0.7
            if self.road: self.road.objects.append(self.goal_landmark)
            if self.vehicle: self.vehicle.goal = self.goal_landmark
        except KeyError:
            if self.vehicle: self.vehicle.goal = None

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> np.ndarray: # Changed return type
        if desired_goal is None:
            # This case should ideally not happen with HER providing goals.
            # If it can, ensure it returns an array of appropriate size if achieved_goal is batched.
            # For simplicity, assuming desired_goal is always present from HER.
            # If achieved_goal is batched, this needs to return an array of zeros.
            if achieved_goal.ndim > 1:
                return np.zeros(achieved_goal.shape[0], dtype=np.float32)
            return 0.0 # Or handle as an error/warning

        # Ensure inputs are at least 1D, and then promote to 2D for consistent batch processing
        achieved_goal_arr = np.atleast_1d(achieved_goal)
        desired_goal_arr = np.atleast_1d(desired_goal)

        is_batched_input = achieved_goal_arr.ndim > 1 and achieved_goal_arr.shape[0] > 1 # A more robust check for batch
        if not is_batched_input and achieved_goal_arr.ndim == 1 : # Single transition, reshape to (1, goal_dim)
            achieved_goal_arr = achieved_goal_arr.reshape(1, -1)
            desired_goal_arr = desired_goal_arr.reshape(1, -1)
        elif achieved_goal_arr.ndim == 0: # Scalar inputs, very unlikely but guard
            achieved_goal_arr = achieved_goal_arr.reshape(1,1)
            desired_goal_arr = desired_goal_arr.reshape(1,1)


        batch_size, goal_dim = achieved_goal_arr.shape

        current_reward_weights = np.array(self.config.get("reward_weights", [])) # Use .get for safety

        diff = achieved_goal_arr - desired_goal_arr # Shape: (batch_size, goal_dim)

        # Condition to use weights should be based on whether weights are valid for the goal_dim
        if len(current_reward_weights) == goal_dim:
            # Weighted calculation
            # Element-wise multiplication then sum over goal_dim
            # np.abs(diff) is (batch_size, goal_dim)
            # current_reward_weights (reshaped to (1, goal_dim) for broadcasting)
            weighted_abs_diff_sum = np.sum(np.abs(diff) * current_reward_weights.reshape(1, goal_dim), axis=1) # Result shape: (batch_size,)
            rewards = -np.power(weighted_abs_diff_sum, p)
        else:
            # Unweighted calculation (using norm)
            # np.linalg.norm needs axis=1 for batched input to get norm for each item in batch
            distance = np.linalg.norm(diff, axis=1) # Result shape: (batch_size,)
            rewards = -np.power(distance, p)
        
        if not is_batched_input and rewards.shape == (1,): # If original input was single, return scalar
            return rewards[0]
        return rewards # Return 1D array of shape (batch_size,)

    def _reward(self, action: np.ndarray) -> float: # Changed Action to np.ndarray
        if self.observation_type is None or self.vehicle is None: return 0.0

        obs_tuple_for_current_goal_eval = self.observation_type.observe()
        kinematics_data_dict = obs_tuple_for_current_goal_eval[1]
        achieved_g = kinematics_data_dict["achieved_goal"]
        desired_g_this_step = kinematics_data_dict["desired_goal"]

        total_reward = 0.0
        self.last_step_goal_met = False

        if desired_g_this_step is not None and self.current_goal_index < len(self.goal_sequence):
            self.last_step_goal_met = self._is_success(achieved_g, desired_g_this_step)
            info_for_compute = {"is_crashed": self.vehicle.crashed, "is_success": self.last_step_goal_met}
            total_reward += self.compute_reward(achieved_g, desired_g_this_step, info_for_compute)
            # print(f"Distance reward: {total_reward}")

            if self.last_step_goal_met:
                is_this_the_final_goal_in_sequence = (self.current_goal_index == len(self.goal_sequence) - 1)
                if is_this_the_final_goal_in_sequence:
                    total_reward += self.config.get("final_goal_completion_reward", self.config["intermediate_goal_reward"])
                else:
                    total_reward += self.config["intermediate_goal_reward"]
                self.current_goal_index += 1
                self._set_destination()
        
        if self.vehicle.crashed:
            total_reward += self.config["collision_reward"]
        total_reward += self.config["action_penalty_weight"] * np.linalg.norm(action)
        # print(f"total reward: {total_reward}")
        # on_road_value = float(self.vehicle.on_road)
        # total_reward += self.config["on_road_shaping_reward"] * on_road_value
        
        # lane_centering_value = 0.0
        # if self.vehicle.on_road and self.vehicle.lane_index and self.road:
        #     try:
        #         current_lane = self.road.network.get_lane(self.vehicle.lane_index)
        #         _, lateral_offset = current_lane.local_coordinates(self.vehicle.position)
        #         lane_centering_value = 1 / (1 + self.config["lane_centering_cost_factor"] * lateral_offset**2)
        #     except (KeyError, AttributeError): pass
        # total_reward += self.config["lane_centering_shaping_reward"] * lane_centering_value
        
        return float(total_reward)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        if desired_goal is None: return False
        dist_sq = np.sum(np.square(achieved_goal[0:2] - desired_goal[0:2]))
        position_ok = dist_sq < self.config["success_distance_threshold"]**2
        if len(achieved_goal) >= 4 and len(desired_goal) >= 4:
            cos_delta_angle = np.clip(np.dot(achieved_goal[2:4], desired_goal[2:4]), -1.0, 1.0)
            heading_ok = cos_delta_angle >= np.cos(self.config["success_heading_threshold_rad"])
        else: heading_ok = True
        return bool(position_ok and heading_ok)

    def _is_terminated(self) -> bool:
        if self.vehicle is None: return True
        crashed = self.vehicle.crashed
        off_road = self.config["offroad_terminal"] and not self.vehicle.on_road
        all_goals_completed = self.current_goal_index >= len(self.goal_sequence)
        return bool(crashed or off_road or all_goals_completed)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> tuple[ObsType, dict]:
        self.time = 0; self.terminated = False; self.truncated = False
        self.last_step_goal_met = False
        self.goal_sequence = list(self.config.get("goal_sequence", []))
        self.current_goal_index = 0

        self._create_road(); self._make_vehicles(); self._make_walls()
        self._set_destination()

        action_config = self.config["action"]
        if self.vehicle and not action_config.get("longitudinal", True) and hasattr(self.vehicle, 'target_speed'):
            target_speeds = action_config.get("target_speeds")
            if target_speeds and len(target_speeds) > 0:
                self.vehicle.target_speed = target_speeds[len(target_speeds) // 2]

        current_observation_tuple = self.observation_type.observe()
        dummy_action = self.action_space.sample() # type: ignore
        info = self._info(current_observation_tuple, dummy_action)
        return current_observation_tuple, info

    def _info(self, obs: ObsType, action: np.ndarray | list) -> dict: # Changed Action to np.ndarray | list
        kinematics_data_dict = obs[1]
        info_dict = {
            "last_goal_attempt_success": self.last_step_goal_met,
            "current_target_goal_index": self.current_goal_index,
            "total_goals_in_sequence": len(self.goal_sequence),
            "all_goals_completed": self.current_goal_index >= len(self.goal_sequence)
        }
        if self.vehicle:
            info_dict.update({
                "speed": self.vehicle.speed, "crashed": self.vehicle.crashed,
                "on_road": self.vehicle.on_road,
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
            })
        return info_dict

    def step(self, action: np.ndarray) -> tuple[ObsType, float, bool, bool, dict]: # Changed Action to np.ndarray
        self.time += 1
        self._simulate(action)
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        obs_for_next_step = self.observation_type.observe()
        info = self._info(obs_for_next_step, action)
        if self.render_mode == "human":
            self.render()

        return obs_for_next_step, reward, terminated, truncated, info

    def _create_road(self) -> None:
        net = RoadNetwork()
        width = self.config["lane_width"]
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        line_type_defs = [[LineType.CONTINUOUS, LineType.CONTINUOUS], [LineType.STRIPED, LineType.CONTINUOUS]]
        x_offset = self.config["x_offset"]; y_offset = self.config["y_offset"]
        size = self.config["road_segment_size"]; gap = self.config["road_segment_gap"]
        extra_len_val = self.config["road_extra_length"][0] if self.config["road_extra_length"] else 10
        self._lane_ids = []
        net.add_lane("a","b", StraightLane(
                [x_offset + width, y_offset + width * 2], [x_offset + width, y_offset + width * 2 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("a", "b", 0))
        net.add_lane("b","c",StraightLane(
                [x_offset + width * 2, y_offset + width * 3 + size], [x_offset + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 0))
        net.add_lane("b","c",StraightLane(
                [x_offset + width * 2 + gap, y_offset + width * 2 + size + width * 2 - (width * 2 + extra_len_val) / 2],
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 2 + size + width * 2 - (width * 2 + extra_len_val) / 2],
                width=width * 2 + extra_len_val, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 1))
        net.add_lane("b","c",StraightLane(
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 2))
        net.add_lane("b","c",StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val, y_offset + width * 2 + size],
                width=width * 4, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 3))
        net.add_lane("b","c",StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val, y_offset + width * 3 + size],
                [x_offset + width * 2 + size, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 4))
        net.add_lane("c","d",StraightLane(
                [x_offset + width * 3 + size, y_offset + width * 2 + size], [x_offset + width * 3 + size, y_offset + width * 2],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("c", "d", 0))
        net.add_lane("d","e",StraightLane(
                [x_offset + width * 2 + size, y_offset + width], [x_offset + width * 2, y_offset + width],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("d", "e", 0))
        net.add_lane("e","f",StraightLane(
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size - extra_len_val],
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("e", "f", 0))
        net.add_lane("f","g",StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val / 2, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val / 2, y_offset + width * 2 + size - width * 2],
                width=extra_len_val, line_types=line_type_defs[0]))
        self._lane_ids.append(("f", "g", 0))
        net.add_lane("g","h",StraightLane(
                [x_offset, y_offset + width * 2 + size / 2], [x_offset + width * 2, y_offset + width * 2 + size / 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 0))
        net.add_lane("g","h",StraightLane(
                [x_offset + width * 2 + size, y_offset + width * 2 + size / 2], [x_offset + width * 4 + size, y_offset + width * 2 + size / 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 1))
        net.add_lane("g","h",StraightLane(
                [x_offset + width * 2 + size / 2, y_offset], [x_offset + width * 2 + size / 2, y_offset + width * 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 2))
        net.add_lane("g","h",StraightLane(
                [x_offset + width * 2 + size / 2, y_offset + width * 2 + size], [x_offset + width * 2 + size / 2, y_offset + width * 4 + size],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 3))
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config.get("show_trajectories", False))
        if not hasattr(self.road, 'objects') or self.road.objects is None: self.road.objects = []

    def _make_walls(self) -> None:
        size = self.config["road_segment_size"]
        gap = self.config["road_segment_gap"]
        road_width = 2 * self.config["lane_width"]
        length = self.config["road_extra_length"][0]

        def create_wall(start, end):
            point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
            if start[1] == end[1]:
                obstacle = Obstacle(self.road, point)
            else:
                obstacle = Obstacle(self.road, point, heading=np.pi / 2)
            obstacle.LENGTH = abs(end[0] - start[0]) + abs(end[1] - start[1])
            obstacle.WIDTH = 1
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        create_wall([0, 0], [0, size + road_width * 2])
        create_wall([0, size + road_width * 2], [size + road_width * 2, size + road_width * 2])
        create_wall([size + road_width * 2, size + road_width * 2], [size + road_width * 2, 0])
        create_wall([size + road_width * 2, 0], [0, 0])

        create_wall([road_width, road_width], [road_width, size + road_width])
        create_wall([size + road_width, size + road_width], [size + road_width, road_width])
        create_wall([size + road_width, road_width], [road_width, road_width])

        create_wall([road_width, size + road_width], [road_width + gap, size + road_width])
        create_wall([road_width + gap, size + road_width], [road_width + gap, size + road_width - length])
        create_wall([road_width + gap, size + road_width - length], [road_width * 2 + gap, size + road_width - length])
        create_wall([road_width * 2 + gap, size + road_width - length], [road_width * 2 + gap, size + road_width])
        create_wall([road_width * 2 + gap, size + road_width], [road_width * 2 + gap * 2, size + road_width])
        create_wall([road_width * 2 + gap * 2, size + road_width], [road_width * 2 + gap * 2, size])
        create_wall([road_width * 2 + gap * 2, size], [road_width * 2 + gap * 2 + length, size])
        create_wall([road_width * 2 + gap * 2 + length, size], [road_width * 2 + gap * 2 + length, road_width + size])
        create_wall([road_width * 2 + gap * 2 + length, road_width + size], [size + road_width, size + road_width])

    def _make_vehicles(self) -> None:
        """Creates and places vehicles on the road."""
        rng = self.np_random; self.controlled_vehicles = []; self.road.vehicles = [] 
        if not self._lane_ids: self.vehicle = None; return
        chosen_lane_idx = self.config.get("start_lane_index", 0) 
        if not (0 <= chosen_lane_idx < len(self._lane_ids)): chosen_lane_idx = 0 
        fixed_lane_id_tuple = self._lane_ids[chosen_lane_idx]
        try: lane_object = self.road.network.get_lane(fixed_lane_id_tuple)
        except KeyError: self.vehicle = None; return
        current_lane_length = lane_object.length
        initial_longitudinal = current_lane_length * 0.05 if current_lane_length > 0 else 0.0
        initial_speed = 0.0 
        controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
            self.road, lane_index=fixed_lane_id_tuple, speed=initial_speed, longitudinal=initial_longitudinal)
        self.controlled_vehicles.append(controlled_vehicle); self.road.vehicles.append(controlled_vehicle)
        self.vehicle = controlled_vehicle; self.vehicle.goal = None



register(id="DrivingClass-v0", entry_point=f"{__name__}:DrivingClassEnv")

if __name__ == "__main__":
    env = gym.make("DrivingClass-v0", render_mode="human")
    print(f"--- {env.spec.id if env.spec else 'DrivingClass-v0'} Test ---")
    print(f"Observation space: {env.observation_space}")
    if isinstance(env.observation_space, spaces.Tuple):
        print(f"  Lidar space (idx 0) shape: {env.observation_space.spaces[0].shape}")
        if isinstance(env.observation_space.spaces[1], spaces.Dict):
            kgo_space = env.observation_space.spaces[1]
            print(f"  KinematicsGoal space (idx 1):")
            for key, space_item in kgo_space.spaces.items():
                 print(f"    {key}: {space_item.shape}")
    print(f"Action space: {env.action_space}")
    num_episodes = 1000
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs_tuple, info = env.reset()
        print(f"Initial Info: {info}")
        kinematics_data_dict_reset = obs_tuple[1]
        if kinematics_data_dict_reset.get("desired_goal") is not None:
            goal_str = [f"{c:.2f}" for c in kinematics_data_dict_reset['desired_goal']]
            print(f"Initial Desired Goal (target_idx {info.get('current_target_goal_index',0)}): {goal_str}")
        terminated = False; truncated = False
        total_reward_ep = 0.0; step_count = 0
        while not (terminated or truncated):
            action = env.action_space.sample() # type: ignore
            obs_tuple, reward_val, terminated, truncated, current_info = env.step(action)
            total_reward_ep += reward_val; step_count += 1
            lidar_data = obs_tuple[0]
            kinematics_data_dict_step = obs_tuple[1]
            act_str = [f"{a:.2f}" for a in action] if isinstance(action, np.ndarray) else str(action)
            if current_info.get("last_goal_attempt_success", False):
                achieved_goal_idx = current_info.get('current_target_goal_index', 1) -1
                print(f"  >>> Intermediate Goal (idx {achieved_goal_idx}) Reached at step {step_count}! <<<")
                if kinematics_data_dict_step.get("desired_goal") is not None:
                     next_goal_str = [f"{c:.2f}" for c in kinematics_data_dict_step['desired_goal']]
                     print(f"      New Desired Goal (target_idx {current_info.get('current_target_goal_index')}): {next_goal_str}")
                elif current_info.get('all_goals_completed', False):
                     print(f"      All goals in sequence completed!")
            if (step_count) % 30 == 0 or terminated or truncated:
                print(f"  St {step_count:3d}, Act: {act_str}, Rew: {reward_val:7.3f}, "
                      f"Lidar Sum: {np.sum(lidar_data):6.2f}, LastGoalOK: {current_info.get('last_goal_attempt_success', False)}, "
                      f"TargetIdx: {current_info.get('current_target_goal_index', 0)}")
            if terminated:
                all_done_flag = current_info.get('all_goals_completed', False)
                crashed_flag = not hasattr(env, "vehicle")
                print(f"Terminated. Overall Success (all goals met & not crashed): {all_done_flag and not crashed_flag}. Steps {step_count}.")
            if truncated: print(f"Truncated. Steps {step_count}.")
        print(f"Ep {episode + 1} end. TotRew: {total_reward_ep:.2f}, Steps: {step_count}")
    env.close()
    print("\n--- Test Finished ---")
