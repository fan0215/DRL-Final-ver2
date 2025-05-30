from __future__ import annotations

from abc import abstractmethod # For GoalEnv interface

import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from gymnasium.core import ObsType 
from gymnasium import Env # For GoalEnv interface

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv 
from highway_env.envs.common.observation import observation_factory, KinematicsGoalObservation, LidarObservation # Ensure LidarObservation is conceptually available
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
    Goal-conditioned Driving Class scenario with Lidar observation.
    Agent uses Lidar for local perception and kinematic goals for task completion.
    """
    # INTERNAL_KINEMATICS_CONFIG is used if main obs type is not KinematicsGoal,
    # to ensure we can always get kinematic achieved/desired goals for reward.
    # However, with KinematicsGoal as the main wrapper, this might be redundant
    # if KinematicsGoal itself can provide these for internal calculations.
    INTERNAL_KINEMATICS_CONFIG = {
        "type": "KinematicsGoal", # This is a bit meta, usually it would be just "Kinematics"
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "goal_features": ["x", "y", "cos_h", "sin_h"], # Define how internal goals are structured
        "scales": [100, 100, 10, 10, 1, 1], 
        "goal_scales": [100, 100, 1, 1], 
        "normalize": False,
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        self.goal_landmark: Landmark | None = None
        super().__init__(config, render_mode) 
        
        # self.observation_type is now KinematicsGoalObservation.
        # Its sub_observation (for the 'observation' field) will be LidarObservation.
        # For internal reward/success calculations, we need access to kinematic achieved/desired goals.
        # KinematicsGoalObservation itself provides these as obs_dict['achieved_goal'] and obs_dict['desired_goal'].
        # So, we can directly use the output of self.observation_type.observe() for these.
        # The self.kinematics_goal_observer can be simplified or confirmed to be self.observation_type.
        if not isinstance(self.observation_type, KinematicsGoalObservation):
            # This case should ideally not happen if config["observation"]["type"] is "KinematicsGoal"
            # print("Warning: Main observation type is not KinematicsGoal. Setting up separate kinematics_goal_observer.")
            # This separate observer is for internal calculations if the main obs isn't already structured.
            self.kinematics_goal_observer = observation_factory(self, self.INTERNAL_KINEMATICS_CONFIG)
        else:
            self.kinematics_goal_observer = self.observation_type


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",  # Top-level wrapper for GoalEnv structure
                    
                    # Config for the 'achieved_goal' and 'desired_goal' parts
                    "goal_features": ["x", "y", "cos_h", "sin_h"], 
                    "goal_scales": [100, 100, 1, 1], # Scales for goal features
                    "normalize_goal": False, # Whether to normalize achieved/desired goals

                    # Config for the 'observation' part of the KinematicsGoal dict: Lidar
                    "observation_config": {
                        "type": "OccupancyGrid",
                        "features": [ "on_road" ],
                        "grid_size": [[-50, 50]],
                        "grid_step": [3, 3],
                        "as_image": False,
                        "align_to_vehicle_axes": True,
                    },
                    
                    # These are for the KinematicsGoal wrapper itself, 
                    # defining its default base observation IF observation_config was not given.
                    # Since observation_config IS given, these 'features' and 'scales'
                    # for the base 'observation' part are effectively overridden by Lidar.
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"], # Fallback/default for 'observation' field
                    "scales": [100, 100, 10, 10, 1, 1],
                    "normalize": False # Normalization for the wrapper itself if it were to use its own kinematics for 'observation'
                },
                "action": {
                    "type": "ContinuousAction", "longitudinal": True, "lateral": True,
                    "acceleration_range": [-2, 3.5], "steering_range": [-np.pi / 3, np.pi / 3], 
                },
                "simulation_frequency": 15, "policy_frequency": 5, "duration": 300,
                "reward_weights": np.array([1.0, 1.0, 0.3, 0.3]), # For [x, y, cos_h, sin_h] goal features
                "collision_reward": -150.0, 
                "success_goal_reward": 0.15, # Used in _is_success: compute_reward_val > -this_value
                "action_penalty_weight": -0.02, 
                "on_road_shaping_reward": 0.005, 
                "lane_centering_shaping_reward": 0.005,
                "lane_centering_cost_factor": 4,
                "controlled_vehicles": 1, 
                "other_vehicles": 0, 
                "goal_lane_index_tuple": ("e", "f", 0), 
                "goal_longitudinal_offset": 0.5, 
                "goal_heading_noise_std": np.deg2rad(3), 
                "goal_position_noise_std": 0.1, 
                "success_distance_threshold": 0.75, # Adjusted physical threshold for success
                "success_heading_threshold_rad": np.deg2rad(15), # Adjusted
                "screen_width": 1200, "screen_height": 900,
                "centering_position": [0.3, 0.6], "scaling": 3.5, 
                "lane_width": 4.0, 
                "show_trajectories": False, "offroad_terminal": True,
                "x_offset": 10, "y_offset": 10,
                "road_segment_size": 80, "road_segment_gap": 8,
                "road_extra_length": [10], "start_lane_index": 1,
                "add_lane_edge_obstacles": False,
                "lane_edge_obstacle_width": 0.1,
            }
        )
        return config

    def _set_destination(self) -> None:
        # (Implementation from previous version - unchanged)
        if not self._lane_ids or self.vehicle is None:
            if self.goal_landmark and self.goal_landmark in self.road.objects: self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = None
            if self.vehicle: self.vehicle.goal = None; return

        goal_lane_id_tuple = self.config["goal_lane_index_tuple"]
        if goal_lane_id_tuple not in self._lane_ids:
            goal_lane_id_tuple = self._lane_ids[-1] if self._lane_ids else None
            if not goal_lane_id_tuple:
                 if self.goal_landmark and self.goal_landmark in self.road.objects: self.road.objects.remove(self.goal_landmark)
                 self.goal_landmark = None; self.vehicle.goal = None; return
        try:
            goal_lane_object = self.road.network.get_lane(goal_lane_id_tuple)
            long_offset_factor = np.clip(self.config["goal_longitudinal_offset"], 0.05, 0.95)
            goal_pos_on_lane = goal_lane_object.position(goal_lane_object.length * long_offset_factor, 0)
            goal_heading_on_lane = goal_lane_object.heading_at(goal_lane_object.length * long_offset_factor)
            goal_pos_on_lane[0] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_pos_on_lane[1] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_heading_on_lane += self.np_random.normal(0, self.config["goal_heading_noise_std"])
            
            if self.goal_landmark and self.goal_landmark in self.road.objects: self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = Landmark(self.road, goal_pos_on_lane, heading=goal_heading_on_lane) 
            self.goal_landmark.color = (0, 200, 0); self.goal_landmark.length = self.config["lane_width"] * 0.7
            self.goal_landmark.width = self.config["lane_width"] * 0.7
            self.road.objects.append(self.goal_landmark)
            self.vehicle.goal = self.goal_landmark 
        except KeyError:
            if self.goal_landmark and self.goal_landmark in self.road.objects: self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = None; self.vehicle.goal = None

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        # (Implementation from previous version - uses p from signature, reward_weights from config)
        current_reward_weights = np.array(self.config["reward_weights"])
        if len(current_reward_weights) != len(achieved_goal): # achieved_goal is [x,y,cos,sin]
            # This check might be problematic if achieved_goal has different length than goal_features due to config
            # However, KinematicsGoalObservation should ensure they match goal_features length.
            # print("Warning: Mismatch in reward_weights length. Using unweighted norm.")
            return -np.power(np.linalg.norm(achieved_goal - desired_goal, ord=p, axis=-1), p)

        
        return -np.power(
            np.dot(np.abs(achieved_goal - desired_goal), current_reward_weights), p)

    def _reward(self, action: np.ndarray) -> float:
        # (Implementation from previous version - calls compute_reward, adds collision and shaping)
        # self.observation_type is KinematicsGoalObservation, so it returns the dict
        obs_dict = self.observation_type.observe()
        achieved_goal = obs_dict["achieved_goal"]
        desired_goal = obs_dict["desired_goal"]
        is_success_flag = self._is_success(achieved_goal, desired_goal)
        info_for_compute_reward = {
            "is_crashed": self.vehicle.crashed if self.vehicle else True,
            "is_success": is_success_flag, 
        }
        goal_based_reward = self.compute_reward(achieved_goal, desired_goal, info_for_compute_reward) # p defaults to 0.5
        total_reward = goal_based_reward
        if self.vehicle and self.vehicle.crashed:
            total_reward += self.config["collision_reward"]
        action_penalty = self.config["action_penalty_weight"] * np.linalg.norm(action)
        total_reward += action_penalty
        if self.vehicle:
            on_road_value = float(self.vehicle.on_road)
            total_reward += self.config["on_road_shaping_reward"] * on_road_value
            lane_centering_value = 0.0
            if self.vehicle.on_road and self.vehicle.lane_index:
                try:
                    current_lane = self.road.network.get_lane(self.vehicle.lane_index)
                    _, lateral_offset = current_lane.local_coordinates(self.vehicle.position)
                    lane_centering_value = 1 / (1 + self.config["lane_centering_cost_factor"] * lateral_offset**2)
                except KeyError: pass 
            total_reward += self.config["lane_centering_shaping_reward"] * lane_centering_value
        return float(total_reward)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        # Using physical thresholds for clarity, can also use compute_reward as before
        dist_sq = np.sum(np.square(achieved_goal[0:2] - desired_goal[0:2]))
        position_ok = dist_sq < self.config["success_distance_threshold"]**2
        
        # Ensure goals have heading components (indices 2 and 3 for cos_h, sin_h)
        if len(achieved_goal) >= 4 and len(desired_goal) >= 4:
            cos_delta_angle = np.clip(np.dot(achieved_goal[2:4], desired_goal[2:4]), -1.0, 1.0)
            heading_ok = cos_delta_angle >= np.cos(self.config["success_heading_threshold_rad"])
        else: # If goal_features doesn't include heading, consider heading always OK for success
            heading_ok = True 
            # print("Warning: Heading components not found in achieved/desired goal for success check.")

        return bool(position_ok and heading_ok)

    def _is_terminated(self) -> bool:
        crashed = self.vehicle.crashed if self.vehicle else True
        off_road = self.config["offroad_terminal"] and (self.vehicle and not self.vehicle.on_road)
        obs_dict = self.observation_type.observe() 
        success = self._is_success(obs_dict["achieved_goal"], obs_dict["desired_goal"])
        return bool(crashed or success or off_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> tuple[ObsType, dict]:
        self.time = 0; self.terminated = False; self.truncated = False
        self._create_road(); self._make_vehicles() 
        self._set_destination() 
        action_config = self.config["action"]
        if self.vehicle and action_config.get("longitudinal", True) is False and hasattr(self.vehicle, 'target_speed'):
            target_speeds = action_config.get("target_speeds") 
            if target_speeds and len(target_speeds) > 0:
                self.vehicle.target_speed = target_speeds[len(target_speeds) // 2] 
        obs_dict = self.observation_type.observe() 
        dummy_action = self.action_space.sample() 
        info = self._info(obs_dict, dummy_action) 
        return obs_dict, info

    def _info(self, obs: dict, action: np.ndarray | list) -> dict:
        is_success_flag = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info_dict = {"is_success": is_success_flag}
        if self.vehicle:
            info_dict.update({
                "speed": self.vehicle.speed, "crashed": self.vehicle.crashed,
                "on_road": self.vehicle.on_road,
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
            })
        # Add Lidar to info for debugging if needed, but it's already in obs['observation']
        # if "observation" in obs and isinstance(obs["observation"], np.ndarray):
        #    info_dict["lidar_sample"] = obs["observation"][:5].tolist() # Sample of lidar
        return info_dict

    def _create_road(self) -> None:
        # (Implementation from user's previous version - unchanged)
        net = RoadNetwork()
        width = self.config["lane_width"] 
        actual_lane_width_for_segment = width * 2
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        line_type = [[c, c], [n, c]]
        x_offset = self.config["x_offset"]; y_offset = self.config["y_offset"]
        size = self.config["road_segment_size"]; gap = self.config["road_segment_gap"]
        length = self.config["road_extra_length"]

        self._lane_ids = [] 
        # (0, 0) to (0, 1)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [x_offset + width, y_offset + width * 2],
                [x_offset + width, y_offset + width * 2 + size],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("a", "b", 0))
        
        # (0, 1) to (1, 1)
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("b", "c", 0))
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap, y_offset + width * 2 + size + width * 2 - (width * 2 + length[0]) / 2],
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 2 + size + width * 2 - (width * 2 + length[0]) / 2],
                width=width * 2 + length[0],
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("b", "c", 1))
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("b", "c", 2))
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + length[0], y_offset + width * 2 + size],
                width=width * 4,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("b", "c", 3))
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + length[0], y_offset + width * 3 + size],
                [x_offset + width * 2 + size, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("b", "c", 4))

        # (1, 1) to (1, 0)
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [x_offset + width * 3 + size, y_offset + width * 2 + size],
                [x_offset + width * 3 + size, y_offset + width * 2],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("c", "d", 0))

        # (1, 0) to (0, 0)
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [x_offset + width * 2 + size, y_offset + width],
                [x_offset + width * 2, y_offset + width],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("d", "e", 0))

        # 倒車
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size - length[0]],
                width=width * 2,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("e", "f", 0))

        # 路邊停車
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + length[0] / 2, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + length[0] / 2, y_offset + width * 2 + size - width * 2],
                width=length[0],
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("f", "g", 0))

        # corner
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [x_offset, y_offset + width * 2 + size / 2],
                [x_offset + width * 2, y_offset + width * 2 + size / 2],
                width=width * 4 + size,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("g", "h", 0))
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [x_offset + width * 2 + size, y_offset + width * 2 + size / 2],
                [x_offset + width * 4 + size, y_offset + width * 2 + size / 2],
                width=width * 4 + size,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("g", "h", 1))
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [x_offset + width * 2 + size / 2, y_offset],
                [x_offset + width * 2 + size / 2, y_offset + width * 2],
                width=width * 4 + size,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("g", "h", 2))
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [x_offset + width * 2 + size / 2, y_offset + width * 2 + size],
                [x_offset + width * 2 + size / 2, y_offset + width * 4 + size],
                width=width * 4 + size,
                line_types=line_type[0]
            )
        )
        self._lane_ids.append(("g", "h", 3))

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False)
        )
        if not hasattr(self.road, 'objects') or self.road.objects is None: 
            self.road.objects = []
        if self.config.get("add_lane_edge_obstacles", False):
            obstacle_width = self.config["lane_edge_obstacle_width"]
            for lane_tuple in self._lane_ids: 
                try:
                    lane = self.road.network.get_lane(lane_tuple)
                    if isinstance(lane, StraightLane):
                        lane_len = lane.length
                        if lane_len <= 1e-6: continue
                        direction = lane.end - lane.start; norm_val = np.linalg.norm(direction)
                        if norm_val < 1e-6: continue 
                        norm_direction = direction / norm_val
                        lane_normal_vec = np.array([-norm_direction[1], norm_direction[0]])
                        left_edge_center = lane.position(lane_len / 2, -lane.width / 2) 
                        obs_left = Obstacle(self.road, left_edge_center, heading=lane.heading_at(lane_len/2)) 
                        obs_left.LENGTH = lane_len; obs_left.WIDTH = obstacle_width
                        obs_left.HITBOX_LENGTH = lane_len; obs_left.HITBOX_WIDTH = obstacle_width
                        self.road.objects.append(obs_left)
                        right_edge_center = lane.position(lane_len / 2, lane.width / 2) 
                        obs_right = Obstacle(self.road, right_edge_center, heading=lane.heading_at(lane_len/2)) 
                        obs_right.LENGTH = lane_len; obs_right.WIDTH = obstacle_width
                        obs_right.HITBOX_LENGTH = lane_len; obs_right.HITBOX_WIDTH = obstacle_width
                        self.road.objects.append(obs_right)
                except KeyError: pass

    def _make_vehicles(self) -> None:
        # (Implementation from previous version - unchanged)
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

# --- Registration and Test Script ---
register(
    id="DrivingClass-v0", 
    entry_point=f"{__name__}:DrivingClassEnv",
)

if __name__ == "__main__":
    env = gym.make("DrivingClass-v0", render_mode="human")
    
    print(f"--- {env.spec.id if env.spec else 'DrivingClass-v0'} (Lidar + Goal) Test ---")
    print(f"Observation space: {env.observation_space}")
    # Example of accessing sub-spaces:
    # print(f"  Lidar observation part space: {env.observation_space['observation']}")
    # print(f"  Achieved goal space: {env.observation_space['achieved_goal']}")

    num_episodes = 1000
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs_dict, info = env.reset() 
        if obs_dict.get("desired_goal") is not None:
            goal_str = [f"{c:.2f}" for c in obs_dict['desired_goal']]
            print(f"Desired Goal (x,y,cosH,sinH): {goal_str}")
        terminated = False; truncated = False; total_reward_ep = 0.0; step_count = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs_dict, reward_val, terminated, truncated, info = env.step(action)
            total_reward_ep += reward_val; step_count += 1
            
            # The 'observation' field in obs_dict is now Lidar data
            lidar_data = obs_dict['observation'] 
            ach_goal_str = [f"{c:.2f}" for c in obs_dict['achieved_goal']]
            act_str = [f"{a:.2f}" for a in action] if isinstance(action, np.ndarray) else str(action)
            if (step_count) % 50 == 0:
                print(f"  St {step_count}, Act: {act_str}, Rew: {reward_val:.3f}, Lidar Sum: {np.sum(lidar_data):.2f}, Succ: {info.get('is_success', False)}")

            if terminated: print(f"Terminated: Steps {step_count}. Success: {info.get('is_success', False)}")
            if truncated: print(f"Truncated: Steps {step_count}. Success: {info.get('is_success', False)}")
        print(f"Ep {episode + 1} end. TotRew: {total_reward_ep:.2f}, Steps: {step_count}")
    env.close(); print("\n--- Test Finished ---")
