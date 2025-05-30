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
                            "max_distance": 30.0,            # Max Lidar distance (meters)
                            "normalize": True,               # Normalize Lidar distances to [0, 1]
                            "see_behind": True,              # Allow Lidar to see behind
                        },
                        {
                            "type": "KinematicsGoal",
                            "features": ["x", "y", "cos_h", "sin_h", "vx", "vy"], # Fallback/default for 'observation' field
                            "scales": [100, 100, 1, 1, 10, 10],
                            "normalize": True# Normalization for the wrapper itself if it were to use its own kinematics for 'observation'
                        }
                    ],
                },
                "action": {
                    "type": "ContinuousAction", "longitudinal": True, "lateral": True,
                    "acceleration_range": [-5, 5], "steering_range": [-np.pi / 3, np.pi / 3], 
                },
                "simulation_frequency": 15, 
                "policy_frequency": 5, 
                "duration": 300,
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
                "screen_width": 1200, 
                "screen_height": 900,
                "centering_position": [0.3, 0.6], "scaling": 3.5, 
                "lane_width": 4.0, 
                "show_trajectories": False, "offroad_terminal": True,
                "x_offset": 0, "y_offset": 0,
                "road_segment_size": 80, "road_segment_gap": 8,
                "road_extra_length": [10], "start_lane_index": 1,
                "add_lane_edge_obstacles": True,
                "lane_edge_obstacle_width": 0.1,
            }
        )
        return config
    def _set_destination(self) -> None:
        """Sets or updates the goal landmark for the ego vehicle."""
        if not self._lane_ids or self.vehicle is None: # Ensure road and vehicle exist
            if self.goal_landmark and self.goal_landmark in self.road.objects:
                self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = None
            if self.vehicle:
                self.vehicle.goal = None
            return

        goal_lane_id_tuple = self.config["goal_lane_index_tuple"]
        # Fallback if configured goal_lane_id_tuple is not in the current road's _lane_ids
        if goal_lane_id_tuple not in self._lane_ids:
            goal_lane_id_tuple = self._lane_ids[-1] if self._lane_ids else None
            if not goal_lane_id_tuple: # No lanes available at all
                if self.goal_landmark and self.goal_landmark in self.road.objects:
                    self.road.objects.remove(self.goal_landmark)
                self.goal_landmark = None
                if self.vehicle: self.vehicle.goal = None
                return
        try:
            goal_lane_object = self.road.network.get_lane(goal_lane_id_tuple)
            long_offset_factor = np.clip(self.config["goal_longitudinal_offset"], 0.05, 0.95)
            goal_pos_on_lane = goal_lane_object.position(goal_lane_object.length * long_offset_factor, 0) # 0 lateral offset from lane center
            goal_heading_on_lane = goal_lane_object.heading_at(goal_lane_object.length * long_offset_factor)

            # Add noise to goal position and heading
            goal_pos_on_lane[0] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_pos_on_lane[1] += self.np_random.normal(0, self.config["goal_position_noise_std"])
            goal_heading_on_lane += self.np_random.normal(0, self.config["goal_heading_noise_std"])

            # Remove old landmark if it exists
            if self.goal_landmark and self.goal_landmark in self.road.objects:
                self.road.objects.remove(self.goal_landmark)

            # Create and add new landmark
            self.goal_landmark = Landmark(self.road, goal_pos_on_lane, heading=goal_heading_on_lane)
            self.goal_landmark.color = VehicleGraphics.GREEN # Use a standard color
            self.goal_landmark.length = self.config["lane_width"] * 0.7 # Visual size
            self.goal_landmark.width = self.config["lane_width"] * 0.7  # Visual size
            self.road.objects.append(self.goal_landmark)
            if self.vehicle: # Assign to vehicle
                self.vehicle.goal = self.goal_landmark
        except KeyError: # If get_lane fails
            if self.goal_landmark and self.goal_landmark in self.road.objects:
                self.road.objects.remove(self.goal_landmark)
            self.goal_landmark = None
            if self.vehicle: self.vehicle.goal = None


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Computes the reward based on the distance to the goal.
        The distance is a weighted L_p norm, p=0.5 means sum of sqrt of weighted errors.
        """
        current_reward_weights = np.array(self.config["reward_weights"])
        # Ensure achieved_goal and desired_goal match the length of reward_weights
        # (which should correspond to 'goal_features')
        if len(current_reward_weights) != len(achieved_goal) or len(current_reward_weights) != len(desired_goal):
            # Fallback to unweighted norm if there's a mismatch. This indicates a config issue.
            # print(f"Warning: Mismatch in reward_weights length. Achieved: {len(achieved_goal)}, Desired: {len(desired_goal)}, Weights: {len(current_reward_weights)}")
            return -np.power(np.linalg.norm(achieved_goal - desired_goal, ord=p), p) # Note: np.linalg.norm doesn't take axis for 1D arrays

        # Element-wise difference, weighted, then L_p norm
        # For p=0.5, this is -sum(sqrt(weights * |achieved - desired|))
        # The formula used in the original code was: -np.power(np.dot(np.abs(achieved_goal - desired_goal), current_reward_weights), p)
        # This is equivalent to -( (w1*|a1-d1|) + (w2*|a2-d2|) + ... )^p
        # This is a valid way to compute a weighted distance.
        weighted_abs_diff = np.dot(np.abs(achieved_goal - desired_goal), current_reward_weights)
        return -np.power(weighted_abs_diff, p)


    def _reward(self, action: np.ndarray) -> float:
        """Computes the total reward for the current step."""
        # The observation_type is TupleObservation, its observe() method returns a tuple.
        # Element 0: LidarScan, Element 1: KinematicsGoalDict
        full_obs_tuple = self.observation_type.observe()
        kinematics_data_dict = full_obs_tuple[1] # This is the dict from KinematicsGoalObservation

        achieved_goal = kinematics_data_dict["achieved_goal"]
        desired_goal = kinematics_data_dict["desired_goal"]

        is_success_flag = self._is_success(achieved_goal, desired_goal)
        info_for_compute_reward = { # Info dict for the compute_reward method
            "is_crashed": self.vehicle.crashed if self.vehicle else True,
            "is_success": is_success_flag,
        }
        goal_based_reward = self.compute_reward(achieved_goal, desired_goal, info_for_compute_reward) # p defaults to 0.5
        total_reward = goal_based_reward

        # Collision penalty
        if self.vehicle and self.vehicle.crashed:
            total_reward += self.config["collision_reward"]

        # Action penalty
        action_penalty = self.config["action_penalty_weight"] * np.linalg.norm(action)
        total_reward += action_penalty

        # Shaping rewards if vehicle exists
        if self.vehicle:
            # On-road shaping reward
            on_road_value = float(self.vehicle.on_road)
            total_reward += self.config["on_road_shaping_reward"] * on_road_value

            # Lane centering shaping reward
            lane_centering_value = 0.0
            if self.vehicle.on_road and self.vehicle.lane_index: # Ensure vehicle is on a valid lane
                try:
                    current_lane = self.road.network.get_lane(self.vehicle.lane_index)
                    _, lateral_offset = current_lane.local_coordinates(self.vehicle.position)
                    # Reward for being centered, decreases with offset
                    lane_centering_value = 1 / (1 + self.config["lane_centering_cost_factor"] * lateral_offset**2)
                except KeyError: # Should not happen if lane_index is valid
                    pass
            total_reward += self.config["lane_centering_shaping_reward"] * lane_centering_value
        return float(total_reward)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Checks if the agent has successfully reached the goal based on physical thresholds."""
        # Position check (first two elements of goals are x, y)
        dist_sq = np.sum(np.square(achieved_goal[0:2] - desired_goal[0:2]))
        position_ok = dist_sq < self.config["success_distance_threshold"]**2

        # Heading check (elements 2 and 3 are cos_h, sin_h)
        # Ensure goals have heading components
        if len(achieved_goal) >= 4 and len(desired_goal) >= 4:
            # Dot product of heading vectors (achieved_goal[2:4] and desired_goal[2:4]) gives cos(delta_angle)
            cos_delta_angle = np.clip(np.dot(achieved_goal[2:4], desired_goal[2:4]), -1.0, 1.0)
            # Success if delta_angle is within threshold
            heading_ok = cos_delta_angle >= np.cos(self.config["success_heading_threshold_rad"])
        else: # If goal_features doesn't include heading, consider heading always OK for success
            heading_ok = True
            # print("Warning: Heading components not found in achieved/desired goal for success check.")

        return bool(position_ok and heading_ok)

    def _is_terminated(self) -> bool:
        """Checks if the episode should terminate."""
        crashed = self.vehicle.crashed if self.vehicle else True # Terminate on crash
        off_road = self.config["offroad_terminal"] and (self.vehicle and not self.vehicle.on_road) # Terminate if off-road

        # Terminate on success
        full_obs_tuple = self.observation_type.observe()
        kinematics_data_dict = full_obs_tuple[1]
        achieved_goal = kinematics_data_dict["achieved_goal"]
        desired_goal = kinematics_data_dict["desired_goal"]
        success = self._is_success(achieved_goal, desired_goal)

        return bool(crashed or success or off_road)

    def _is_truncated(self) -> bool:
        """Checks if the episode should be truncated (e.g., time limit)."""
        return self.time >= self.config["duration"]

    def _reset(self) -> tuple[ObsType, dict]:
        """Resets the environment to an initial state."""
        self.time = 0
        self.terminated = False
        self.truncated = False
        self._create_road() # Create road layout
        self._make_walls()
        self._make_vehicles() # Create and place vehicles
        self._set_destination() # Set the goal for the ego vehicle

        # Set initial target speed if longitudinal control is off (as per original logic)
        action_config = self.config["action"]
        if self.vehicle and action_config.get("longitudinal", True) is False and hasattr(self.vehicle, 'target_speed'):
            target_speeds = action_config.get("target_speeds")
            if target_speeds and len(target_speeds) > 0:
                self.vehicle.target_speed = target_speeds[len(target_speeds) // 2]

        # Get the initial observation (tuple of LidarScan, KinematicsGoalDict)
        current_observation_tuple = self.observation_type.observe()
        # Generate dummy action for initial info dict (some info fields might depend on action)
        dummy_action = self.action_space.sample()
        info = self._info(current_observation_tuple, dummy_action)

        return current_observation_tuple, info # Return the tuple observation and info dict

    def _info(self, obs_tuple: tuple, action: np.ndarray | list) -> dict:
        """Generates an info dictionary for the current step."""
        # obs_tuple is (LidarScan, KinematicsGoalDict)
        kinematics_data_dict = obs_tuple[1]
        achieved_goal = kinematics_data_dict["achieved_goal"]
        desired_goal = kinematics_data_dict["desired_goal"]
        is_success_flag = self._is_success(achieved_goal, desired_goal)

        info_dict = {"is_success": is_success_flag}
        if self.vehicle:
            info_dict.update({
                "speed": self.vehicle.speed,
                "crashed": self.vehicle.crashed,
                "on_road": self.vehicle.on_road,
                "action": action.tolist() if isinstance(action, np.ndarray) else action, # Store action taken
                # "raw_lidar_sum": np.sum(obs_tuple[0]), # Example: add some Lidar summary
            })
        return info_dict

    def _create_road(self) -> None:
        """Creates the road network for the environment."""
        net = RoadNetwork()
        width = self.config["lane_width"]
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        # Define line types for lanes: e.g., [LeftLine, RightLine]
        # line_type_defs[0] could be solid lines, line_type_defs[1] could be dashed/solid etc.
        line_type_defs = [
            [LineType.CONTINUOUS, LineType.CONTINUOUS], # Both solid
            [LineType.STRIPED, LineType.CONTINUOUS]     # Left striped, Right solid
        ]
        x_offset = self.config["x_offset"]
        y_offset = self.config["y_offset"]
        size = self.config["road_segment_size"]
        gap = self.config["road_segment_gap"]
        # Ensure 'length' is treated as a list, typically with one element for specific segments.
        extra_len_val = self.config["road_extra_length"][0] if self.config["road_extra_length"] else 10

        self._lane_ids = [] # Reset lane IDs for the new road

        # Segment 1: (a -> b) Vertical
        net.add_lane("a","b", StraightLane(
                [x_offset + width, y_offset + width * 2], # Start point
                [x_offset + width, y_offset + width * 2 + size], # End point
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("a", "b", 0))

        # Segment 2: (b -> c) Horizontal with various parts
        net.add_lane("b","c",StraightLane( # Part 1
                [x_offset + width * 2, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 0))
        net.add_lane("b","c",StraightLane( # Part 2 (wider)
                [x_offset + width * 2 + gap, y_offset + width * 2 + size + width * 2 - (width * 2 + extra_len_val) / 2],
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 2 + size + width * 2 - (width * 2 + extra_len_val) / 2],
                width=width * 2 + extra_len_val, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 1))
        net.add_lane("b","c",StraightLane( # Part 3
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 2))
        net.add_lane("b","c",StraightLane( # Part 4 (wider)
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 2 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val, y_offset + width * 2 + size],
                width=width * 4, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 3))
        net.add_lane("b","c",StraightLane( # Part 5
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val, y_offset + width * 3 + size],
                [x_offset + width * 2 + size, y_offset + width * 3 + size],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("b", "c", 4))

        # Segment 3: (c -> d) Vertical
        net.add_lane("c","d",StraightLane(
                [x_offset + width * 3 + size, y_offset + width * 2 + size],
                [x_offset + width * 3 + size, y_offset + width * 2],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("c", "d", 0))

        # Segment 4: (d -> e) Horizontal
        net.add_lane("d","e",StraightLane(
                [x_offset + width * 2 + size, y_offset + width],
                [x_offset + width * 2, y_offset + width],
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("d", "e", 0))

        # Segment 5: (e -> f) "Reversing Bay" - 倒車 (This is the default goal lane)
        net.add_lane("e","f",StraightLane(
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size], # Start
                [x_offset + width * 2 + gap + width, y_offset + width * 2 + size - extra_len_val], # End (shorter)
                width=width * 2, line_types=line_type_defs[0]))
        self._lane_ids.append(("e", "f", 0))

        # Segment 6: (f -> g) "Roadside Parking" - 路邊停車
        net.add_lane("f","g",StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val / 2, y_offset + width * 2 + size], # Top center
                [x_offset + width * 2 + gap + width * 2 + gap + extra_len_val / 2, y_offset + width * 2 + size - width * 2], # Bottom center
                width=extra_len_val, line_types=line_type_defs[0])) # Width of spot is extra_len_val
        self._lane_ids.append(("f", "g", 0))

        # Segment 7: (g -> h) "Corner" sections (very wide lanes, might be open areas)
        net.add_lane("g","h",StraightLane( # Horizontal part 1
                [x_offset, y_offset + width * 2 + size / 2],
                [x_offset + width * 2, y_offset + width * 2 + size / 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 0))
        net.add_lane("g","h",StraightLane( # Horizontal part 2
                [x_offset + width * 2 + size, y_offset + width * 2 + size / 2],
                [x_offset + width * 4 + size, y_offset + width * 2 + size / 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 1))
        net.add_lane("g","h",StraightLane( # Vertical part 1
                [x_offset + width * 2 + size / 2, y_offset],
                [x_offset + width * 2 + size / 2, y_offset + width * 2],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 2))
        net.add_lane("g","h",StraightLane( # Vertical part 2
                [x_offset + width * 2 + size / 2, y_offset + width * 2 + size],
                [x_offset + width * 2 + size / 2, y_offset + width * 4 + size],
                width=width * 4 + size, line_types=line_type_defs[0]))
        self._lane_ids.append(("g", "h", 3))


        self.road = Road(network=net, np_random=self.np_random, record_history=self.config.get("show_trajectories", False))
        # # Ensure road.objects list exists for adding landmarks or obstacles
        # if not hasattr(self.road, 'objects') or self.road.objects is None: 
        #     self.road.objects = []
        # if self.config.get("add_lane_edge_obstacles", False):
        #     obstacle_width = self.config["lane_edge_obstacle_width"]
        #     for lane_id in self._lane_ids: 
        #         try:
        #             lane = self.road.network.get_lane(lane_id)
        #             if isinstance(lane, StraightLane):
        #                 lane_len = lane.length
        #                 if lane_len <= 1e-6: continue
        #                 direction = lane.end - lane.start; norm_val = np.linalg.norm(direction)
        #                 if norm_val < 1e-6: continue 
        #                 left_edge_center = lane.position(lane_len / 2, -lane.width / 2) 
        #                 obs_left = Obstacle(self.road, left_edge_center, heading=lane.heading_at(lane_len/2)) 
        #                 obs_left.LENGTH = lane_len; obs_left.WIDTH = obstacle_width
        #                 obs_left.HITBOX_LENGTH = lane_len; obs_left.HITBOX_WIDTH = obstacle_width
        #                 self.road.objects.append(obs_left)
        #                 right_edge_center = lane.position(lane_len / 2, lane.width / 2) 
        #                 obs_right = Obstacle(self.road, right_edge_center, heading=lane.heading_at(lane_len/2)) 
        #                 obs_right.LENGTH = lane_len; obs_right.WIDTH = obstacle_width
        #                 obs_right.HITBOX_LENGTH = lane_len; obs_right.HITBOX_WIDTH = obstacle_width
        #                 self.road.objects.append(obs_right)
        #         except KeyError: pass

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


# --- Registration and Test Script ---
register(
    id="DrivingClass-v0", # Standard name for this version
    entry_point=f"{__name__}:DrivingClassEnv", # Entry point: this_file_name:ClassName
)

if __name__ == "__main__":
    # Test the environment
    env = gym.make("DrivingClass-v0", render_mode="human")

    print(f"--- {env.spec.id if env.spec else 'DrivingClass-v0'} (Tuple Lidar + Goal) Test ---")
    print(f"Observation space: {env.observation_space}")
    # Detailed breakdown of TupleObservation space
    if isinstance(env.observation_space, spaces.Tuple):
        print(f"  Lidar observation part space (index 0): {env.observation_space.spaces[0]}")
        print(f"  KinematicsGoal observation part space (index 1): {env.observation_space.spaces[1]}")
        if isinstance(env.observation_space.spaces[1], spaces.Dict):
            kgo_space = env.observation_space.spaces[1]
            print(f"    Achieved goal space: {kgo_space['achieved_goal']}")
            print(f"    Desired goal space: {kgo_space['desired_goal']}")
            print(f"    Ego kinematics ('observation' key) space: {kgo_space['observation']}")

    print(f"Action space: {env.action_space}")

    num_episodes = 1000 # Number of episodes to run for testing
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        # obs_tuple is (LidarScan_ndarray, KinematicsGoal_dict)
        obs_tuple, info = env.reset()

        # Access desired_goal from the KinematicsGoal_dict part of the tuple
        kinematics_data_dict_reset = obs_tuple[1]
        if kinematics_data_dict_reset.get("desired_goal") is not None:
            goal_str = [f"{c:.2f}" for c in kinematics_data_dict_reset['desired_goal']]
            print(f"Desired Goal (x,y,cosH,sinH): {goal_str}")

        terminated = False
        truncated = False
        total_reward_ep = 0.0
        step_count = 0
        while not (terminated or truncated):
            action = env.action_space.sample() # Sample a random action
            obs_tuple, reward_val, terminated, truncated, info = env.step(action)
            total_reward_ep += reward_val
            step_count += 1

            # Access Lidar data and kinematics data from the observation tuple
            lidar_data = obs_tuple[0]
            kinematics_data_dict_step = obs_tuple[1]

            act_str = [f"{a:.2f}" for a in action] if isinstance(action, np.ndarray) else str(action)
            # Print info periodically or at the end of an episode segment
            if (step_count) % 20 == 0 or terminated or truncated :
                print(f"  St {step_count:3d}, Act: {act_str}, Rew: {reward_val:7.3f}, Lidar Sum: {np.sum(lidar_data):6.2f}, Succ: {info.get('is_success', False)}")

            env.render() # Render the environment (if in "human" mode)

            if terminated:
                print(f"Terminated after {step_count} steps. Success: {info.get('is_success', False)}")
            if truncated:
                print(f"Truncated after {step_count} steps. Success: {info.get('is_success', False)}") # Should also check success on truncation

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward_ep:.2f}, Total Steps: {step_count}")
    env.close()
    print("\n--- Test Finished ---")
