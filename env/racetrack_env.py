from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from gymnasium.envs.registration import register
from highway_env.road.lane import CircularLane, LineType


class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.
    The agent needs to learn two skills:
    - follow the tracks (now a square/rectangular track)
    - avoid collisions with other vehicles
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-20, 20], [-20, 20]],
                    "grid_step": [2, 2],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10], # This is where target_speeds is defined
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300, # seconds
                "collision_reward": -1.0,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1.0,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 3,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "scaling": 7.0,
                "speed_limit": 10.0,
                "track_side_length": 30.0,
                "lane_width": 4.0,
                "show_trajectories": False,
                "offroad_terminal": True, # Terminate if vehicle goes off-road
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward_value for name, reward_value in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], self.config["lane_centering_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        lateral_offset = 0.0
        on_road = self.vehicle.on_road
        if on_road and self.vehicle.lane: # Check if vehicle.lane is not None
            _, lateral_offset = self.vehicle.lane.local_coordinates(self.vehicle.position)
        
        lane_centering_reward = 0.0
        if on_road:
             lane_centering_reward = 1 / (1 + self.config["lane_centering_cost"] * lateral_offset**2)

        return {
            "lane_centering_reward": lane_centering_reward,
            "action_reward": np.linalg.norm(action),
            "collision_reward": float(self.vehicle.crashed),
            "on_road_reward": float(on_road),
        }

    def _is_terminated(self) -> bool:
        terminated = self.vehicle.crashed
        if self.config["offroad_terminal"] and not self.vehicle.on_road:
            terminated = True
        return terminated

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._create_road()
        self._make_vehicles()
        if not self.config["action"].get("longitudinal", True) and hasattr(self.vehicle, 'target_speed'):
            # Set a default target speed from the middle of the available speeds
            # Access target_speeds correctly from the 'action' sub-dictionary
            num_speeds = len(self.config["action"]["target_speeds"]) # CORRECTED
            if num_speeds > 0:
                self.vehicle.target_speed = self.config["action"]["target_speeds"][num_speeds // 2] # CORRECTED


    def _create_road(self) -> None:
        """
        Create a circular (ring-shaped) road.
        """
        net = RoadNetwork()
        lane_width = self.config["lane_width"]
        side_length = self.config["track_side_length"]
        radius = 200

        center = np.array([0, 0])

        lane = CircularLane(
            center=center,
            radius=radius,
            start_phase=0,        # 從0度起始
            end_phase=2 * np.pi,  # 到360度結束
            width=lane_width,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
            speed_limit=self.config["speed_limit"]
        )
        # 範例只用一個lane_id
        net.add_lane("circle", "circle", lane)
        self._lane_ids = [("circle", "circle", 0)]

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False)
        )

    def _make_vehicles(self) -> None:
        """
        Populate the square track with vehicles.
        """
        rng = self.np_random
        
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            start_lane_tuple_idx = rng.choice(len(self._lane_ids))
            chosen_lane_id_tuple = self._lane_ids[start_lane_tuple_idx]

            initial_speed = 0
            # Access target_speeds correctly from the 'action' sub-dictionary
            num_speeds = len(self.config["action"]["target_speeds"]) # CORRECTED
            if num_speeds > 0:
                initial_speed = self.config["action"]["target_speeds"][num_speeds // 2] # CORRECTED
            
            lane_object = self.road.network.get_lane(chosen_lane_id_tuple)
            initial_longitudinal = rng.uniform(low=lane_object.length * 0.05, high=lane_object.length * 0.2)
            
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=chosen_lane_id_tuple,
                speed=initial_speed,
                longitudinal=initial_longitudinal 
            )
            
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            
            if i == 0:
                self.vehicle = controlled_vehicle

        for _ in range(self.config["other_vehicles"]):
            if not hasattr(self, 'road') or not self.road or not self.road.network:
                print("Warning: Road or network not initialized in _make_vehicles. Skipping NPC creation.")
                continue
            if not self._lane_ids:
                print("Warning: No lane IDs available for NPC creation.")
                continue

            random_lane_tuple = self.road.network.random_lane_index(rng)
            if not random_lane_tuple:
                random_lane_tuple = self._lane_ids[rng.choice(len(self._lane_ids))]

            actual_lane = self.road.network.get_lane(random_lane_tuple)
            if not actual_lane:
                print(f"Warning: Could not get lane for {random_lane_tuple}. Skipping NPC.")
                continue
            
            npc_speed = rng.uniform(low=self.config["speed_limit"] * 0.2, high=self.config["speed_limit"] * 0.6)
            
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                lane_index=random_lane_tuple,
                longitudinal=rng.uniform(low=0, high=actual_lane.length * 0.9),
                speed=npc_speed
            )
            
            can_add = True
            for v_existing in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v_existing.position) < vehicle.LENGTH * 3:
                    can_add = False
                    break
            if can_add:
                self.road.vehicles.append(vehicle)
    
    def _info(self, obs, action) -> dict:
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "on_road": self.vehicle.on_road,
            "action": action,
        }
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index is not None:
            info["lane_index"] = self.vehicle.lane_index
        if hasattr(self.vehicle, 'lane') and self.vehicle.lane is not None:
             _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
             info["lateral_offset"] = lateral
        else:
            info["lateral_offset"] = "N/A (off-lane or no lane)"
        return info

register(
    id="Racetrack-v0", 
    entry_point="racetrack_env:RacetrackEnv",
)
