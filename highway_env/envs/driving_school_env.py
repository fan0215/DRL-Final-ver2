from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from gymnasium.envs.registration import register
from highway_env.road.lane import CircularLane, LineType
from highway_env.vehicle.objects import Landmark, Obstacle


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
                    "target_speeds": [-10, -5, 0, 5, 10], # This is where target_speeds is defined
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300, # seconds
                "collision_reward": -1.0,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1.0,
                "action_reward": -0.3,
                "checkpoint_reward": 10.0,
                "checkpoint_radius": 5.0,
                "success_goal_reward": 50,
                "controlled_vehicles": 1,
                "other_vehicles": 3,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "scaling": 5.0,
                "speed_limit": 10.0,
                "track_side_length": 30.0,
                "lane_width": 4.0,
                "show_trajectories": False,
                "offroad_terminal": True, # Terminate if vehicle goes off-road
                "steering_range": np.deg2rad(45),
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
        width = 4.0
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        line_type = [[c, c], [n, c]]
        x_offset = 10
        y_offset = 10
        size = 75
        gap = 8
        length = [10]
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

    def _make_vehicles(self) -> None:
        """
        Populate the square track with vehicles.
        """
        rng = self.np_random
        
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            chosen_lane_id_tuple = self._lane_ids[1]

            initial_speed = 0
            num_speeds = len(self.config["action"]["target_speeds"])
            if num_speeds > 0:
                initial_speed = self.config["action"]["target_speeds"][num_speeds // 2 + 1]
            
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

        # Goal
        for vehicle in self.controlled_vehicles:
            lane_id = ("a", "b", 1)
            lane = self.road.network.get_lane(lane_id)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length - 5, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)

            lane_id = ("e", "f", 0)
            lane = self.road.network.get_lane(lane_id)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)

            lane_id = ("f", "g", 0)
            lane = self.road.network.get_lane(lane_id)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
    
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