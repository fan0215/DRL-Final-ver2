from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._create_road()
        self._make_vehicles()

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        line_type = [[c, c], [c, n]]
        x_offset = 10
        y_offset = 10
        size = 100
        gap = 15
        length = [20]

        # (0, 0) to (0, 1)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [x_offset + width, y_offset],
                [x_offset + width, y_offset + width * 2],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [x_offset + width, y_offset + width * 2 + size],
                [x_offset + width, y_offset + width * 4 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )
        
        # (0, 1) to (1, 1)
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset, y_offset + width * 3 + size],
                [x_offset + width * 2, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 3 + size],
                [x_offset + width * 2 + gap + width * 2 + gap + length[0], y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [x_offset + width * 2 + size, y_offset + width * 3 + size],
                [x_offset + width * 4 + size, y_offset + width * 3 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )

        # (1, 1) to (1, 0)
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [x_offset + width * 3 + size, y_offset + width * 4 + size],
                [x_offset + width * 3 + size, y_offset + width * 2 + size],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [x_offset + width * 3 + size, y_offset + width * 2],
                [x_offset + width * 3 + size, y_offset],
                width=width * 2,
                line_types=line_type[1]
            )
        )

        # (1, 0) to (0, 0)
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [x_offset + width * 4 + size, y_offset + width],
                [x_offset + width * 2 + size, y_offset + width],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [x_offset + width * 2, y_offset + width],
                [x_offset + width, y_offset + width],
                width=width * 2,
                line_types=line_type[1]
            )
        )

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
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [x_offset + width * 2 + gap, y_offset + width * 2 + size - length[0] + (width * 2 + length[0]) / 2],
                [x_offset + width * 2 + gap + width * 2, y_offset + width * 2 + size - length[0] + (width * 2 + length[0]) / 2],
                width=width * 2 + length[0],
                line_types=line_type[0]
            )
        )

        # 路邊停車
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [x_offset + width * 2 + gap + width * 2 + gap + length[0], y_offset + width * 2 + size - width],
                [x_offset + width * 2 + gap + width * 2 + gap, y_offset + width * 2 + size - width],
                width=width * 2,
                line_types=line_type[1]
            )
        )
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

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                random_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    random_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(random_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)
from gymnasium.envs.registration import register

register(
    id="Racetrack-v0", # 你可以選擇一個ID
    entry_point="racetrack_env:RacetrackEnv", # 假設你的檔案名是 racetrack_env.py
)

