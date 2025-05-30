"""
SquareTrackEnv — 回字形封閉車道（highway-env 1.10.x）
"""
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.behavior import IDMVehicle
from gymnasium.envs.registration import register # 移到這裡，因為後面會用到

class SquareTrackEnv(AbstractEnv):
    # ---------- 預設參數 ----------
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(
            dict(
                lanes_count=1,
                vehicles_count=6,
                duration=60,  # seconds
                simulation_frequency=15, # Hz
                policy_frequency=5, # Hz
                screen_width=600,
                screen_height=600,
                centering_position=[0.5, 0.5],
                scaling=8,
                # 添加一些終止條件相關的配置 (可選)
                collision_reward=-1, # 碰撞時的懲罰 (如果要在 reward 中使用)
                offroad_terminal=True, # 駛出道路是否終止
                crash_terminal=True,   # 碰撞是否終止
            )
        )
        return cfg

    # ---------- reset ----------
    def _reset(self, **kwargs):
        self._create_road()
        self._create_vehicles()
        # 在 AbstractEnv 的 _reset 中，self.time 會被重置為 0
        # 如果你的 AbstractEnv 版本較舊或有不同，可能需要手動 self.time = 0

    # ---------- 建路 ----------
    def _create_road(self):
        side = 100.0
        width = 4.0
        pts = np.array([(0, 0), (side, 0), (side, side), (0, side)])

        net = RoadNetwork()
        self._lane_keys = []
        for i in range(4):
            start = tuple(pts[i])
            end   = tuple(pts[(i + 1) % 4])
            lane  = StraightLane(start, np.subtract(end, start), width=width)
            net.add_lane(start, end, lane)
            self._lane_keys.append((start, end, 0))

        self.road = Road(network=net, np_random=self.np_random)

    # ---------- 放車 ----------
    def _create_vehicles(self):
        # 受控車
        lane = self.road.network.get_lane(self._lane_keys[0])
        self.vehicle = self.action_type.vehicle_class( # highway-env 會自動設定 action_type
            self.road, lane.position(10, 0), speed=25
        )
        self.road.vehicles.append(self.vehicle)

        # 交通車
        for _ in range(self.config["vehicles_count"] - 1):
            idx = self.np_random.integers(0, len(self._lane_keys))
            lane = self.road.network.get_lane(self._lane_keys[idx])

            s = self.np_random.uniform(0, lane.length)
            pos = lane.position(s, 0)
            heading = lane.heading_at(s)
            speed = self.np_random.uniform(15, 30)

            v = IDMVehicle(self.road, position=pos, heading=heading, speed=speed)
            try: # 嘗試添加車輛，如果太近可能會失敗
                self.road.vehicles.append(v)
            except ValueError: # 例如 "ValueError: Cannot add vehicle, a collision would occur."
                pass # 如果添加失敗，就跳過這輛車

    # ---------- 回饋函數（最小版：固定 0） ----------
    def _reward(self, action):
        # 這裡可以根據需求設計更複雜的獎勵
        # 例如：根據速度、是否碰撞、是否在路上等
        # if self.vehicle.crashed:
        #     return self.config["collision_reward"]
        # return self.vehicle.speed / 30.0 # 簡單獎勵速度
        return 0.0

    # ---------- 判斷是否終止 ----------
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or went off road."""
        terminated = False
        if self.config["crash_terminal"] and self.vehicle.crashed:
            terminated = True
        if self.config["offroad_terminal"] and not self.vehicle.on_road:
            terminated = True
        return terminated

    # ---------- 判斷是否截斷 ----------
    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        # self.time 是 AbstractEnv 維護的當前時間 (秒)
        return self.time >= self.config["duration"]

    # ---------- 額外資訊 (可選，但建議實作) ----------
    def _info(self, obs, action) -> dict:
        """Return a dictionary of additional information."""
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "on_road": self.vehicle.on_road,
            "action": action,
        }
        # 你可以添加更多有用的資訊
        return info

# ---------- 註冊 ----------
# from gymnasium.envs.registration import register # 移到檔案開頭

register(
    id="SquareTrack-v0",
    entry_point="square_track_env:SquareTrackEnv", # 如果你的檔案名稱是 square_track_env.py
)
