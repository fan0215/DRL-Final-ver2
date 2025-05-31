import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import os

import driving_class  # 這行要根據你的環境模組來寫

log_dir = "./her_sac_tensorboard/"
os.makedirs(log_dir, exist_ok=True)
class TupleToDictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        lidar_space = env.observation_space.spaces[0]
        kgo_space = env.observation_space.spaces[1]
        self.observation_space = spaces.Dict({
            "lidar": lidar_space,
            "observation": kgo_space['observation'],
            "achieved_goal": kgo_space['achieved_goal'],
            "desired_goal": kgo_space['desired_goal'],
        })

    def observation(self, obs):
        lidar, kgo = obs
        return {
            "lidar": lidar,
            "observation": kgo["observation"],
            "achieved_goal": kgo["achieved_goal"],
            "desired_goal": kgo["desired_goal"],
        }

def make_env():
    env = gym.make("DrivingClass-v0", render_mode="human")
    env = TupleToDictObsWrapper(env)
    return env

vec_env = DummyVecEnv([make_env])

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # DummyVecEnv: infos 是一個 list
        for info in self.locals["infos"]:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                # log 到 TensorBoard 的 custom/ep_reward
                self.logger.record("custom/ep_reward", ep_rew)
        return True

reward_callback = RewardLoggerCallback()

model = SAC(
    policy="MultiInputPolicy",            
    env=vec_env,
    replay_buffer_class=HerReplayBuffer,  
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                     
        goal_selection_strategy="future",     
    ),
    verbose=1,
    tensorboard_log=log_dir, 
    batch_size=256,        
    learning_rate=3e-4,    
)

model.learn(total_timesteps=200_000, callback=reward_callback)
model.save("her_sac_driving_class")

env.close()
vec_env.close()
