import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces

import driving_class
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

env_id = "DrivingClass-v0"
env = gym.make(env_id, render_mode="human")
env = TupleToDictObsWrapper(env)
# check_env(env, warn=True)

model = PPO(
    "MultiInputPolicy",    
    env,
    verbose=1,      
    tensorboard_log="./ppo_driving_class_tensorboard/"
)

total_timesteps = 100_000   
model.learn(total_timesteps=total_timesteps)

model.save("ppo_driving_class")

env.close()