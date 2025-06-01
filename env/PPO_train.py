import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
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
# env = gym.make(env_id, render_mode="human")
env = gym.make(env_id)
env = TupleToDictObsWrapper(env)
# check_env(env, warn=True)

total_timesteps = int(3e6)
save_step = int(1e5)
current_step = 0

checkpoint_callback = CheckpointCallback(
    save_freq=save_step,
    save_path='./ppo_new',
    name_prefix='ppo_model'
)

model = PPO(
    "MultiInputPolicy",    
    env,
    tensorboard_log='./ppo_new/tensorboard',
    verbose=1
)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

env.close()