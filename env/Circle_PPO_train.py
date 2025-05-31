import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import racetrack_env

env_id = "Racetrack-v0"
env = gym.make(env_id, render_mode="human")

check_env(env, warn=True)

model = PPO(
    "MlpPolicy",    
    env,
    verbose=1,      
    tensorboard_log="./ppo_racetrack_tensorboard/"
)

total_timesteps = 100_000   
model.learn(total_timesteps=total_timesteps)

model.save("ppo_racetrack")

env.close()