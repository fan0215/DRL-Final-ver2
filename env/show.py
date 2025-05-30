import gymnasium as gym
from stable_baselines3 import PPO
import racetrack_env

env = gym.make("Racetrack-v0", render_mode="human")
model = PPO.load("ppo_racetrack")

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()  
env.close()
