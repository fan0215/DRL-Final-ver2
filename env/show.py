import gymnasium as gym
from stable_baselines3 import PPO
import driving_class

env = gym.make("DrivingClass-v0", render_mode="human")
model = PPO.load("ppo_driving_class")

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()  
env.close()
