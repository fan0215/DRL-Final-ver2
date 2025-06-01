import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import driving_class
from gymnasium import spaces


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


env = gym.make("DrivingClass-v0", render_mode="human")
env = TupleToDictObsWrapper(env)
model = PPO.load("ppo/ppo_model_2700000_steps")
obs, info = env.reset()

terminated, truncated = False, False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
env.close()
