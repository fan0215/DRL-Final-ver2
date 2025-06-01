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


def make_env():
    env = gym.make("DrivingClass-v0")
    env = TupleToDictObsWrapper(env)
    return env

env = DummyVecEnv([make_env])
model = SAC.load("sac/sac_model_2500000_steps", env=env)
obs = env.reset()

show_env = TupleToDictObsWrapper(gym.make("DrivingClass-v0", render_mode="human"))
show_env.reset()

terminated, truncated = False, False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
    show_env.step(action[0])
    show_env.render()
env.close()
