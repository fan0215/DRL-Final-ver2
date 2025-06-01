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
            "lidar": spaces.Box(
                low=lidar_space.low.astype('float32'),
                high=lidar_space.high.astype('float32'),
                shape=lidar_space.shape,
                dtype='float32'
            ),
            "observation": spaces.Box(
                low=kgo_space['observation'].low.astype('float32'),
                high=kgo_space['observation'].high.astype('float32'),
                shape=kgo_space['observation'].shape,
                dtype='float32'
            ),
            "achieved_goal": spaces.Box(
                low=kgo_space['achieved_goal'].low.astype('float32'),
                high=kgo_space['achieved_goal'].high.astype('float32'),
                shape=kgo_space['achieved_goal'].shape,
                dtype='float32'
            ),
            "desired_goal": spaces.Box(
                low=kgo_space['desired_goal'].low.astype('float32'),
                high=kgo_space['desired_goal'].high.astype('float32'),
                shape=kgo_space['desired_goal'].shape,
                dtype='float32'
            ),
        })

    def observation(self, obs):
        lidar, kgo = obs
        return {
            "lidar": lidar.astype('float32'),
            "observation": kgo["observation"].astype('float32'),
            "achieved_goal": kgo["achieved_goal"].astype('float32'),
            "desired_goal": kgo["desired_goal"].astype('float32'),
        }

def make_env():
    env = gym.make("DrivingClass-v0")
    env = TupleToDictObsWrapper(env)
    return env

env = DummyVecEnv([make_env])
model = DDPG.load("ddpg/ddpg_model_1800000_steps", env=env)
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
