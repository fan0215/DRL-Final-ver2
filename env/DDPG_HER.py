import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import driving_class

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
    env = gym.make("DrivingClass-v0", render_mode="human")
    env = TupleToDictObsWrapper(env)
    return env

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = DDPG(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        batch_size=256,            
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=3000,     
        tensorboard_log="./ddpg_her_tensorboard/"
    )


    model.learn(total_timesteps=500_000)
    model.save("ddpg_her_driving_class")

    print("Training complete. Model saved as 'ddpg_her_driving_class'.")
