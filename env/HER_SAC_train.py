import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
import os

import driving_class  # 這行要根據你的環境模組來寫

log_dir = "./sac_new/tensorboard"
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
    env = gym.make("DrivingClass-v0")
    env = TupleToDictObsWrapper(env)
    return env

vec_env = DummyVecEnv([make_env])

total_timesteps = int(3e6)
save_step = int(1e5)
current_step = 0

checkpoint_callback = CheckpointCallback(
    save_freq=save_step,
    save_path='./sac_new',
    name_prefix='sac_model'
)

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

model.learn(total_timesteps=3000000, callback=checkpoint_callback)

# env.close()
vec_env.close()
