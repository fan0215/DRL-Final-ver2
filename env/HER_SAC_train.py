import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
import driving_class  

env = gym.make("DrivingClass-v0", render_mode="human")

vec_env = DummyVecEnv([lambda: env])

max_episode_length = getattr(vec_env.envs[0].unwrapped, "_max_episode_steps", 300)

model = SAC(
    policy="MultiInputPolicy",            
    env=vec_env,
    replay_buffer_class=HerReplayBuffer,  
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                     
        goal_selection_strategy="future",     
    ),
    verbose=1,
    tensorboard_log="./her_sac_tensorboard/", 
    batch_size=256,        
    learning_rate=3e-4,    
)

model.learn(total_timesteps=200_000)
model.save("her_sac_driving_class")

env.close()
