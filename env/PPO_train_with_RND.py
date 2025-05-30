import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from rllte.xplore.reward import RND
import torch as th
import numpy as np
from gymnasium import spaces
import driving_class  # Triggers internal registration of DrivingClass-v0


# Custom TensorBoard callback for logging intrinsic rewards
class IntrinsicRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.intrinsic_reward = 0.0  # Placeholder for intrinsic reward

    def _on_step(self) -> bool:
        # Access the intrinsic reward from the environment
        env = self.training_env.envs[0]  # Get the first (and only) environment
        if hasattr(env, 'intrinsic_reward'):
            self.intrinsic_reward = env.intrinsic_reward
            self.logger.record('train/intrinsic_reward', self.intrinsic_reward)
        return True

# Wrapper to flatten Dict observation space into a single Box
class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float64
        )

    def observation(self, obs):
        return np.concatenate([
            obs['achieved_goal'],
            obs['desired_goal'],
            obs['observation']
        ])

# Custom reward wrapper for intrinsic rewards
class IntrinsicRewardEnv(gym.Wrapper):
    def __init__(self, env, intrinsic_module, intrinsic_weight=0.01):
        super().__init__(env)
        self.intrinsic_module = intrinsic_module
        self.intrinsic_weight = intrinsic_weight
        self.last_obs = None
        self.intrinsic_reward = 0.0  # Store intrinsic reward for logging

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = 0.0
        if self.last_obs is not None and self.intrinsic_module is not None:
            # Reshape to (n_steps, n_envs, *shape)
            last_obs_tensor = th.tensor(self.last_obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 18]
            action_tensor = th.tensor(action, dtype=th.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 18]
            reward_tensor = th.tensor([extrinsic_reward], dtype=th.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
            terminated_tensor = th.tensor([terminated], dtype=th.bool).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
            truncated_tensor = th.tensor([truncated], dtype=th.bool).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
            samples = {
                "observations": last_obs_tensor,
                "actions": action_tensor,
                "next_observations": obs_tensor,
                "rewards": reward_tensor,
                "terminateds": terminated_tensor,
                "truncateds": truncated_tensor
            }
            intrinsic_reward_tensor = self.intrinsic_module.compute(samples, sync=True)
            if th.isnan(intrinsic_reward_tensor).any():
                print("Warning: Intrinsic reward is NaN, using 0.0")
                intrinsic_reward = 0.0
            else:
                intrinsic_reward = intrinsic_reward_tensor.item()
            # print(f"Intrinsic reward: {intrinsic_reward:.4f}")  # Confirm RND usage
        self.intrinsic_reward = intrinsic_reward  # Store for callback
        total_reward = extrinsic_reward + self.intrinsic_weight * intrinsic_reward
        self.last_obs = obs
        return obs, total_reward, terminated, truncated, info

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

# Create environment
try:
    env = gym.make("DrivingClass-v0", render_mode="human")
    print("Environment created successfully")
    print("Original observation space:", env.observation_space)
    print("Action space:", env.action_space)
except Exception as e:
    print(f"Error creating environment: {e}")
    print("Ensure driving_class module is in PYTHONPATH and registers DrivingClass-v0")
    exit(1)

# Apply wrappers
env = TupleToDictObsWrapper(env)
env = FlattenDictWrapper(env)
env = IntrinsicRewardEnv(env, intrinsic_module=None, intrinsic_weight=0.01)
vec_env = DummyVecEnv([lambda: env])

# Instantiate RND module
device = "cuda" if th.cuda.is_available() else "cpu"
intrinsic_module = RND(
    envs=vec_env,
    device=device,
    beta=1.0,
    kappa=0.0,
    gamma=None,
    rwd_norm_type="none",  # Avoid variance issue
    obs_norm_type="none",
    latent_dim=128,
    lr=1e-4,  # Stable learning rate
    batch_size=32,  # Suitable for small batches
    update_proportion=1.0,
    weight_init="orthogonal"
)
env.intrinsic_module = intrinsic_module

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./ppo_driving_class_tensorboard/"
)

# Train model with TensorBoard callback
callback = CallbackList([IntrinsicRewardCallback()])
try:
    model.learn(total_timesteps=100_000, callback=callback)
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save model
model.save("ppo_driving_class_rnd")

# Close environment
vec_env.close()