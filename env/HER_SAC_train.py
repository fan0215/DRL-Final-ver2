import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os

# ====== 1. Import your custom environment =====
import driving_class  # 這行要根據你的環境模組來寫

# ====== 2. 設定 TensorBoard log 目錄 ======
log_dir = "./her_sac_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# ====== 3. 建立 Gymnasium 環境並包裝 VecEnv ======
env = gym.make("DrivingClass-v0")
vec_env = DummyVecEnv([lambda: gym.make("DrivingClass-v0")])

# ====== 4. 自訂 Callback log 每集 reward 到 TensorBoard ======
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # DummyVecEnv: infos 是一個 list
        for info in self.locals["infos"]:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                # log 到 TensorBoard 的 custom/ep_reward
                self.logger.record("custom/ep_reward", ep_rew)
        return True

reward_callback = RewardLoggerCallback()

# ====== 5. 建立 SAC + HER 模型 ======
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

# ====== 6. 訓練模型 ======
model.learn(total_timesteps=200_000, callback=reward_callback)
model.save("her_sac_driving_class")

# ====== 7. 關閉環境 ======
env.close()
vec_env.close()
