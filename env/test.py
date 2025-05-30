import square_track_env # 確保環境被註冊
import gymnasium as gym
# import time # 如果你想在每一步之間加入延遲，可以取消註解

env = gym.make("SquareTrack-v0", render_mode="human")

try:
    episode_count = 0
    while True: # 無限迴圈，用於不斷開始新的 episodes
        episode_count += 1
        print(f"Starting Episode: {episode_count}")
        obs, info = env.reset() # 重置環境，開始新的 episode
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated): # 單個 episode 的迴圈
            action = env.action_space.sample() # 代理採取隨機動作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # env.render() 通常由 render_mode="human" 在 env.step() 內部自動調用。
            # 如果視窗沒有更新，可以嘗試取消註解下面這行，但一般不需要。
            # env.render()

            # 如果你想放慢模擬速度以便觀察，可以加入短暫延遲
            # time.sleep(0.05) # 例如，每步延遲 0.05 秒

            if terminated:
                print(f"Episode {episode_count} terminated.")
            if truncated:
                print(f"Episode {episode_count} truncated (time limit or other condition).")

        print(f"Episode {episode_count} finished. Final reward: {reward}")
        # 此處，一個 episode 結束，外部的 while True 會使其重新開始
        # 如果你想在 episodes 之間有停頓，可以在這裡加 time.sleep()

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    env.close()
    print("Environment closed.")
