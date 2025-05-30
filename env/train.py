# test.py

import racetrack_env  # 匯入此檔案以確保 RacetrackEnv 被 Gymnasium 註冊
import gymnasium as gym
import time # 可以取消註解以在步驟之間添加延遲

# 創建環境實例，使用在 racetrack_env.py 中註冊的 ID
# render_mode="human" 會自動開啟並更新模擬視窗
env_id = "Racetrack-v0"
try:
    env = gym.make(env_id, render_mode="human")
except gym.error.NameNotFound as e:
    print(f"Error: Environment ID '{env_id}' not found. ")
    print("Please ensure:")
    print(f"1. You have a file named 'racetrack_env.py' with the RacetrackEnv class.")
    print(f"2. 'racetrack_env.py' includes the registration code at the end:")
    print(f"   from gymnasium.envs.registration import register")
    print(f"   register(id='{env_id}', entry_point='racetrack_env:RacetrackEnv')")
    print(f"3. 'racetrack_env.py' is in the same directory as test.py or in Python's path.")
    exit()


try:
    episode_count = 0
    max_episodes = 100 # 你可以設定一個最大 episode 數量，或者讓它無限運行直到手動停止

    # for episode_count in range(1, max_episodes + 1): # 如果想限制 episode 數量
    while True: # 無限迴圈，用於不斷開始新的 episodes
        episode_count += 1
        print(f"----------------------------------")
        print(f"Starting Episode: {episode_count}")
        
        # 重置環境，開始新的 episode
        # obs 是一個字典，因為 RacetrackEnv 的 observation type 是 OccupancyGrid
        # info 也是一個字典，包含額外資訊
        obs, info = env.reset()
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # 單個 episode 的迴圈
        while not (terminated or truncated):
            # 從動作空間中隨機採樣一個動作
            # RacetrackEnv 的 action type 是 ContinuousAction
            action = env.action_space.sample() 
            
            # 執行動作，獲取下一步的狀態、獎勵、是否終止/截斷以及額外資訊
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1

            # env.render() 通常由 render_mode="human" 在 env.step() 內部自動調用。
            # 如果視窗沒有即時更新，或者你想更精確控制渲染時機，可以取消註解下面這行。
            # 但對於 highway-env 的環境，通常不需要手動調用 render()。
            # env.render()

            # 如果你想放慢模擬速度以便觀察，可以加入短暫延遲
            # time.sleep(0.05) # 例如，每步延遲 0.05 秒

            if terminated:
                print(f"Episode {episode_count} terminated after {step_count} steps.")
            if truncated:
                print(f"Episode {episode_count} truncated after {step_count} steps (e.g., time limit reached).")
            
            # 可以在這裡打印每一步的資訊 (如果需要)
            # print(f"Step: {step_count}, Action: {action}, Reward: {reward:.2f}")

        print(f"Episode {episode_count} finished.")
        print(f"Total steps: {step_count}")
        print(f"Total reward for episode {episode_count}: {total_reward:.2f}")
        if "lane_centering_reward" in info : # info可能不包含這些key，取決於AbstractEnv的_info
             print(f"Final info: Speed: {info.get('speed', 'N/A')}, Crashed: {info.get('crashed', 'N/A')}, On_road: {info.get('on_road', 'N/A')}")


        # 如果你想在 episodes 之間有停頓，可以在這裡加 time.sleep()
        # time.sleep(1) 

except KeyboardInterrupt:
    print("\n==================================")
    print("Simulation stopped by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'env' in locals() and env is not None: # 確保 env 已經被初始化
        env.close()
        print("Environment closed.")
    print("Exiting test script.")
