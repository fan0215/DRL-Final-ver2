# test.py

import racetrack_env  # Import this file to ensure RacetrackEnv is registered with Gymnasium
import gymnasium as gym
import time  # You can uncomment time.sleep() below to add delays between steps

# Create the environment instance using the ID registered in racetrack_env.py
# render_mode="human" will open and update the simulation window automatically
env_id = "Racetrack-v0"
try:
    env = gym.make(env_id, render_mode="human")
except gym.error.NameNotFound as e:
    print(f"Error: Environment ID '{env_id}' not found.")
    print("Please ensure the following:")
    print("1. You have a file named 'racetrack_env.py' with the RacetrackEnv class.")
    print("2. 'racetrack_env.py' includes the registration code at the end:")
    print("   from gymnasium.envs.registration import register")
    print(f"   register(id='{env_id}', entry_point='racetrack_env:RacetrackEnv')")
    print("3. 'racetrack_env.py' is in the same directory as test.py or in your Python path.")
    exit()

try:
    episode_count = 0
    max_episodes = 100  # You can set a maximum number of episodes, or let it run indefinitely

    # To limit the number of episodes, use:
    # for episode_count in range(1, max_episodes + 1):
    while True:  # Infinite loop, continuously starts new episodes
        episode_count += 1
        print(f"----------------------------------")
        print(f"Starting Episode: {episode_count}")

        # Reset the environment to begin a new episode
        # obs is a dictionary, since RacetrackEnv uses OccupancyGrid as the observation type
        # info is also a dictionary containing additional information
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # Loop for a single episode
        while not (terminated or truncated):
            # Sample a random action from the action space
            # RacetrackEnv uses a ContinuousAction action space
            action = env.action_space.sample()

            # Take a step in the environment; get the next state, reward, termination flags, and additional info
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            # Usually, env.render() is called automatically within env.step() when render_mode="human".
            # If the window does not update in real time, or if you want more precise control, you can manually call render().
            # For most highway-env style environments, manual rendering is not needed.
            # env.render()

            # If you want to slow down the simulation for observation, add a brief delay here
            # time.sleep(0.05)  # For example, delay each step by 0.05 seconds

            if terminated:
                print(f"Episode {episode_count} terminated after {step_count} steps.")
            if truncated:
                print(f"Episode {episode_count} truncated after {step_count} steps (e.g., time limit reached).")

            # You can print step-by-step details here if needed
            # print(f"Step: {step_count}, Action: {action}, Reward: {reward:.2f}")

        print(f"Episode {episode_count} finished.")
        print(f"Total steps: {step_count}")
        print(f"Total reward for episode {episode_count}: {total_reward:.2f}")
        if "lane_centering_reward" in info:  # 'info' keys may vary depending on AbstractEnv's _info method
            print(f"Final info: Speed: {info.get('speed', 'N/A')}, Crashed: {info.get('crashed', 'N/A')}, On_road: {info.get('on_road', 'N/A')}")

        # If you want a pause between episodes, add time.sleep() here
        # time.sleep(1)

except KeyboardInterrupt:
    print("\n==================================")
    print("Simulation stopped by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'env' in locals() and env is not None:  # Ensure the environment was initialized
        env.close()
        print("Environment closed.")
    print("Exiting test script.")

