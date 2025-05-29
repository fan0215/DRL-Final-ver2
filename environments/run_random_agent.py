import gymnasium as gym
import pygame # Needed for env's rendering if it uses pygame features directly
import sys
import os

# --- MODIFICATION FOR IMPORT ---
# Add the directory containing driving_school_environment.py to sys.path
# Replace '/path/to/directory/containing/environment_file' 
# with the actual absolute or relative path.
# For example, if driving_school_environment.py is in a folder named 'my_envs' 
# located one level up from this script, you might use:
# path_to_env_dir = os.path.join(os.path.dirname(__file__), '..', 'my_envs')
# Or, provide an absolute path:
# path_to_env_dir = '/Users/yourusername/projects/my_driving_sim/my_envs'

# --- !! IMPORTANT: REPLACE THIS PLACEHOLDER !! ---
PATH_TO_DRIVING_SCHOOL_ENV_DIR = '/home/jason/code/drl/final/DRL-Final-ver2/environments' 
# --- !! END OF PLACEHOLDER !! ---

# A more robust way if the script and env file have a fixed relative structure:
# For example, if driving_school_environment.py is in the same directory as this script,
# this would be os.path.dirname(__file__) or simply '.'
# If it's in a subdirectory "envs" relative to this script:
# path_to_env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs')

# For this example, let's assume the user will provide the correct path
# or that the original file structure (same directory) is intended.
# If PATH_TO_DRIVING_SCHOOL_ENV_DIR is not the placeholder, add it to sys.path
if PATH_TO_DRIVING_SCHOOL_ENV_DIR != '/path/to/directory/containing/driving_school_environment_file':
    if os.path.isdir(PATH_TO_DRIVING_SCHOOL_ENV_DIR):
        sys.path.insert(0, os.path.abspath(PATH_TO_DRIVING_SCHOOL_ENV_DIR))
        print(f"Added to sys.path: {os.path.abspath(PATH_TO_DRIVING_SCHOOL_ENV_DIR)}")
    else:
        print(f"Warning: Provided path '{PATH_TO_DRIVING_SCHOOL_ENV_DIR}' is not a valid directory. Import might fail.")
# If they are in the same directory, adding '.' (current directory) is often implicit but can be made explicit.
elif not any(os.path.samefile(p, os.path.abspath('.')) for p in sys.path if os.path.exists(p) and os.path.isdir(p)):
     sys.path.insert(0, os.path.abspath('.')) # Add current directory if not already effectively there
     print(f"Assuming driving_school_environment.py is in the current directory or PYTHONPATH. Added '.' to sys.path for robustness.")


# Import the custom environment
try:
    import driving_school_environment 
except ImportError as e:
    print(f"Error: Could not import driving_school_environment.py: {e}")
    print("Please ensure 'driving_school_environment.py' is in the specified path or your PYTHONPATH.")
    print("Current sys.path includes:")
    for p in sys.path:
        print(f"  - {p}")
    exit()

# action_list = [4] * 100 + [1] * 100

if __name__ == '__main__':
    # Configuration for the environment can be overridden here if needed
    env_config = {
        "manual_control": True,
        "screen_width": 900,  # Slightly wider for better visibility
        "screen_height": 600, 
        "scaling": 6.0,       # Adjust zoom level
        "policy_frequency": 5, 
        "simulation_frequency": 15,
    }
    
    env = None
    # The environment ID used in registration
    env = gym.make('DrivingSchoolEnv-v0', render_mode='human', config=env_config)

    print("--- Driving School Environment Test with Random Agent ---")
    print(f"Action Space: {env.action_space}")
    if env.observation_space:
        print(f"Observation Space Shape: {env.observation_space.shape}")
    else:
        print("Observation Space: Not defined (this is unusual)")


    num_episodes = 10 
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # It's good practice to catch potential errors during reset too

        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0
        ep_step = 0
        
        current_stage_reported = info.get('current_stage', -1)
        print(f"Starting Stage: {current_stage_reported}")
        prev_info = info # Store initial info

        print(env.action_space)
        # actions = action_list.copy()

        while not (terminated or truncated):
            action = env.action_space.sample()
            # action = actions.pop(0)
            
            try:
                obs, reward, terminated, truncated, info = env.step(4)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                terminated = True # End episode on step error
                info = prev_info # Use previous info if current step failed badly
                break


            total_reward += reward
            ep_step += 1
            
            # env.render() is called internally by highway-env's AbstractEnv.step() 
            # when render_mode='human'. So, an explicit call here is usually not needed
            # unless you want to render at a different frequency or specific points.

            if ep_step % 25 == 0 or terminated or truncated: 
                print(f"  Ep {episode+1}, Step {ep_step}, Stage {info.get('current_stage', 'N/A')}, "
                      f"S.Step {info.get('stage_step_count', 'N/A')}, Rew: {reward:.2f}, TotRew: {total_reward:.2f}, "
                      f"Speed: {info.get('speed', 0):.2f}, Crash: {info.get('crashed', False)}")
                if 'failure_type' in info and info['failure_type']:
                    print(f"  Failure: {info['failure_type']}")
            
            if terminated:
                if info.get('failure_type'): # Actual crash/offroad
                    print(f"  Episode terminated at step {ep_step} due to failure: {info['failure_type']}. Agent will restart from Stage 1.")
                # Check if all stages were completed (current_stage would be TOTAL_STAGES, meaning it completed the last one)
                elif info.get('current_stage', 0) >= driving_school_environment.TOTAL_STAGES: # Use constant from imported module
                     print(f"  All stages successfully completed in {ep_step} steps!")
                else: # Stage completed, will reset to next stage
                    print(f"  Stage {prev_info.get('current_stage', 'N/A')} completed. Episode ends to reset for next stage.")
            elif truncated:
                print(f"  Episode truncated at step {ep_step} (stage {info.get('current_stage','N/A')} time limit).")
            
            prev_info = info.copy() # Store current info for next iteration's stage change check

        print(f"Episode {episode + 1} finished. Total steps: {ep_step}. Total reward: {total_reward:.2f}")
        # Check if the agent passed all stages based on the final state before reset
        if prev_info.get('current_stage', 0) >= driving_school_environment.TOTAL_STAGES and not prev_info.get('failure_type'):
             print("--- AGENT PASSED DRIVING SCHOOL (Randomly)! ---")

    env.close()
    print("\n--- Simulation Finished ---")
