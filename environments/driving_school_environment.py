import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import pygame # For custom rendering within the environment's render method

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle # Keep if specific controller features are used, else Vehicle is often enough for controlled ones
from highway_env.envs.common.action import DiscreteMetaAction, Action # Action is good, DiscreteMetaAction if that's the chosen type
from highway_env.road.lane import StraightLane, CircularLane, LineType, AbstractLane
from highway_env.utils import near_split

# Constants for stages
STAGE_STRAIGHT = 0
STAGE_VERTICAL_PARKING = 1
STAGE_PARALLEL_PARKING = 2
STAGE_S_CURVE = 3
STAGE_TRAFFIC_LIGHT = 4
STAGE_LEVEL_CROSSING = 5
STAGE_NARROW_STRAIGHT = 6
TOTAL_STAGES = 7

# Traffic Light States
LIGHT_RED = 0
LIGHT_GREEN = 1
LIGHT_YELLOW = 2 # Optional, currently not used in cycle logic

# Level Crossing States
CROSSING_CLOSED = 0
CROSSING_OPEN = 1


class DrivingSchoolEnv(AbstractEnv):
    """
    A multi-stage driving school environment.
    The agent must complete a series of tasks:
    1. Drive straight.
    2. Park backward into a vertical slot.
    3. Park backward into a parallel slot.
    4. Navigate an S-curve forward and then backward.
    5. Obey a traffic light.
    6. Obey a level crossing signal.
    7. Drive through a narrow straight passage.
    Collisions or going off-road reset the agent to the beginning of Stage 1.
    """

    # --- Environment Configuration ---
    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "heading", "long_off", "lat_off", "ang_off"],
            "features_range": {
                "x": [-100, 100], "y": [-100, 100],
                "vx": [-30, 30], "vy": [-30, 30],
            },
            "absolute": False,
            "normalize": True,
            "see_behind": True,
        },
        "action": {
            "type": "DiscreteMetaAction", # Ensure your agent outputs actions compatible with this
        },
        "simulation_frequency": 15,  # Hz
        "policy_frequency": 5,  # Hz, decision frequency (synonymous with simulation_frequency if action is applied every sim step)
                                # If policy_frequency < simulation_frequency, then simulation runs multiple steps per agent action.
                                # highway-env AbstractEnv sets self.dt = 1 / self.config["policy_frequency"]
                                # and road.step is often called with this dt.
                                # If sim_freq is the physics update rate and policy_freq is agent decision rate,
                                # road.step might need to be called policy_frequency/simulation_frequency times or adjust dt.
                                # For now, assuming 1 agent step = 1 simulation step based on user's road.step call.
                                # Let's make them consistent or clarify. Default is often policy_frequency for road.step.
                                # User code calls road.step(1 / self.config["simulation_frequency"]), so this is the effective dt.
        "screen_width": 800,
        "screen_height": 600,
        "centering_position": [0.5, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        "manual_control": True, # Will require event handling in render loop if True

        "parking_slot_width": 2.5,
        "parking_slot_length": 5.0,
        "parking_spot_margin": 0.1, # Margin for checking if parked correctly
        "s_curve_radius": 20.0,
        "s_curve_lane_width": 4.0,
        "traffic_light_distance": 50.0, # x-coordinate of traffic light / crossing line
        "traffic_light_red_duration": 5 * 15, # in simulation steps
        "traffic_light_green_duration": 7 * 15, # in simulation steps
        "level_crossing_closed_duration": 6 * 15, # in simulation steps
        "level_crossing_open_duration": 8 * 15, # in simulation steps
        "narrow_lane_width": 2.8,
        "lane_width": 20.0, # Default lane width for non-narrow sections

        "collision_penalty": -100.0,
        "offroad_penalty": -75.0,
        "stage_completion_reward": 50.0, # Reward for completing an intermediate stage
        "goal_achievement_reward": 100.0, # Reward for completing all stages (can be same as stage_completion_reward)
        "time_penalty_per_step": -0.1, # Penalty per simulation step
        "control_penalty_weight": 0.01, # Example, not explicitly used in user's reward funcs yet
        "distance_reward_weight": 0.5, # General weight for distance-based rewards

        "max_steps_per_stage": { # Maximum simulation steps allowed for each stage
            STAGE_STRAIGHT: 100,
            STAGE_VERTICAL_PARKING: 250,
            STAGE_PARALLEL_PARKING: 300,
            STAGE_S_CURVE: 400, # Consider if forward and backward parts need combined or separate step counts
            STAGE_TRAFFIC_LIGHT: 150,
            STAGE_LEVEL_CROSSING: 180,
            STAGE_NARROW_STRAIGHT: 120,
        },
        "global_start_position": [0.0, 0.0], # Default initial position if a stage doesn't specify
        "global_start_heading": 0.0, # Default initial heading
    }

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG
        self.config.update(config)

        self.current_stage = STAGE_STRAIGHT
        self.stage_step_count = 0
        self.stage_goal_achieved = False # Status of current stage's primary goal

        # Traffic Light and Level Crossing states and timers
        self.traffic_light_state = LIGHT_RED
        self.traffic_light_timer = 0 # Counter for current light state duration
        self.level_crossing_state = CROSSING_CLOSED
        self.level_crossing_timer = 0 # Counter for current crossing state duration

        # Stage-specific state variables
        self.s_curve_forward_done = False # For S-curve stage: true if forward part is completed

        # Target definitions for parking stages (will be np.ndarray)
        self.target_parking_slot_center = None
        self.target_parking_slot_heading = None # Radians
        self.target_parking_slot_polygon = None # List of [x,y] vertices

        # Path/Target for S-Curve (example, might not be explicitly stored if logic is dynamic)
        # self.s_curve_path_lane = None
        # self.s_curve_target_point = None

        # General goal for current stage (can be a position, or implicitly defined by stage logic)
        self.current_goal_position = None # np.ndarray [x,y]
        self.current_goal_heading = None # Radians, if applicable

        # These will be set by stage setup methods
        self.ego_start_position = np.array(self.config["global_start_position"])
        self.ego_start_heading = self.config["global_start_heading"]
        self.ego_start_speed = 0.0
        self.other_vehicles_definitions: List[Dict] = []

        super().__init__(config=self.config, render_mode=render_mode) # AbstractEnv handles config merging

    def _reset(self) -> None:
        # This method is called by AbstractEnv.reset() after it handles common reset tasks (time, vehicle, road).
        # We don't need to return obs/info here; AbstractEnv.reset() does that.
        self.stage_step_count = 0
        self.stage_goal_achieved = False
        self.s_curve_forward_done = False
        # self.current_stage is NOT reset here if a failure occurred in the previous episode,
        # as _handle_failure would have set it to STAGE_STRAIGHT.
        # If reset is called after successful completion of all stages (current_stage >= TOTAL_STAGES),
        # we should reset it to STAGE_STRAIGHT to start a new full sequence.
        if self.current_stage >= TOTAL_STAGES:
            self.current_stage = STAGE_STRAIGHT

        self._setup_stage() # Sets up the road, vehicle, and goals for the self.current_stage

    def _setup_stage(self):
        self.road = Road(network=RoadNetwork(), np_random=self.np_random, record_history=self.config["show_trajectories"])
        
        # Reset to global defaults, specific stages will override
        self.ego_start_position = np.array(self.config["global_start_position"])
        self.ego_start_heading = self.config["global_start_heading"]
        self.ego_start_speed = 0.0
        self.other_vehicles_definitions = []
        self.current_goal_position = None
        self.target_parking_slot_polygon = None # Clear parking slot from previous stage
        self.target_parking_slot_center = None
        self.target_parking_slot_heading = None

        if self.current_stage == STAGE_STRAIGHT:
            self._setup_stage_straight_line()
        elif self.current_stage == STAGE_VERTICAL_PARKING:
            self._setup_stage_vertical_parking()
        elif self.current_stage == STAGE_PARALLEL_PARKING:
            self._setup_stage_parallel_parking()
        elif self.current_stage == STAGE_S_CURVE:
            self._setup_stage_s_curve()
        elif self.current_stage == STAGE_TRAFFIC_LIGHT:
            self._setup_stage_traffic_light()
        elif self.current_stage == STAGE_LEVEL_CROSSING:
            self._setup_stage_level_crossing()
        elif self.current_stage == STAGE_NARROW_STRAIGHT:
            self._setup_stage_narrow_straight()
        else: # Should not happen if current_stage is managed properly
            print(f"Warning: Unknown stage {self.current_stage} in _setup_stage. Resetting to Stage 0.")
            self.current_stage = STAGE_STRAIGHT
            self._setup_stage_straight_line()

        # Create ego vehicle
        # action_type is initialized in AbstractEnv based on config
        self.vehicle = self.action_type.vehicle_class(
            self.road,
            position=self.ego_start_position,
            heading=self.ego_start_heading,
            speed=self.ego_start_speed
        )
        self.road.vehicles.append(self.vehicle)

        # Link observer vehicle in observation type (if it's vehicle-centric)
        if hasattr(self.observation_type, 'observer_vehicle'):
            self.observation_type.observer_vehicle = self.vehicle

        # Add other vehicles defined by the stage setup
        for veh_def in self.other_vehicles_definitions:
            v_class = veh_def.get("class", Vehicle) # Default to basic Vehicle
            v = v_class(self.road,
                        position=np.array(veh_def["position"]),
                        heading=veh_def["heading"],
                        speed=veh_def.get("speed", 0.0))
            if "color" in veh_def: # Assuming color is a tuple e.g. (R,G,B)
                v.color = veh_def["color"]
            self.road.vehicles.append(v)
        self.other_vehicles_definitions = [] # Clear definitions after use


    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # This method now overrides gymnasium.Env.step directly.
        self.time += 1 # Manually increment time, as AbstractEnv.step is bypassed.
        self.stage_step_count += 1

        # --- Simulation ---
        if self.vehicle:
            self.action_type.act(action) # Apply agent action to ego vehicle
            self.road.act() # Other vehicles act (if they have behavior)
            # Use simulation_frequency for the dt of road.step
            self.road.step(1 / self.config["simulation_frequency"])
        else:
            # Critical error: vehicle missing after reset. Should not happen with proper _setup_stage.
            # Attempt to get an observation, though it might be meaningless
            obs = np.array([])
            if hasattr(self, 'observation_type') and self.observation_type:
                obs = self.observation_type.observe()
            return obs, self.config["collision_penalty"], True, False, self._get_info(failure=True, failure_type="vehicle_missing_error")

        self._update_stage_dynamics() # For traffic lights, level crossings

        # --- Check for Failures or Overall Success (completion of ALL stages) ---
        episode_should_end_majorly, failure_type_or_None_for_overall_success = self._check_failure_conditions()

        if episode_should_end_majorly:
            current_obs = self.observation_type.observe()
            reward = 0.0
            info_failure_type = None

            if failure_type_or_None_for_overall_success: # Actual failure
                reward = self._handle_failure(failure_type_or_None_for_overall_success) # Resets current_stage to STAGE_STRAIGHT
                info_failure_type = failure_type_or_None_for_overall_success
                info = self._get_info(failure=True, failure_type=info_failure_type)
            else: # All stages completed successfully
                reward = self.config["goal_achievement_reward"] # Special reward for completing the entire curriculum
                info = self._get_info() # No failure
            return current_obs, reward, True, False, info # terminated = True, truncated = False

        # --- If no major end condition, calculate step reward and check for *current* stage completion ---
        reward = self._reward(action) # Calculate reward for this step in the current stage
        current_stage_is_completed = self._check_stage_completion() # Check if current stage's specific goal is met

        if current_stage_is_completed:
            reward += self.config["stage_completion_reward"] # Add bonus for completing the current intermediate stage
            prev_stage_for_log = self.current_stage
            self.current_stage += 1
            
            print(f"Stage {prev_stage_for_log} completed! Advancing to Stage {self.current_stage}.")
            
            # Reset counters for the next stage (env.reset() will be called by agent, leading to _setup_stage)
            self.stage_step_count = 0
            self.stage_goal_achieved = False # Will be checked at the start of the next stage or by its setup
            self.s_curve_forward_done = False # Reset s-curve specific state

            current_obs = self.observation_type.observe()
            info = self._get_info()
            # Terminate this "stage episode" so env.reset() can set up the next stage
            return current_obs, reward, True, False, info # terminated = True (current stage done), truncated = False

        # --- If episode continues for the current stage, check for truncation (time limit for this stage) ---
        truncated = self._is_truncated() # Checks self.stage_step_count against max_steps_per_stage
        
        current_obs = self.observation_type.observe() # Observe current state
        info = self._get_info() # Get info based on current state

        if truncated: # Only apply truncation if not already terminated by failure/success/stage_completion
            # Add a lump sum penalty for timing out the current stage
            # (self.config["time_penalty_per_step"] is typically negative)
            max_steps_this_stage = self.config["max_steps_per_stage"].get(self.current_stage, 1000) # Default just in case
            reward += self.config["time_penalty_per_step"] * max_steps_this_stage
            return current_obs, reward, False, True, info # terminated = False, truncated = True

        # --- Normal step, episode continues for the current stage ---
        return current_obs, reward, False, False, info # terminated = False, truncated = False


    def _reward(self, action: Action) -> float:
        # This method calculates reward for the current step, based on the active stage.
        # It's called when the episode is ongoing (not yet terminated or truncated for major reasons).
        if not self.vehicle:
            return self.config["collision_penalty"] # Should not happen if vehicle checks are in place

        reward = self.config["time_penalty_per_step"] # Base penalty for taking a step

        # Add stage-specific reward components
        if self.current_stage == STAGE_STRAIGHT:
            reward += self._reward_stage_straight_line()
        elif self.current_stage == STAGE_VERTICAL_PARKING:
            reward += self._reward_stage_vertical_parking()
        elif self.current_stage == STAGE_PARALLEL_PARKING:
            reward += self._reward_stage_parallel_parking()
        elif self.current_stage == STAGE_S_CURVE:
            reward += self._reward_stage_s_curve()
        elif self.current_stage == STAGE_TRAFFIC_LIGHT:
            reward += self._reward_stage_traffic_light()
        elif self.current_stage == STAGE_LEVEL_CROSSING:
            reward += self._reward_stage_level_crossing()
        elif self.current_stage == STAGE_NARROW_STRAIGHT:
            reward += self._reward_stage_narrow_straight()
        
        # Example: Add a small penalty for excessive control effort (if action is continuous or has magnitude)
        # if isinstance(action, np.ndarray): # For continuous actions
        #     control_magnitude = np.linalg.norm(action)
        #     reward -= self.config["control_penalty_weight"] * control_magnitude

        return reward

    def _is_terminated(self) -> bool:
        # This method is a helper, primarily used by the step() method's logic.
        # It checks for conditions that would end the entire multi-stage task (major failure or all stages completed).
        if not self.vehicle:
            return True # Critical error
        
        # _check_failure_conditions encapsulates both fatal errors and overall success.
        failure_condition_met, _ = self._check_failure_conditions()
        return failure_condition_met

    def _is_truncated(self) -> bool:
        # Checks if the current stage has run out of allocated steps.
        # Note: self.time is total time, self.stage_step_count is for current stage
        max_steps_for_current_stage = self.config["max_steps_per_stage"].get(self.current_stage, float('inf'))
        return self.stage_step_count >= max_steps_for_current_stage

    def _check_failure_conditions(self) -> Tuple[bool, Optional[str]]:
        """
        Checks for critical failure conditions (crash, off-road) or successful completion of ALL stages.
        Returns:
            Tuple[bool, Optional[str]]:
            - bool: True if the episode should terminate due to one of these conditions.
            - Optional[str]: The type of failure (e.g., "collision", "offroad"), or None if all stages were successfully completed.
        """
        if not self.vehicle:
            return True, "vehicle_missing_critical" # Critical error if vehicle doesn't exist

        if self.vehicle.crashed:
            return True, "collision"
        
        if not self.vehicle.on_road: # vehicle.on_road itself might be True if on any part of a complex road object.
                                     # We need to check if it's on a drivable lane.
            is_on_any_lane = False
            if self.road and self.road.network:
                for lane_index in self.road.network.lanes_list(): # Iterate over LaneIndex objects
                    lane_object = self.road.network.get_lane(lane_index)
                    if lane_object.on_lane(self.vehicle.position, margin=0.5): # Check with a small margin
                        is_on_any_lane = True
                        break
            if not is_on_any_lane:
                 return True, "offroad"
        
        # Check for successful completion of ALL stages
        # This means the current_stage counter has advanced beyond the last defined stage
        # (i.e., the last stage was completed in the previous step and current_stage was incremented).
        if self.current_stage >= TOTAL_STAGES:
            return True, None # Terminate, overall success (no failure type string)

        return False, None # No critical failure, and not all stages completed yet

    def _handle_failure(self, failure_type: str) -> float:
        """
        Handles the consequences of a major failure.
        Resets the agent to the beginning of Stage 1.
        Returns the penalty associated with the failure.
        """
        print(f"Major failure: {failure_type}! Resetting to Stage 1 for the next episode.")
        self.current_stage = STAGE_STRAIGHT # Agent starts from scratch in the curriculum
        self.s_curve_forward_done = False # Reset any persistent stage-specific states
        
        # Return the penalty for this failure
        if failure_type == "collision":
            return self.config["collision_penalty"]
        elif failure_type == "offroad":
            return self.config["offroad_penalty"]
        elif failure_type == "vehicle_missing_critical" or failure_type == "vehicle_missing_error": # from step()
            return self.config["collision_penalty"] # Treat as a severe penalty
        return 0.0 # Default if somehow an unknown failure type occurs


    def _get_info(self, failure: bool = False, failure_type: Optional[str] = None) -> Dict:
        info = {
            "speed": self.vehicle.speed if self.vehicle else 0,
            "crashed": self.vehicle.crashed if self.vehicle else False, # Vehicle's own crash flag
            "on_road": self.vehicle.on_road if self.vehicle else False, # Vehicle's own on_road flag
            "action": getattr(self.action_type, 'last_action', -1), # Get last executed action if available
            "current_stage": self.current_stage,
            "stage_step_count": self.stage_step_count,
            "stage_goal_achieved_current_step": self.stage_goal_achieved, # Reflects if goal was met this step
            "s_curve_forward_done": self.s_curve_forward_done,
        }
        if self.current_stage == STAGE_TRAFFIC_LIGHT:
            info["traffic_light_state"] = self.traffic_light_state
            info["traffic_light_timer"] = self.traffic_light_timer
        if self.current_stage == STAGE_LEVEL_CROSSING:
            info["level_crossing_state"] = self.level_crossing_state
            info["level_crossing_timer"] = self.level_crossing_timer

        if failure and failure_type:
            info["failure_type"] = failure_type
        
        if self.current_goal_position is not None:
            info["goal_position"] = self.current_goal_position.tolist() # For JSON compatibility

        if self.target_parking_slot_polygon is not None and \
           (self.current_stage == STAGE_VERTICAL_PARKING or self.current_stage == STAGE_PARALLEL_PARKING) and \
           self.vehicle:
             # This check is for information; actual completion is handled by _check_completion_stage_X
             info["is_correctly_parked_info"] = self._is_vehicle_in_polygon(self.vehicle, self.target_parking_slot_polygon, margin=self.config["parking_spot_margin"])
        return info

    def _update_stage_dynamics(self):
        # Handles time-based changes like traffic lights or level crossings
        if self.current_stage == STAGE_TRAFFIC_LIGHT:
            self.traffic_light_timer += 1
            if self.traffic_light_state == LIGHT_RED and self.traffic_light_timer >= self.config["traffic_light_red_duration"]:
                self.traffic_light_state = LIGHT_GREEN
                self.traffic_light_timer = 0
            elif self.traffic_light_state == LIGHT_GREEN and self.traffic_light_timer >= self.config["traffic_light_green_duration"]:
                self.traffic_light_state = LIGHT_RED
                self.traffic_light_timer = 0
        
        elif self.current_stage == STAGE_LEVEL_CROSSING:
            self.level_crossing_timer += 1
            if self.level_crossing_state == CROSSING_CLOSED and self.level_crossing_timer >= self.config["level_crossing_closed_duration"]:
                self.level_crossing_state = CROSSING_OPEN
                self.level_crossing_timer = 0
            elif self.level_crossing_state == CROSSING_OPEN and self.level_crossing_timer >= self.config["level_crossing_open_duration"]:
                self.level_crossing_state = CROSSING_CLOSED
                self.level_crossing_timer = 0
    
    def _check_stage_completion(self) -> bool:
        # Checks if the goal of the *current* specific stage is met.
        # Does not handle overall curriculum completion.
        if not self.vehicle: return False
        
        completed = False
        if self.current_stage == STAGE_STRAIGHT:
            completed = self._check_completion_stage_straight_line()
        elif self.current_stage == STAGE_VERTICAL_PARKING:
            completed = self._check_completion_stage_vertical_parking()
        elif self.current_stage == STAGE_PARALLEL_PARKING:
            completed = self._check_completion_stage_parallel_parking()
        elif self.current_stage == STAGE_S_CURVE:
            completed = self._check_completion_stage_s_curve()
        elif self.current_stage == STAGE_TRAFFIC_LIGHT:
            completed = self._check_completion_stage_traffic_light()
        elif self.current_stage == STAGE_LEVEL_CROSSING:
            completed = self._check_completion_stage_level_crossing()
        elif self.current_stage == STAGE_NARROW_STRAIGHT:
            completed = self._check_completion_stage_narrow_straight()
        
        self.stage_goal_achieved = completed # Update status for info dict
        return completed


    def _get_vehicle_polygon(self, vehicle: Vehicle) -> np.ndarray:
        # Helper to get vehicle's corner coordinates in world frame
        x, y = vehicle.position
        heading = vehicle.heading
        l, w = vehicle.LENGTH, vehicle.WIDTH
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        
        # Relative corners (vehicle centered at origin, aligned with x-axis)
        corners_rel = np.array([
            [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2] # Order for pygame polygon (e.g. clockwise)
        ])
        
        # Rotate and translate
        rotated_corners = np.array([
            [c[0]*cos_h - c[1]*sin_h, c[0]*sin_h + c[1]*cos_h] for c in corners_rel
        ])
        abs_corners = rotated_corners + np.array([x, y])
        return abs_corners

    def _is_vehicle_in_polygon(self, vehicle: Vehicle, target_polygon_abs_corners: np.ndarray, margin: float = 0.0) -> bool:
        # Check if the vehicle is correctly parked within a target polygon (e.g., parking slot)
        # This version assumes target_polygon_abs_corners are the absolute world coordinates of the slot.
        # And uses the target_parking_slot_center and target_parking_slot_heading for alignment checks.

        if self.target_parking_slot_center is None or self.target_parking_slot_heading is None:
            # print("Debug: Target slot center or heading is None.")
            return False
        if not vehicle:
            # print("Debug: Vehicle is None.")
            return False

        # 1. Check heading alignment
        # For vertical parking, car can be forward or backward. For parallel, usually forward.
        heading_diff_direct = utils.wrap_to_pi(vehicle.heading - self.target_parking_slot_heading)
        
        angle_tolerance = np.deg2rad(15) # General tolerance
        if self.current_stage == STAGE_VERTICAL_PARKING:
            heading_diff_reversed = utils.wrap_to_pi(vehicle.heading - utils.wrap_to_pi(self.target_parking_slot_heading + np.pi))
            angle_tolerance = np.deg2rad(25) # Slightly more tolerance for vertical
            if not (abs(heading_diff_direct) < angle_tolerance or abs(heading_diff_reversed) < angle_tolerance):
                # print(f"Debug: Heading fail. V_H:{np.rad2deg(vehicle.heading):.1f} S_H:{np.rad2deg(self.target_parking_slot_heading):.1f} DiffD:{np.rad2deg(heading_diff_direct):.1f} DiffR:{np.rad2deg(heading_diff_reversed):.1f}")
                return False
        elif abs(heading_diff_direct) > angle_tolerance:
            # print(f"Debug: Heading fail (parallel). V_H:{np.rad2deg(vehicle.heading):.1f} S_H:{np.rad2deg(self.target_parking_slot_heading):.1f} DiffD:{np.rad2deg(heading_diff_direct):.1f}")
            return False

        # 2. Check if all vehicle corners are within the target slot's effective area
        # Transform vehicle corners to the slot's local coordinate system.
        # Slot's origin is self.target_parking_slot_center.
        # Slot's x-axis is aligned with self.target_parking_slot_heading.

        # Effective dimensions of the slot, reduced by margin
        # Assuming parking_slot_length is along the slot's local x-axis, and width along its local y-axis.
        slot_l_eff = self.config["parking_slot_length"] - 2 * margin
        slot_w_eff = self.config["parking_slot_width"] - 2 * margin
        if slot_l_eff < 0 or slot_w_eff < 0 : # Margin too large
            return False

        cos_slot_h_neg = np.cos(-self.target_parking_slot_heading)
        sin_slot_h_neg = np.sin(-self.target_parking_slot_heading)
        
        vehicle_abs_corners = self._get_vehicle_polygon(vehicle)

        for corner_world in vehicle_abs_corners:
            # Translate corner to slot's origin
            translated_corner_x = corner_world[0] - self.target_parking_slot_center[0]
            translated_corner_y = corner_world[1] - self.target_parking_slot_center[1]
            
            # Rotate corner into slot's frame
            corner_in_slot_frame_x = translated_corner_x * cos_slot_h_neg - translated_corner_y * sin_slot_h_neg
            corner_in_slot_frame_y = translated_corner_x * sin_slot_h_neg + translated_corner_y * cos_slot_h_neg
            
            # Check if the transformed corner is within the slot's effective AABB
            if not (abs(corner_in_slot_frame_x) <= slot_l_eff / 2 and \
                    abs(corner_in_slot_frame_y) <= slot_w_eff / 2):
                # print(f"Debug: Corner out. World:{corner_world} SlotFrame:({corner_in_slot_frame_x:.2f},{corner_in_slot_frame_y:.2f}) L/2:{slot_l_eff/2:.2f} W/2:{slot_w_eff/2:.2f}")
                return False
        
        # print("Debug: Vehicle considered in polygon.")
        return True

    # --- Stage 0: Straight Line ---
    def _setup_stage_straight_line(self):
        # Road: a 100m long straight lane
        self.road.network.add_lane("a", "b", StraightLane([0, 0], [100, 0], width=self.config["lane_width"]))
        # Ego vehicle: starts near the beginning of the lane
        self.ego_start_position = np.array([5.0, 0.0])
        self.ego_start_heading = 0.0 # Aligned with the lane
        # Goal: reach near the end of the lane
        self.current_goal_position = np.array([90.0, 0.0])
        self.other_vehicles_definitions = [] # No other vehicles in this stage

    def _reward_stage_straight_line(self) -> float:
        rew = 0.0
        if not self.vehicle or self.current_goal_position is None: return rew

        # Reward for reducing distance to goal
        dist_to_goal = np.linalg.norm(self.vehicle.position - self.current_goal_position)
        prev_dist_to_goal = getattr(self, "prev_dist_to_goal_straight", dist_to_goal) # Get previous or current if first step
        rew += (prev_dist_to_goal - dist_to_goal) * self.config["distance_reward_weight"]
        self.prev_dist_to_goal_straight = dist_to_goal

        # Penalty for lateral deviation from lane center
        current_lane_idx = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        if current_lane_idx:
            lane_obj = self.road.network.get_lane(current_lane_idx)
            _, lat_dist_abs = lane_obj.local_coordinates(self.vehicle.position)
            lat_dist_abs = abs(lat_dist_abs)
            rew -= lat_dist_abs * 0.1 # Small penalty for being off-center

        # Reward for maintaining a reasonable speed
        target_speed = 10.0 # m/s
        speed_error = abs(self.vehicle.speed - target_speed)
        rew -= speed_error * 0.05 # Penalty for deviating from target speed
        # speed_reward = utils.lmap(self.vehicle.speed, [0, 15], [0, 0.1]) # Original approach
        # rew += np.clip(speed_reward, 0, 0.1)
        return rew

    def _check_completion_stage_straight_line(self) -> bool:
        if not self.vehicle or self.current_goal_position is None: return False
        return np.linalg.norm(self.vehicle.position - self.current_goal_position) < 5.0 # Within 5m of goal

    # --- Stage 1: Vertical Parking (Backward) ---
    def _setup_stage_vertical_parking(self):
        # Main approach lane
        self.road.network.add_lane("s", "p", StraightLane([-30, 0], [0, 0], width=self.config["lane_width"]))
        # Parking area visualization (not a drivable lane for collision, but for context)
        # Using a wide "lane" to represent the general parking area.
        # Boundaries of this area are not physical barriers unless defined by other objects or map limits.
        self.road.network.add_lane("p_area_left", "p_area_right",
                                   StraightLane([0, -15], [30, -15], width=30,
                                                line_types=(LineType.NONE, LineType.NONE)))

        slot_w = self.config["parking_slot_width"]
        slot_l = self.config["parking_slot_length"] # Length of the slot (depth for vertical parking)
        
        # Slot is to the right of the approach lane, oriented downwards (for backward parking)
        self.target_parking_slot_center = np.array([10.0, -1.0 - slot_l/2]) # x=10, y is offset below the lane
        self.target_parking_slot_heading = -np.pi/2 # Slot pointing downwards (-90 degrees)

        # Define the polygon for checking parking (corners in world coordinates)
        # Slot's "length" (depth) is along its local Y axis if heading is 0. With -pi/2 heading, it's along world X.
        # Slot's "width" is along its local X axis if heading is 0. With -pi/2 heading, it's along world -Y.
        # Let's define corners relative to slot center, assuming slot's main axis is along its local X.
        h_l, h_w = slot_l / 2, slot_w / 2 # half-length, half-width
        # Relative corners if slot heading was 0 (slot length along X, width along Y)
        corners_rel_at_0_heading = np.array([
            [h_l, -h_w], [h_l, h_w], [-h_l, h_w], [-h_l, -h_w]
        ])
        cos_h, sin_h = np.cos(self.target_parking_slot_heading), np.sin(self.target_parking_slot_heading)
        rotated_corners = np.array([
            [c[0]*cos_h - c[1]*sin_h, c[0]*sin_h + c[1]*cos_h] for c in corners_rel_at_0_heading
        ])
        self.target_parking_slot_polygon = rotated_corners + self.target_parking_slot_center
        
        # Ego vehicle starts on the approach lane, before the parking area
        self.ego_start_position = np.array([-10.0, 0.0])
        self.ego_start_heading = 0.0 # Facing towards the parking area
        self.current_goal_position = self.target_parking_slot_center # For distance reward guidance

        # Add parked cars as obstacles
        car_len_dummy = Vehicle.LENGTH # Use default vehicle length for obstacles
        car_wid_dummy = Vehicle.WIDTH
        
        # Car to the "left" of the slot (from perspective of approaching car)
        # Slot width is slot_w. Obstacle car placed adjacent.
        # If slot heading is -pi/2, slot X is world Y, slot Y is world -X.
        # Obstacles are parallel to the slot.
        obstacle_offset_from_slot_edge = car_wid_dummy / 2 + 0.3 # distance from slot edge to center of obstacle car

        # Position of car1 (left of slot, if slot pointing down, this is more positive X)
        pos_car1_x = self.target_parking_slot_center[0] + slot_w/2 + obstacle_offset_from_slot_edge
        pos_car1_y = self.target_parking_slot_center[1]
        
        # Position of car2 (right of slot, if slot pointing down, this is more negative X)
        pos_car2_x = self.target_parking_slot_center[0] - slot_w/2 - obstacle_offset_from_slot_edge
        pos_car2_y = self.target_parking_slot_center[1]

        self.other_vehicles_definitions = [
             {"position": [pos_car1_x, pos_car1_y], "heading": self.target_parking_slot_heading, "color": (100,100,100)}, # Grey
             {"position": [pos_car2_x, pos_car2_y], "heading": self.target_parking_slot_heading, "color": (100,100,100)}, # Grey
        ]

    def _reward_stage_vertical_parking(self) -> float:
        rew = 0.0
        if not self.vehicle or self.target_parking_slot_center is None or self.target_parking_slot_heading is None:
            return rew

        # Distance to slot center
        dist_to_slot = np.linalg.norm(self.vehicle.position - self.target_parking_slot_center)
        prev_dist = getattr(self, "prev_dist_to_slot_vert", dist_to_slot)
        rew += (prev_dist - dist_to_slot) * self.config["distance_reward_weight"] * 0.2 # Scaled down for parking
        self.prev_dist_to_slot_vert = dist_to_slot

        # Heading error (penalize if not aligned with slot or its reverse)
        h_err_direct = abs(utils.wrap_to_pi(self.vehicle.heading - self.target_parking_slot_heading))
        h_err_reversed = abs(utils.wrap_to_pi(self.vehicle.heading - utils.wrap_to_pi(self.target_parking_slot_heading + np.pi)))
        heading_error = min(h_err_direct, h_err_reversed)
        rew -= heading_error * 0.3 # Penalty factor for heading misalignment

        # Penalty for high speed when close to slot
        if dist_to_slot < self.config["parking_slot_length"] * 1.5 and abs(self.vehicle.speed) > 2.0: # Speed > 2 m/s
            rew -= (abs(self.vehicle.speed) - 2.0) * 0.1

        # Bonus for being correctly parked (this is a shaping reward, completion gives more)
        if self._is_vehicle_in_polygon(self.vehicle, self.target_parking_slot_polygon, margin=self.config["parking_spot_margin"] * 1.5): # Wider margin for shaping
            rew += self.config["goal_achievement_reward"] * 0.1 # Small bonus for being roughly in place
        return rew

    def _check_completion_stage_vertical_parking(self) -> bool:
        if not self.vehicle or self.target_parking_slot_polygon is None: return False
        # Stricter check for completion: within polygon with defined margin, and very low speed
        is_parked = self._is_vehicle_in_polygon(self.vehicle, self.target_parking_slot_polygon, margin=self.config["parking_spot_margin"])
        is_stopped = abs(self.vehicle.speed) < 0.5 # m/s
        return is_parked and is_stopped

    # --- Stage 2: Parallel Parking (Backward) ---
    def _setup_stage_parallel_parking(self):
        # Main lane alongside which parking happens
        self.road.network.add_lane("s", "e", StraightLane([-30, 0], [30, 0], width=self.config["lane_width"]))
        
        slot_w_depth = self.config["parking_slot_width"]    # Depth of the slot (e.g., 2.5m)
        slot_l_along_curb = self.config["parking_slot_length"] # Length of the slot along the curb (e.g., 5m)

        # Slot is to the right of the lane, aligned with the lane (heading 0)
        # Center y is lane_center_y (0) - lane_half_width - slot_half_depth
        self.target_parking_slot_center = np.array([0.0, -(self.config["lane_width"]/2 + slot_w_depth/2)])
        self.target_parking_slot_heading = 0.0 # Slot aligned with the road's x-axis

        # Define polygon for checking parking
        h_l, h_w = slot_l_along_curb / 2, slot_w_depth / 2
        corners_rel = np.array([
            [h_l, -h_w], [h_l, h_w], [-h_l, h_w], [-h_l, -h_w]
        ]) # Assuming slot length (slot_l_along_curb) is along X, width (slot_w_depth) is along Y
        # No rotation needed as target_parking_slot_heading is 0
        self.target_parking_slot_polygon = corners_rel + self.target_parking_slot_center

        # Ego vehicle starts on the lane, ahead of the slot, positioned for reverse maneuver
        self.ego_start_position = np.array([slot_l_along_curb / 2 + Vehicle.LENGTH*0.75 , 0.0]) # Start slightly past the slot
        self.ego_start_heading = 0.0
        self.current_goal_position = self.target_parking_slot_center

        # Obstacle cars: one in front, one behind the slot
        car_len_dummy = Vehicle.LENGTH
        
        # Car in front of the slot (more negative X)
        pos_car_front_x = self.target_parking_slot_center[0] - slot_l_along_curb/2 - car_len_dummy/2 - 0.3 # 0.3m gap
        pos_car_front_y = self.target_parking_slot_center[1]
        
        # Car behind the slot (more positive X)
        pos_car_rear_x = self.target_parking_slot_center[0] + slot_l_along_curb/2 + car_len_dummy/2 + 0.3 # 0.3m gap
        pos_car_rear_y = self.target_parking_slot_center[1]

        self.other_vehicles_definitions = [
             {"position": [pos_car_front_x, pos_car_front_y], "heading": 0.0, "color": (100,100,100)},
             {"position": [pos_car_rear_x, pos_car_rear_y], "heading": 0.0, "color": (100,100,100)},
        ]

    def _reward_stage_parallel_parking(self) -> float:
        rew = 0.0
        if not self.vehicle or self.target_parking_slot_center is None or self.target_parking_slot_heading is None:
            return rew

        dist_to_slot = np.linalg.norm(self.vehicle.position - self.target_parking_slot_center)
        prev_dist = getattr(self, "prev_dist_to_slot_para", dist_to_slot)
        rew += (prev_dist - dist_to_slot) * self.config["distance_reward_weight"] * 0.2
        self.prev_dist_to_slot_para = dist_to_slot

        # Parallel parking expects vehicle heading to be similar to slot heading (or its reverse if allowed, but typically not for parallel)
        heading_error = abs(utils.wrap_to_pi(self.vehicle.heading - self.target_parking_slot_heading))
        # Could also allow reverse heading:
        # h_err_rev = abs(utils.wrap_to_pi(self.vehicle.heading - utils.wrap_to_pi(self.target_parking_slot_heading + np.pi)))
        # heading_error = min(heading_error, h_err_rev)
        rew -= heading_error * 0.3

        if dist_to_slot < self.config["parking_slot_length"] * 1.2 and abs(self.vehicle.speed) > 1.5: # Speed > 1.5 m/s
            rew -= (abs(self.vehicle.speed) - 1.5) * 0.1
        
        if self._is_vehicle_in_polygon(self.vehicle, self.target_parking_slot_polygon, margin=self.config["parking_spot_margin"] * 1.5):
            rew += self.config["goal_achievement_reward"] * 0.1
        return rew

    def _check_completion_stage_parallel_parking(self) -> bool:
        if not self.vehicle or self.target_parking_slot_polygon is None: return False
        is_parked = self._is_vehicle_in_polygon(self.vehicle, self.target_parking_slot_polygon, margin=self.config["parking_spot_margin"])
        is_stopped = abs(self.vehicle.speed) < 0.5 # m/s
        return is_parked and is_stopped

    # --- Stage 3: S-Curve (Forward and Backward) ---
    def _setup_stage_s_curve(self):
        R = self.config["s_curve_radius"]
        W = self.config["s_curve_lane_width"]
        
        # Define the S-curve path using a sequence of lanes
        net = RoadNetwork()
        # Initial straight segment
        net.add_lane("s0", "s1", StraightLane([0, 0], [R, 0], width=W))
        # First curve (e.g., counter-clockwise)
        # Center at [R, R], from 270 deg (south) to 360 deg (east)
        net.add_lane("s1", "c1_end", CircularLane([R, R], R, np.deg2rad(270), np.deg2rad(360), width=W, clockwise=False))
        # Second curve (e.g., clockwise), connects to end of first curve which is at [R,0] + [R,0] = [2R,0] (this is wrong)
        # End of first curve: position [R, R] + R*[cos(2pi), sin(2pi)] = [R,R] + [R,0] = [2R,R] (if angles are relative to +x from center)
        # highway_env CircularLane: start_angle from +x axis. Center [R,R]. End of s1 is [R,0]. Center of c1 is [R,R]. Start phase is -pi/2 or 270deg. End phase is 0 or 360deg.
        # End of c1_end is [R, R] + R*[cos(0), sin(0)] = [2R, R]
        # Center of c2_end should be [2R, R+R] = [2R, 2R] for a smooth transition if it curves back.
        # Or, if S-curve is like: --- CCW_UP --- CW_UP ---
        # c1_end position is [R, R] (center) + R*[cos(360deg), sin(360deg)] = [R,R] + [R,0] = [2R, R]
        # For c2_end (clockwise, radius R), starting from [2R,R]. If it curves "up and right", center could be e.g. [2R, 2R]
        # Let's use the user's original S-curve definition:
        #   net.add_lane("s1", "c1_end", CircularLane([R, R], R, np.deg2rad(270), np.deg2rad(360), width=W, clockwise=False))
        #   This ends at [R,R] + R*[cos(360), sin(360)] = [2R, R]. Heading is 0.
        #   net.add_lane("c1_end", "c2_end", CircularLane([R, 3*R], R, np.deg2rad(180), np.deg2rad(270), width=W, clockwise=True))
        #   This means c2 starts at [2R,R]. Its center is [R, 3R]. This will create a kink.
        #   To make it smooth, start of c2 must match end of c1.
        #   If c1_end is at [2R,R] heading 0. For c2 (CW) to start here, its center must be at [2R, R+R] = [2R,2R]. Start angle 270. End 180.
        # Let's simplify the S-curve slightly for guaranteed continuity based on common patterns.
        # Path: --- Straight --- Curve1 (CCW) --- StraightMiddle --- Curve2 (CW) --- StraightEnd
        # Or: --- Straight --- Curve1 (CCW up) --- Curve2 (CW up, connects smoothly) --- Straight ---
        # Using user's structure:
        # Lane s0-s1: (0,0) to (R,0)
        # Lane s1-c1_end: Center (R,R), R=R, StartAngle=270, EndAngle=360 (CCW). Ends at (2R,R), heading 0.
        # Lane c1_end-c2_end: Center (R,3R), R=R, StartAngle=180, EndAngle=270 (CW).
        #   Start point of this lane: (R,3R) + R*[cos(180), sin(180)] = (R,3R) + [-R,0] = [0,3R]. This doesn't connect to (2R,R).
        #
        # Let's make a canonical S-curve:
        # Start straight: [0,0] to [L_start, 0]
        # Curve 1 (CCW): center [L_start, R_curve], radius R_curve, from -pi/2 to pi/2. Ends at [L_start, 2*R_curve], heading pi/2.
        # Curve 2 (CW): center [L_start+2*R_curve, R_curve], radius R_curve, from pi/2 to -pi/2. Ends at [L_start+2*R_curve, 0], heading -pi/2.
        # This is complex. Let's use the original definition and assume it visually works for the task.
        # User's S-curve from original code:
        net.add_lane("s0_start", "s0_end", StraightLane([0, 0], [R, 0], width=W)) # Ends at [R,0]
        net.add_lane("s0_end", "c1_end", CircularLane([R, R], R, np.deg2rad(270), np.deg2rad(360), width=W, clockwise=False)) # Ends at [2R,R]
        # The second curve must start at [2R,R]. If center is [2R, 2R] and it's CW, radius R.
        # It would start at phase 270 (-pi/2) and go to e.g. 180 (pi).
        # User: CircularLane([R, 3*R], R, np.deg2rad(180), np.deg2rad(270), width=W, clockwise=True))
        # This starts at [0, 3R]. This S-curve definition is disjointed.
        # A common S-bend:
        # 1. Straight: (0,0) to (R,0)
        # 2. Arc1 (CCW): Center (R, R), Radius R, from -pi/2 to 0. Ends at (R+R, R) = (2R,R). Heading pi/2.
        # 3. Arc2 (CW): Center (2R, R+R) = (2R, 2R), Radius R, from pi/2 to 0. Ends at (2R+R, 2R) = (3R,2R). Heading 0.
        # 4. Straight: (3R,2R) to (4R,2R).
        # Let's assume the user's provided lane definitions are what they intend, even if geometrically complex.
        # Using the user's S-curve:
        # net.add_lane("s1", "c1_end", CircularLane([R, R], R, np.deg2rad(270), np.deg2rad(360), width=W, clockwise=False)) # End: [2R,R], H:0
        # net.add_lane("c1_end", "c2_end", CircularLane([R, 3*R], R, np.deg2rad(180), np.deg2rad(270), width=W, clockwise=True)) # Start: [0,3R], H:-pi/2
        # This will require the car to "jump" or navigate off-lane.
        # For a working environment, lanes must connect.
        # Re-defining a simpler, connected S-curve:
        # Initial straight
        net.add_lane("o", "a", StraightLane([0,0], [R,0], width=W)) # Ends at [R,0], H=0
        # First curve (90 deg up, CCW)
        net.add_lane("a", "b", CircularLane([R,R], R, start_angle=-np.pi/2, end_angle=0, width=W, clockwise=False)) # Ends at [2R,R], H=pi/2
        # Second curve (90 deg forward, CW, continuing from first curve's end direction)
        net.add_lane("b", "c", CircularLane([2*R,2*R], R, start_angle=np.pi, end_angle=np.pi/2, width=W, clockwise=True)) # Ends at [2R,3R], H=0 (Error in angles)
        # Corrected S-Curve:
        # Lane 1: Straight from (0,0) to (R,0). End heading 0.
        # Lane 2: CCW curve, center (R,R), radius R. From angle -pi/2 (point (R,0)) to angle pi/2 (point (R,2R)). End heading pi/2.
        # Lane 3: CW curve, center (R, 2R+R)=(R,3R), radius R. From angle pi/2 (point (R,2R)) to angle -pi/2 (point (R,4R)). End heading -pi/2. (This is a U-turn shape)

        # Let's use the original code's S-curve but force a continuous path for demonstration
        # The issue seems to be that highway-env lanes are identified by start/end node names.
        # The nodes provided ("s0", "s1", "c1_end", "c2_end", "e0") imply connectivity.
        # We trust RoadNetwork to handle the connections if node names match.
        net.add_lane("s0", "s1", StraightLane(start=np.array([0,0]), end=np.array([R,0]), width=W))
        # c1_end node is at [2R, R]
        net.add_lane("s1", "c1_end_node", CircularLane(center=np.array([R,R]), radius=R, start_angle=np.deg2rad(270), end_angle=np.deg2rad(360), width=W, clockwise=False, line_types=[LineType.STRIPED, LineType.STRIPED]))
        # c2_end node must be the end of the second curve. For the second curve to start from c1_end_node ([2R,R]), its geometry must match.
        # User's second curve: CircularLane([R, 3*R], R, np.deg2rad(180), np.deg2rad(270), width=W, clockwise=True)
        # This starts at [0,3R]. The provided node names "c1_end" and "c2_end" suggest they should connect.
        # If we assume the nodes connect, the exact geometry might be less critical than the agent staying on *some* path.
        # The RoadNetwork links lanes by matching the end node of one lane to the start node of the next.
        # So, c1_end_node from the first curve is the start for the second lane.
        # Let's use the user's provided lane structure, assuming nodes connect them:
        self.road.network = RoadNetwork.graph_to_network(
            {
                "s0": {"s1": StraightLane(start=np.array([0,0]), end=np.array([R,0]), width=W)},
                "s1": {"c1_end": CircularLane(center=np.array([R,R]), radius=R, start_angle=np.deg2rad(270), end_angle=np.deg2rad(360), width=W, clockwise=False)},
                "c1_end": {"c2_end": CircularLane(center=np.array([2*R,2*R]), radius=R, start_angle=np.deg2rad(270), end_angle=np.deg2rad(180), width=W, clockwise=True)}, # Adjusted to connect
                # Original user: CircularLane([R, 3*R], R, np.deg2rad(180), np.deg2rad(270), width=W, clockwise=True)
                # The adjusted one connects: Starts at (2R,R) (end of prev), ends at (R,2R).
                "c2_end": {"e0": StraightLane(start=np.array([R,2*R]), end=np.array([R, 2*R+10]), width=W)} # Adjusted to connect
                # Original user: StraightLane([2*R, 3*R], [2*R, 4*R + 5], width=W))
            }
        )
        # Find nodes in the created network for start/end points
        # This graph_to_network creates lanes named like "s0_s1", "s1_c1_end", etc.
        # It's simpler to add lanes sequentially if using custom node names for RoadNetwork.
        # Reverting to simpler add_lane, assuming user's original lanes are what they want to test with,
        # even if potentially disjointed (agent would need to go off-road).
        # For a robust env, ensure lanes connect. For now, using the user's specified structure:
        self.road.network.add_lane("s0", "s1", StraightLane([0, 0], [R, 0], width=W))
        self.road.network.add_lane("s1", "c1_end", CircularLane([R, R], R, np.deg2rad(270), np.deg2rad(360), width=W, clockwise=False)) # Ends at [2R,R]
        # The original next lane starts at [0,3R]. If we want connection:
        # A lane connecting [2R,R] to [0,3R] would be complex.
        # Let's assume the user's intent was that "c1_end" node links to the start of the next defined lane segment.
        # This means the RoadNetwork handles connecting lanes if node names match.
        # If "c1_end" is the END_NODE of the first circular lane, and START_NODE of the second, they link.
        # However, the geometry needs to be such that the second circular lane *starts* at [2R,R].
        # User's second circular lane: center [R, 3*R], angle [180, 270] CW. Starts at [R,3R]+R*[-1,0] = [0,3R].
        # This IS disjointed. Forcing connection by adjusting geometry:
        self.road.network.add_lane("c1_end", "c2_end", CircularLane(center=[2*R, 2*R], radius=R, start_angle=np.deg2rad(270), end_angle=np.deg2rad(180), width=W, clockwise=True)) # Starts at [2R,R], ends at [R,2R]
        self.road.network.add_lane("c2_end", "e0", StraightLane(start=[R, 2*R], end=[R, 2*R + 10], width=W)) # Final straight. End point: [R, 2*R+10]
        
        self.ego_start_position = np.array([R/2, 0.0]) # Start on the first straight
        self.ego_start_heading = 0.0
        self.s_curve_forward_done = False # Start with forward part
        
        # Goal for forward: end of the S-curve path
        self.current_goal_position_forward = np.array([R, 2*R + 7.5]) # Near end of "e0"
        # Goal for backward: start of the S-curve path
        self.current_goal_position_backward = np.array([R/2, 0.0]) # Back to start
        
        self.current_goal_position = self.current_goal_position_forward # Initial goal
        self.other_vehicles_definitions = []


    def _reward_stage_s_curve(self) -> float:
        rew = 0.0
        if not self.vehicle: return rew

        target_pos = self.current_goal_position_forward if not self.s_curve_forward_done else self.current_goal_position_backward
        if target_pos is None: return rew # Should not happen if setup is correct

        dist_to_target = np.linalg.norm(self.vehicle.position - target_pos)
        prev_dist_attr = "prev_dist_s_fwd" if not self.s_curve_forward_done else "prev_dist_s_bwd"
        prev_dist = getattr(self, prev_dist_attr, dist_to_target)
        rew += (prev_dist - dist_to_target) * self.config["distance_reward_weight"]
        setattr(self, prev_dist_attr, dist_to_target)

        # Penalty for lateral deviation and heading error relative to the current lane segment
        current_lane_idx = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        if current_lane_idx:
            lane_obj = self.road.network.get_lane(current_lane_idx)
            longi, lat_dist_signed = lane_obj.local_coordinates(self.vehicle.position)
            rew -= abs(lat_dist_signed) * 0.2 # Penalty for lateral distance

            lane_heading_at_pos = lane_obj.heading_at(longi)
            # Target heading: lane heading if forward, opposite if backward
            target_vehicle_heading = lane_heading_at_pos
            if self.s_curve_forward_done: # If reversing
                target_vehicle_heading = utils.wrap_to_pi(lane_heading_at_pos + np.pi)
            
            heading_error = abs(utils.wrap_to_pi(self.vehicle.heading - target_vehicle_heading))
            rew -= heading_error * 0.1 # Penalty for heading error

        # Speed penalty: different limits for forward and backward
        max_speed = 8.0 if not self.s_curve_forward_done else 4.0 # m/s
        # Note: vehicle.speed is absolute. If reversing, it's positive. Controller handles direction.
        if self.vehicle.speed > max_speed:
            rew -= (self.vehicle.speed - max_speed) * 0.1
        
        return rew

    def _check_completion_stage_s_curve(self) -> bool:
        if not self.vehicle: return False

        if not self.s_curve_forward_done:
            # Check completion of forward part
            if np.linalg.norm(self.vehicle.position - self.current_goal_position_forward) < 3.0 and \
               self.vehicle.speed < 2.0: # Low speed near target
                self.s_curve_forward_done = True
                print("S-Curve: Forward part completed. Prepare to reverse.")
                self.current_goal_position = self.current_goal_position_backward # Set new goal for reversing
                # Reset distance tracking for backward part
                if hasattr(self, "prev_dist_s_fwd"): delattr(self, "prev_dist_s_fwd")
                # Do NOT return True yet; stage continues with backward part
                return False
            return False # Forward part not yet done
        else:
            # Check completion of backward part
            if np.linalg.norm(self.vehicle.position - self.current_goal_position_backward) < 3.0 and \
               abs(self.vehicle.speed) < 1.0: # Very low speed near origin
                print("S-Curve: Backward part completed. Stage S-Curve finished.")
                return True # Entire S-curve stage is done
            return False # Backward part not yet done

    # --- Stage 4: Traffic Light ---
    def _setup_stage_traffic_light(self):
        self.traffic_light_line_x = self.config["traffic_light_distance"] # x-coord of the stop line
        # Road: straight lane approaching and passing the traffic light
        self.road.network.add_lane("s", "tl_line_node", StraightLane([0, 0], [self.traffic_light_line_x, 0], width=self.config["lane_width"]))
        self.road.network.add_lane("tl_line_node", "e", StraightLane([self.traffic_light_line_x, 0], [self.traffic_light_line_x + 50, 0], width=self.config["lane_width"]))
        
        self.ego_start_position = np.array([5.0, 0.0]) # Start before the light
        self.ego_start_heading = 0.0
        self.current_goal_position = np.array([self.traffic_light_line_x + 40.0, 0.0]) # Goal well past the light
        
        # Initialize traffic light state
        self.traffic_light_state = LIGHT_RED # Start with red light
        self.traffic_light_timer = 0
        self.other_vehicles_definitions = []

    def _reward_stage_traffic_light(self) -> float:
        rew = 0.0
        if not self.vehicle or self.current_goal_position is None: return rew
        
        vehicle_x = self.vehicle.position[0]
        # Define stopping zone before the light (e.g., 1m to 10m before the line)
        stopping_zone_start = self.traffic_light_line_x - 10.0
        stopping_zone_end = self.traffic_light_line_x - 1.0 # Vehicle front should not pass this on red

        if self.traffic_light_state == LIGHT_RED:
            # Penalize crossing red light: if front of vehicle is past line and moving
            if vehicle_x + self.vehicle.LENGTH/2 > self.traffic_light_line_x and self.vehicle.speed > 0.5:
                rew -= 5.0 # Significant penalty for running red light
            # Reward for stopping correctly in the zone
            elif stopping_zone_start < vehicle_x < stopping_zone_end and self.vehicle.speed < 1.0:
                rew += 0.5 # Reward for waiting appropriately
        
        elif self.traffic_light_state == LIGHT_GREEN:
            # Reward for proceeding on green
            if vehicle_x > self.traffic_light_line_x and vehicle_x < self.traffic_light_line_x + 15 and self.vehicle.speed > 1.0:
                rew += 1.0
            # Penalize stopping or being too slow on green (if past initial hesitation)
            elif vehicle_x < self.traffic_light_line_x and self.vehicle.speed < 0.5 and self.traffic_light_timer > 2 * self.config["policy_frequency"]: # Timer check to allow reaction time
                rew -= 0.5 # Penalty for hesitation or stopping on green

        # General reward for progressing towards goal
        dist_to_goal = np.linalg.norm(self.vehicle.position - self.current_goal_position)
        prev_dist = getattr(self, "prev_dist_tl", dist_to_goal)
        rew += (prev_dist - dist_to_goal) * self.config["distance_reward_weight"] * 0.2 # Smaller weight for this component
        self.prev_dist_tl = dist_to_goal
        return rew

    def _check_completion_stage_traffic_light(self) -> bool:
        if not self.vehicle: return False
        # Stage completed if vehicle has passed well beyond the traffic light line
        # And did not run a red light (implicit: if red light violation happens, it's a penalty, not stage end unless crashed)
        # Consider if red light violation should be a "failure" for the stage. For now, it's a penalty.
        return self.vehicle.position[0] > self.traffic_light_line_x + 15.0

    # --- Stage 5: Level Crossing ---
    def _setup_stage_level_crossing(self):
        self.level_crossing_line_x = self.config["traffic_light_distance"] # Re-use config, or define separate
        self.road.network.add_lane("s", "lc_line_node", StraightLane([0, 0], [self.level_crossing_line_x, 0], width=self.config["lane_width"]))
        self.road.network.add_lane("lc_line_node", "e", StraightLane([self.level_crossing_line_x, 0], [self.level_crossing_line_x + 50, 0], width=self.config["lane_width"]))
        
        self.ego_start_position = np.array([5.0, 0.0])
        self.ego_start_heading = 0.0
        self.current_goal_position = np.array([self.level_crossing_line_x + 40.0, 0.0])
        
        self.level_crossing_state = CROSSING_CLOSED # Start with crossing closed
        self.level_crossing_timer = 0
        self.other_vehicles_definitions = []

    def _reward_stage_level_crossing(self) -> float:
        rew = 0.0
        if not self.vehicle or self.current_goal_position is None: return rew

        vehicle_x = self.vehicle.position[0]
        stopping_zone_start = self.level_crossing_line_x - 10.0
        stopping_zone_end = self.level_crossing_line_x - 1.0

        if self.level_crossing_state == CROSSING_CLOSED:
            if vehicle_x + self.vehicle.LENGTH/2 > self.level_crossing_line_x and self.vehicle.speed > 0.5:
                rew -= 5.0 # Penalty for crossing when closed
            elif stopping_zone_start < vehicle_x < stopping_zone_end and self.vehicle.speed < 1.0:
                rew += 0.5 # Reward for waiting
        
        elif self.level_crossing_state == CROSSING_OPEN:
            if vehicle_x > self.level_crossing_line_x and vehicle_x < self.level_crossing_line_x + 15 and self.vehicle.speed > 1.0:
                rew += 1.0 # Reward for proceeding when open
            elif vehicle_x < self.level_crossing_line_x and self.vehicle.speed < 0.5 and self.level_crossing_timer > 2 * self.config["policy_frequency"]:
                rew -= 0.5 # Penalty for undue hesitation
        
        dist_to_goal = np.linalg.norm(self.vehicle.position - self.current_goal_position)
        prev_dist = getattr(self, "prev_dist_lc", dist_to_goal)
        rew += (prev_dist - dist_to_goal) * self.config["distance_reward_weight"] * 0.2
        self.prev_dist_lc = dist_to_goal
        return rew

    def _check_completion_stage_level_crossing(self) -> bool:
        if not self.vehicle: return False
        return self.vehicle.position[0] > self.level_crossing_line_x + 15.0

    # --- Stage 6: Narrow Straight Line ---
    def _setup_stage_narrow_straight(self):
        narrow_width = self.config["narrow_lane_width"]
        lane_entry_len = 20.0
        narrow_segment_len = 50.0
        lane_exit_len = 20.0

        start_narrow_x = lane_entry_len
        end_narrow_x = lane_entry_len + narrow_segment_len
        end_exit_x = end_narrow_x + lane_exit_len

        # Entry lane (normal width)
        self.road.network.add_lane("entry_start", "entry_end", StraightLane([0,0], [start_narrow_x,0], width=self.config["lane_width"]))
        # Narrow segment (continuous lines indicating no overtaking and narrowness)
        self.road.network.add_lane("entry_end", "narrow_exit", StraightLane([start_narrow_x, 0], [end_narrow_x, 0], width=narrow_width, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)))
        # Exit lane (normal width)
        self.road.network.add_lane("narrow_exit", "final_exit", StraightLane([end_narrow_x,0], [end_exit_x,0], width=self.config["lane_width"]))
        
        self.ego_start_position = np.array([5.0, 0.0]) # Start on the entry lane
        self.ego_start_heading = 0.0
        self.current_goal_position = np.array([end_exit_x - 5.0, 0.0]) # Goal near the end of the exit lane
        self.other_vehicles_definitions = []

    def _reward_stage_narrow_straight(self) -> float:
        rew = 0.0
        if not self.vehicle or self.current_goal_position is None: return rew
        
        on_narrow_segment = False
        # Check if vehicle is on the narrow lane segment "entry_end" -> "narrow_exit"
        # This requires knowing the exact lane ID if multiple lanes exist.
        # A simpler check is by x-coordinate, assuming single path.
        vehicle_x = self.vehicle.position[0]
        start_narrow_x = 20.0 # Must match setup
        end_narrow_x = 20.0 + 50.0 # Must match setup

        if start_narrow_x <= vehicle_x < end_narrow_x:
            on_narrow_segment = True
            # Get the narrow lane object to check lateral distance
            # This assumes the narrow lane is uniquely identifiable or is the current closest
            current_lane_idx = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading, lane_type_filter=[("entry_end","narrow_exit")])
            # A robust way would be to get the lane object directly if its ID is fixed, e.g., self.road.network.get_lane(("entry_end", "narrow_exit", 0))
            # For now, using closest lane and checking its width or explicit ID
            lane_obj = None
            if current_lane_idx and current_lane_idx[0] == "entry_end" and current_lane_idx[1] == "narrow_exit":
                 lane_obj = self.road.network.get_lane(current_lane_idx)

            if lane_obj:
                 _, lat_dist_signed = lane_obj.local_coordinates(self.vehicle.position)
                 lat_dist_abs = abs(lat_dist_signed)
                 
                 # Max safe lateral distance from center before edge of vehicle hits edge of lane
                 max_safe_lat_dist = (self.config["narrow_lane_width"] / 2) - (self.vehicle.WIDTH / 2)
                 if max_safe_lat_dist < 0.05: max_safe_lat_dist = 0.05 # Avoid division by zero, ensure some small safe zone

                 if lat_dist_abs > max_safe_lat_dist:
                     rew -= 2.0 # Penalty for being too close to edge / hitting edge
                 else:
                     # Quadratic penalty for deviation within safe zone
                     rew -= (lat_dist_abs / max_safe_lat_dist)**2 * 0.5
            else: # If not on the specific narrow lane (e.g., drifted to an adjacent one if any)
                if on_narrow_segment : rew -= 1.0 # Penalty for being in narrow x-range but not on the narrow lane

        # Target speed in narrow section
        speed_target = 6.0 # m/s
        speed_error = abs(self.vehicle.speed - speed_target)
        rew -= speed_error * 0.05 # Penalty for deviating from target speed
        
        # Penalize being too slow/stopped in narrow passage after initial entry
        if on_narrow_segment and self.vehicle.speed < 1.0 and self.stage_step_count > 1 * self.config["policy_frequency"]:
            rew -= 0.5

        # Progress reward
        dist_to_goal = np.linalg.norm(self.vehicle.position - self.current_goal_position)
        prev_dist = getattr(self, "prev_dist_narrow", dist_to_goal)
        rew += (prev_dist - dist_to_goal) * self.config["distance_reward_weight"]
        self.prev_dist_narrow = dist_to_goal
        return rew

    def _check_completion_stage_narrow_straight(self) -> bool:
        if not self.vehicle or self.current_goal_position is None: return False
        # Completed if near goal and moving (not stopped before it)
        return np.linalg.norm(self.vehicle.position - self.current_goal_position) < 5.0 and self.vehicle.speed > 0.5

    # --- Rendering ---
    def _render(self, mode: str) -> Optional[np.ndarray]:
        # This calls AbstractEnv._render to handle standard rendering setup
        super()._render(mode)

        if mode == 'human' and self.viewer is not None and self.viewer.sim_surface is not None:
            surface = self.viewer.sim_surface # This is the Pygame surface for world objects

            # Draw parking slot polygon if defined for the current stage
            if (self.current_stage == STAGE_VERTICAL_PARKING or self.current_stage == STAGE_PARALLEL_PARKING) and \
               self.target_parking_slot_polygon is not None:
                try:
                    # Convert world coordinates to pixel coordinates
                    pixel_slot_polygon = [surface.vec2pix(point) for point in self.target_parking_slot_polygon]
                    pygame.draw.polygon(surface, (0, 200, 0, 150), pixel_slot_polygon, 3) # Green, slightly transparent, 3px thick
                except Exception as e: print(f"Error rendering parking slot: {e}")


            # Draw traffic light visual cue
            if self.current_stage == STAGE_TRAFFIC_LIGHT:
                try:
                    # Position light slightly above and to the side of the stop line
                    light_world_pos = np.array([self.traffic_light_line_x, self.config["lane_width"] * 0.65]) # y-offset for visibility
                    light_pix_pos = surface.vec2pix(light_world_pos)
                    light_pix_radius = int(max(1, surface.scaling * 0.5)) # Radius in pixels
                    
                    color_map = {LIGHT_RED: (255,0,0), LIGHT_GREEN: (0,255,0), LIGHT_YELLOW: (255,255,0)}
                    light_color_rgb = color_map.get(self.traffic_light_state, (50,50,50)) # Default to grey if state unknown
                    pygame.draw.circle(surface, light_color_rgb, light_pix_pos, light_pix_radius)
                except Exception as e: print(f"Error rendering traffic light: {e}")

            # Draw level crossing bar visual cue
            if self.current_stage == STAGE_LEVEL_CROSSING:
                try:
                    if self.level_crossing_state == CROSSING_CLOSED:
                        # Bar across the lane at self.level_crossing_line_x
                        p1_world = np.array([self.level_crossing_line_x, -self.config["lane_width"]/2 * 1.1]) # Extend slightly beyond lane
                        p2_world = np.array([self.level_crossing_line_x, self.config["lane_width"]/2 * 1.1])
                        p1_pix = surface.vec2pix(p1_world)
                        p2_pix = surface.vec2pix(p2_world)
                        bar_thickness = int(max(2, surface.scaling * 0.2)) # Min 2px thick
                        pygame.draw.line(surface, (200,0,0), p1_pix, p2_pix, bar_thickness)
                except Exception as e: print(f"Error rendering level crossing: {e}")

            # Draw current goal position as a circle
            if self.current_goal_position is not None:
                try:
                    goal_pix_pos = surface.vec2pix(self.current_goal_position)
                    goal_pix_radius = int(max(2, surface.scaling * 0.6)) # Min 2px radius
                    pygame.draw.circle(surface, (0,0,255, 180), goal_pix_pos, goal_pix_radius, 2) # Blue, slightly transparent, 2px border
                except Exception as e: print(f"Error rendering goal position: {e}")
            
            # Display current stage text on screen (using self.viewer.screen which is the main display)
            try:
                font = pygame.font.Font(None, 30) # Default Pygame font, size 30
                text_surface = font.render(f"Stage: {self.current_stage}", True, (20, 20, 20)) # Black text
                if hasattr(self.viewer, 'screen'): # Ensure screen attribute exists
                     self.viewer.screen.blit(text_surface, (10, 10)) # Draw at top-left
            except Exception as e: print(f"Error rendering stage text: {e}")

        # For 'rgb_array' mode, AbstractEnv._render already calls viewer.get_image() if viewer exists.
        # If we needed to return it from here:
        if mode == 'rgb_array' and self.viewer is not None and hasattr(self.viewer, 'get_image'):
            return self.viewer.get_image()
        return None # For 'human' mode, or if no specific rgb_array handling here

    def close(self):
        super().close() # Important to call super().close() for proper cleanup by AbstractEnv/Viewer


# Register the environment with Gymnasium
# Using a namespaced ID as recommended by Gymnasium, e.g., MyOrg/MyEnv-v0
# For this example, using a simple descriptive ID.
ENV_ID = 'DrivingSchoolEnv-v0'
try:
    gym.register(
        id=ENV_ID,
        entry_point='driving_school_environment:DrivingSchoolEnv', # Assumes the file is run directly as __main__
                                                 # If file is my_env_file.py, use 'my_env_file:DrivingSchoolEnv'
    )
    print(f"Environment '{ENV_ID}' registered successfully.")
except gym.error.Error as e: # More specific exception if known, e.g., gym.error.NameAlreadyRegistered
    if "Cannot re-register id" in str(e) or "already registered" in str(e).lower():
        print(f"Environment '{ENV_ID}' already registered. Skipping.")
    else:
        print(f"Warning: Error registering environment '{ENV_ID}': {e}")