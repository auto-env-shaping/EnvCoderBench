
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # The main goal is to reward the agent for running as fast as possible.
    # We assume running fast in a positive x direction which is encoded in commands.
    # Normalized difference between desired velocity and actual velocity to encourage speed.
    target_speed = state_dict["commands"][:, 0]  # Assume the x component is forward speed.
    actual_speed = state_dict["root_states"][:, 7]  # Assume index 7 is the actual forward speed in root_states.
    
    speed_error = target_speed - actual_speed
    # Speed reward: applying exponential transformation to make it sensitive to changes
    temperature_speed = 0.1
    speed_reward = torch.exp(-temperature_speed * speed_error.abs())  # Encouraging matching the target speed
    
    # Penalize excessive energy use (assuming actions correlate with energy expenditure)
    energy_penalty = action.pow(2).sum(dim=1)
    temperature_energy = 0.01
    energy_reward = torch.exp(-temperature_energy * energy_penalty)  # Penalizing high energy (action magnitude)
    
    # Combine the two components into a total reward
    reward = speed_reward + energy_reward
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # In this scenario, the action directly corresponds to the torque applied to each joint.
    # However, we might want to scale the raw action values to ensure they're within a realistic range.
    torque_scaling_factor = 5.0  # This value might need tuning based on the dynamics of the environment.
    torque = action * torque_scaling_factor
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Considerations for observation for the task:
    # 1. Root states to understand its positioning and orientation
    # 2. DOF (Degrees of Freedom) positions and velocities to know the current motion state
    # 3. Commands to understand the current target/expected movement
    observation_pieces = [state_dict["root_states"], state_dict["dof_pos"], state_dict["dof_vel"], state_dict["commands"]]
    observation = torch.cat(observation_pieces, dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # A simple check for episode completion can be based on the number of steps taken so far.
    # Alternatively, we could end episodes when the agent falls over or leaves a defined area.
    progress = state_dict["progress_buf"]
    max_episode_length = state_dict["max_episode_length"]
    done = (progress >= max_episode_length).float()
    return done
