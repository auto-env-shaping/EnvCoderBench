
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Parameters for reward shaping
    velocity_reward_weight = 1.0  # Encourage forward movement
    action_penalty_weight = 0.01  # Penalize large actions to encourage energy efficiency
    temperature_velocity = 0.1  # Temperature parameter for velocity reward transformation
    temperature_action = 0.01  # Temperature parameter for action cost transformation

    # Extract relevant state information
    root_velocity = state_dict["root_states"][:, 7:10]  # [num_envs, 3] (x, y, z velocities)
    forward_velocity = root_velocity[:, 0]  # We are only interested in the x (forward) velocity
    
    # Reward for moving forward, transformed using an exponential function to normalize it
    velocity_reward = torch.exp(velocity_reward_weight * forward_velocity / temperature_velocity)
    
    # Penalize large actions to ensure smooth and efficient movements
    action_cost = torch.sum(action**2, dim=1)  # Sum of squared actions
    action_penalty = torch.exp(action_penalty_weight * action_cost / temperature_action)
    
    # Compute total reward
    reward = velocity_reward - action_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # No transformation needed; action directly corresponds to torque
    # However, you might scale it or apply limits in a more complex scenario
    torque = action
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Simple observation model concatenating position and velocity
    dof_pos = state_dict["dof_pos"]  # Shape: [num_envs, 12]
    dof_vel = state_dict["dof_vel"]  # Shape: [num_envs, 12]
    
    # Normalize dof_pos and dof_vel based on provided scales
    normalized_dof_pos = dof_pos / state_dict["dof_pos_scale"]
    normalized_dof_vel = dof_vel / state_dict["dof_vel_scale"]
    
    # Optionally include root velocity and orientation for a richer observation
    root_states = state_dict["root_states"][:, 0:7]  # shape: [num_envs, 7] (x, y, z, quat)
    root_velocity = state_dict["root_states"][:, 7:10]  # shape: [num_envs, 3] (x, y, z velocities)
    
    observation = torch.cat([normalized_dof_pos, normalized_dof_vel, root_states, root_velocity], dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example termination condition - episode length
    max_episode_length = state_dict["max_episode_length"]
    current_length = state_dict["progress_buf"]
    
    # Determine if the episode should be terminated based on length
    done = current_length >= max_episode_length
    return done
