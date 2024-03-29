
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming the x-direction velocity (root_states[:, 3]) is what we want to maximize
    velocity_x = state_dict['root_states'][:, 3]
    # Transform the velocity using an exponential function to emphasize higher speeds and include a temperature parameter
    temperature = 0.1  # Adjustables parameter to scale the reward function
    reward = torch.exp(velocity_x / temperature)
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming the 'actions' provided by the policy are scaled appropriately for the environment
    # Directly using action as torque without further processing
    torque = action  # action tensor already on the same device, can be directly used
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Combining relevant parts of the state for observation
    # For a running task, we might be interested in the root state, dof positions, and velocities
    # So, concatenate these tensors along the second axis (column-wise concatenation)
    root_states = state_dict['root_states']
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    observation = torch.cat([root_states, dof_pos, dof_vel], dim=1)
    return observation

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # In a task focused on running speed, we might terminate an episode if the agent falls over.
    # Assuming that root_states[:, 2] represents the height of the robot's base from the ground
    # and setting a threshold below which we consider the robot to have fallen
    base_height = state_dict['root_states'][:, 2]
    fall_threshold = 0.2  # Threshold height; below this, we consider the robot to have fallen
    done = torch.lt(base_height, fall_threshold)
    return done


