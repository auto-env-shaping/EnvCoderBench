
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Assume running as fast as possible is measured by the forward velocity along the x-axis of the root
    forward_velocity = state_dict['root_states'][:, 7]  # Assuming root_states [pos(3), quat(4), vel(3), ang_vel(3)]
    velocity_reward = torch.exp(0.1 * forward_velocity)  # Temperature parameter for scaling

    # Penalize large actions to encourage energy efficiency
    action_cost = torch.sum(action**2, dim=1)
    action_penalty = torch.exp(0.01 * action_cost)  # Temperature parameter for action cost

    # Combine the rewards and penalties
    reward = velocity_reward - action_penalty
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Here we can simply apply a scaling factor to the raw action outputs to represent torques,
    # as a simplistic model for this example
    # Consider the action scaling factor to map action outputs to actual torque values based on the robot's specifications
    action_scaling_factor = 1.0  # Assuming actions are already normalized
    torque = action * action_scaling_factor
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example observation concatenates position, velocity of DOFs (joints) and root velocity
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    root_velocity = state_dict['root_states'][:, 7:10]  # Extracting linear velocity part
    observation = torch.cat([dof_pos, dof_vel, root_velocity], dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example terminal condition based on maximum episode length or if the robot has fallen (e.g., z-position < threshold)
    max_episode_length = state_dict['max_episode_length']
    progress_buf = state_dict['progress_buf']
    z_pos = state_dict['root_states'][:, 2]  # Assuming the third component is the z-position
    fall_threshold = 0.3  # arbitrary threshold below which we consider the robot has fallen
    
    done_by_fall = z_pos < fall_threshold
    done_by_length = progress_buf >= max_episode_length
    
    done = torch.logical_or(done_by_fall, done_by_length).int()  # 1 if done, 0 otherwise
    return done
