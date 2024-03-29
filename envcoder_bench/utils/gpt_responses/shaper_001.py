
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Normalizing forward velocity to reward faster speeds more
    forward_velocity = state_dict['root_states'][:, 3]  # Assuming the 4th element is forward velocity
    speed_reward = torch.exp(forward_velocity - 1.0)  # Subtract a base value to scale and use exp for normalization

    # Temperature parameter for the velocity reward normalization
    velocity_temp = 0.1
    normalized_speed_reward = torch.exp((forward_velocity - 1.0) / velocity_temp)

    # Combine rewards (here only forward speed is considered)
    reward = normalized_speed_reward
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming action directly corresponds to torques with some scaling factor
    torque_scaling_factor = 1.0  # This can be adjusted based on the specific needs
    torque = action * torque_scaling_factor
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Concatenating selected state information as observation
    # For a running task, we might consider root state (position, orientation), dof positions, and velocities
    observation = torch.cat([state_dict['root_states'], state_dict['dof_pos'], state_dict['dof_vel']], dim=-1)
    return observation

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # For simplicity, let's end the episode if the agent falls over (z position < threshold)
    z_position = state_dict['root_states'][:, 2]  # Assuming the 3rd element is the vertical (z) position
    done_threshold = 0.2  # Threshold below which we consider the robot has fallen
    done = torch.tensor(z_position < done_threshold, dtype=torch.float32, device=z_position.device)

    # Another condition could be exceeding the maximum episode length
    progress = state_dict['progress_buf']
    max_length = state_dict['max_episode_length']
    done_max_episode = torch.tensor(progress >= max_length, dtype=torch.float32, device=progress.device)

    # Combine done signals
    done = torch.max(done, done_max_episode)
    return done


