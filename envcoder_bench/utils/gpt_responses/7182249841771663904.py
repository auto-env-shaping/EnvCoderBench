
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants
    speed_temp = 0.1  # Temperature parameter for speed reward component
    energy_temp = 0.01  # Temperature parameter for energy penalty component
    progress_temp = 0.05  # Temperature parameter for progress reward component
    
    root_velocities = state_dict['root_states'][:, 7:10]  # Extract root linear velocities (x, y, z)
    forward_speed = root_velocities[:, 0]  # We're interested in the forward (x) direction speed
    
    # Reward for moving forward, penalize slow speed
    speed_reward = torch.exp(forward_speed * speed_temp)
    
    # Penalty for using too much energy (squared sum of action values)
    energy_penalty = torch.sum(action**2, dim=1) * energy_temp
    
    # Progress reward (assuming progress is measured by moving forward)
    progress = state_dict['progress_buf'].float()
    progress_reward = torch.exp(progress * progress_temp)
    
    # Combine rewards and penalties
    reward = speed_reward - energy_penalty + progress_reward
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # In this task, we'll assume the actions directly map to torques with some scaling factors
    torque_scaling = 0.1  # Scale to make sure torques are within a reasonable range
    
    torque = action * torque_scaling
    
    return torque
