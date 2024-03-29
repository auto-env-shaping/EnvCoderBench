
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants for reward scaling
    speed_temp = 0.1  # Temperature for the speed reward component
    energy_temp = 0.01  # Temperature for the energy use (motor torques) component
    progress_temp = 0.05  # Temperature for the progress towards max episode length
    
    # Extracting necessary components from state dictionary
    root_states = state_dict['root_states']  # Shape: [num_envs, 13]
    dof_vel = state_dict['dof_vel']  # Shape: [num_envs, 12]
    torques = state_dict['torques']  # Shape: [num_envs, 12]
    progress_buf = state_dict['progress_buf'].type(torch.float32)  # Convert to float for computation
    max_episode_length = state_dict['max_episode_length'].type(torch.float32)  # Convert to float
    
    # Calculate forward speed (assuming forward direction is aligned with one of the axes in root_states)
    # Forward speed is the third element of velocity in root_states which is represented by index [2]
    forward_speed = root_states[:, 2]
    
    # Calculate energy expenditure (squared sum of torques)
    energy_expenditure = torch.sum(torques**2, dim=1)
    
    # Progress ratio towards the end of an episode
    progress_ratio = progress_buf / max_episode_length
    
    # Reward for speed, penalize for energy use and incentivize episode progress
    speed_reward = torch.exp(speed_temp * forward_speed) - 1
    energy_penalty = -torch.exp(energy_temp * energy_expenditure)
    progress_reward = torch.exp(progress_temp * progress_ratio) - 1
    
    # Final reward is a weighted sum of the individual components
    reward = speed_reward + energy_penalty + progress_reward
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly using actions as torques for simplicity, assuming action space is already mapped to desired torque
    torque = action
    return torque
