import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming the task ends when the agent falls over or an episode reaches a maximum length
    progress_buf = state_dict['progress_buf']  # Episode progress
    max_episode_length = state_dict['max_episode_length']  # Max length of an episode
    root_states = state_dict['root_states']  # [num_envs, 13]
    
    # Check if the robot has fallen over, for simplicity, let's assume if the z (vertical) component of the position is below a threshold, it's considered fallen
    fallen = root_states[:, 2] < 0.5  # Threshold for being considered fallen
    
    # Check if the episode has reached its max length
    max_length_reached = progress_buf >= max_episode_length
    
    done = fallen | max_length_reached
    return done



