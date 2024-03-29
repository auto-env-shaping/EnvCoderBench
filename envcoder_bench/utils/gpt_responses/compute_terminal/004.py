import torch
from typing import * 

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Terminal condition can be reaching the max episode length or falling (detected by abnormal root state values)
    progress_buf = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length'].item()
    root_states = state_dict['root_states']
    base_height = root_states[:, 2]  # Assuming Z-axis is the vertical axis
    
    done_by_time = progress_buf >= max_episode_length
    done_by_fall = base_height < 0.2  # Assuming the robot is considered fallen if the base height is less than 0.2 meters
    
    done = done_by_time | done_by_fall
    return done.float()



