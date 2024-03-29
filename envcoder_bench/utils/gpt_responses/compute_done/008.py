import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming episode ends either when it exceeds max episode length or when a terminal condition is met (e.g., robot falls)
    progress_buf = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length']
    done = progress_buf >= max_episode_length
    
    return done



