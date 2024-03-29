import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Compute the done signal based on current progress and maximum episode length
    progress = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length']
    
    done = progress >= max_episode_length
    return done



