import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Task completion condition could be based on progress or time-out
    progress = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length']
    done = torch.ge(progress, max_episode_length)  # Returns a tensor of bools
    return done



