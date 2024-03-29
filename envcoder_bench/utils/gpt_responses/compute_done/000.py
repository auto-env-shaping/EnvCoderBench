import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Episode ends if the progress counter reaches the max episode length
    done = state_dict['progress_buf'] >= state_dict['max_episode_length']
    return done.float()  # Ensure float dtype for compatibility



