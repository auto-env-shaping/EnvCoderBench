import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the done signal based on the state dictionary."""
    # A simple way to determine 'done' is to check if the progress buffer has reached the maximum episode length
    done = state_dict['progress_buf'] >= state_dict['max_episode_length']
    return done.float()  # Ensure output is a float tensor



