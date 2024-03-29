import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example condition for episode to be marked as done
    # Ending the episode if the progress buffer reaches or exceeds max episode length
    done = state_dict["progress_buf"] >= state_dict["max_episode_length"]
    # Note: Ensure done is a Boolean tensor
    done = done.to(torch.bool)
    return done



