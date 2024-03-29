import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Task termination condition: exceed maximum episode length
    # Also, one might consider adding failure conditions, such as the robot flipping over, but no explicit mention here
    max_episode_length = state_dict["max_episode_length"].unsqueeze(0)  # shape adjustment for broadcasting
    done = state_dict["progress_buf"] >= max_episode_length
    return done



