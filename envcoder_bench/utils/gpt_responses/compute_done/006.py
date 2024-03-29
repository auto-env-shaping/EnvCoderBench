import torch
from typing import * 

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # A simple termination condition could be based on the progress in the environment
    # or when the anymal falls (detected by abnormal root_states)
    progress = state_dict["progress_buf"]
    max_episode_length = state_dict["max_episode_length"]

    root_states = state_dict["root_states"]
    z_position = root_states[:, 2]  # Assuming third component is the up/down direction

    # Assuming the anymal fell if z_position goes below a certain threshold
    z_threshold = 0.5
    fell = z_position < z_threshold

    done = (progress >= max_episode_length) | fell
    return done



