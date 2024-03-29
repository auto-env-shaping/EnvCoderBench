import torch
from typing import * 

    # We can define an episode to be done if it has reached a maximum number of timesteps
    max_episode_length = state_dict['max_episode_length']
    progress_buf = state_dict['progress_buf']
    
    done = progress_buf >= max_episode_length
    
    return done



