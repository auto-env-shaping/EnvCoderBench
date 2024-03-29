from envcoder_bench.utils.new_code_utils import CodeCompiler
from envcoder_bench import create_env, rollout, get_state_dict_summary
from rsl_rl.ppo_ensemble import Runner
from typing import * 

import numpy as np
import os, datetime
import argparse

parser = argparse.ArgumentParser(description='Outer Loop Parameters')

# Training Parameters
parser.add_argument('--shaper_types', nargs='+', default=['compute_reward', 'compute_torque', 'compute_observation', 'compute_terminal'], help='Types of shapers')
parser.add_argument('--task', default = 'Anymal')
parser.add_argument('--learning_iterations', type=int, default=500, help='Number of learning iterations')
parser.add_argument('--wandb_project', type=str, default="parallel-training", help='Wandb project name')
parser.add_argument('--wandb_group', type=str, default="eureka-dev", help='Wandb group name')
parser.add_argument('--wandb_name', type=str, default="", help='Wandb run name')
parser.add_argument('--wandb_job_type', type=str, default="", help='Wandb job type')
parser.add_argument('--wandb_entity', type=str, default="robot-locomotion", help='Wandb entity name')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=0, help='Seed')

args = parser.parse_args()

# Load shaping candidates
directory = "../user_functions"

# step 1. read the files from the folder
files = os.listdir(directory)

# step 2. read the content of the files
shaping_candidates = []
for file in files:
    with open(os.path.join(directory, file), 'r') as f:
        shaping_candidates.append(f.read())

num_envs = 1000 * len(shaping_candidates)

# ###########################
# # select individual shapers as a baseline for the parallel training
# candidate_idx = 2
# shaping_candidates = shaping_candidates[candidate_idx:candidate_idx+1]
# num_envs = 1000
# ###########################

# Helper classes
env, _, _ = create_env(gpu_id=args.gpu_id, task_name = args.task, task={"env": {"numEnvs": num_envs}}, seed=args.seed)
rollout(env)
state_dict_summary = get_state_dict_summary(env)

# Compile the shaping candidates
code_handler = CodeCompiler(env)

valid_shapers = []
for candidate in shaping_candidates: 
    executable = code_handler.run(candidate, args.shaper_types)
    check = code_handler.check(executable)
    if check is not None:
        if 'compute_observation' in executable.keys():
            executable['obs_dim'] = check['compute_observation']
        if 'compute_torque' in executable.keys():
            executable['policy_dim'] = check['compute_torque']
        valid_shapers.append(executable)

# for iter in range(num_iters):
print(f'''==== Training with manual reward functions ====''')

env.register_shapers(valid_shapers)

# initialize wandb
import wandb
wandb.init(
# set the wandb project where this run will be logged
project=args.wandb_project,
group=args.wandb_group,
entity=args.wandb_entity,
job_type=args.wandb_job_type,
name=args.wandb_name,
mode="online",
)

runner = Runner(env, device=env.device)
runner.learn(num_learning_iterations=args.learning_iterations, init_at_random_ep_len=True, eval_freq=100)

performance = runner.get_metrics()
