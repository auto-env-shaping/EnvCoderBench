from envcoder_bench.utils.llm_utils import LanguageModel
from envcoder_bench import create_env, rollout, get_state_dict_summary
from rsl_rl.ppo_ensemble import Runner
from typing import * 
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Outer Loop Parameters')

# Outer Loop Parameters
parser.add_argument('--num_iters', type=int, default=5, help='Number of iterations')
parser.add_argument('--task', default = 'Anymal')
parser.add_argument('--task_description', type=str, default="make the anymal run as fast as possible")
parser.add_argument('--shapers_per_iter', type=int, default=10, help='Number of shapers per iteration')
parser.add_argument('--shaper_types', nargs='+', default=['compute_reward', 'compute_torque', 'compute_observation', 'compute_terminal'], help='Types of shapers')
parser.add_argument('--learning_iterations', type=int, default=500, help='Number of learning iterations')
parser.add_argument('--llm', type=str, default="gpt-4-turbo-preview")
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=0, help='Seed')

args = parser.parse_args()

best_reward_so_far = None

# Helper classes
env, _, _ = create_env( gpu_id=args.gpu_id, task_name = args.task, 
                        task={"env": {"numEnvs": 4000}}, 
                        seed=args.seed,
                        )
rollout(env)
state_dict_summary = get_state_dict_summary(env)

LLM = LanguageModel(env, 
                    task_description = args.task_description,
                    shaper_types = args.shaper_types, 
                    prompt_dir = 'default.yaml',
                    # model = 'claude-3-opus-20240229')
                    model = 'gpt-4-turbo-preview')

valid_shapers = LLM.query( n = 10 )

for iter in range(args.num_iters):
    print(f'''==== AutoShaping Iteration {iter} ====''')
    
    env.register_shapers(valid_shapers)

    runner = Runner(env, device=env.device)
    runner.learn(num_learning_iterations=args.learning_iterations, init_at_random_ep_len=True, eval_freq=100)
    
    best_idx, extras = runner.get_metrics()

    new_shapers = LLM.query(n = 10, best_idx = best_idx, extras = extras)
    valid_shapers = new_shapers


    # # # Evaluate reward candidates
    # # runner.reset() # reset environments
    # # runner.reset_metrics() # reset metrics
    # # runner.reinitialize_policies() # reinitialize policies
    # # runner.set_rewards(reward_fns) # load the reward functions
    # # runner.learn(learning_iterations=learning_iterations) # train for learning_iterations
    
    # # Reward reflection
    # performance_history = runner.get_metrics(steps=np.linspace(0, learning_iterations, num=10, endpoint=True, dtype=int))
    # final_performance = runner.get_metrics()
    
    # best_reward_so_far = reward_strings[np.argmax(final_performance)]
    # best_reward_performance_history = performance_history[np.argmax(final_performance)]
    
    # reward_strings = LLM.query(reflection_prompt + best_reward_so_far + str(best_reward_performance_history), n=10)
    
    # print(f"\nBest reward so far:\n{best_reward_so_far}")
    