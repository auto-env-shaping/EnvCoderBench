# EnvCoderBench

This repository supports a compute-efficient, parallel evaluation of different environment shaping operations, i.e., evaluating RL performance under different
- reward functions 
- observation spaces
- action spaces
- initial/goal states
- terminal conditions

## Why do we need parallel evaluation? 

Let's consider a scenario; imagine training a quadruped with RL -- we might want to test out different reward functions to get the best locomotion behavior. Let's say we have $n$ reward functions to test out, i.e., 

$$[r_1, r_2, \cdots, r_n].$$

The most common way to evaluate the performance of these reward functions is to either *sequentially* or *parallely* run independent Isaac Gym/Sim RL training processes for each reward functions. 

```bash
python train.py --reward_idx 0  --num_envs 256;
python train.py --reward_idx 1  --num_envs 256;
...
python train.py --reward_idx n  --num_envs 256;
```

Let's imagine launching all these scripts i **parallel** in a single GPU, i.e., RTX 4090 with 24GB of VRAM. With 256 environments for each reward function, launching more than 8 procesess is impossible -- it is bottlenecked by the VRAM. Speed of training for each processes rapidly decreases as well as we launch more processes. Imagine we want to run this evaluation for over $n>100$ reward functions (which is absolutely a realistic number with hyperparameter sweeps) -- things become extremely inefficient. This makes us wonder;  **Is this really the best way to use our GPU compute?**

Isaac Gym/Sim is designed to support massively parallel environments with independent agents. It is thus possible to assign different rewards to different environment subgroups, and evaluate each reward in a single process, i.e., 

```bash 
python parallel_train.py --reward_idx 0,1,2,..,n --num_envs ${256*n}
```

In fact, the latter approach has significant benefit over the former in terms of (a) computational speed, and (b) VRAM usage. When using RTX4090 with 24GB VRAM, the latter approach allows evaluation of up to 250 rewards, while the former approach only allows 8 rewards. Decrease of training speed is much less significant as well.  

![](https://share.cleanshot.com/FVMSBZJf+)


## Code Structure

### Evaluate human-written shaping candidates 

If you want to test out any shaping operations, include all your shaping candidates under `./envcoder_bench/user_functions` following the [instruction](). Then, run: 

```bash 
cd envcoder_bench
python eval_shapers.py --task Anymal --shaper_types compute_reward compute_observation 
```

This will evaluate all your shaping candidates in a single Isaac Gym process 

### Run Outer-loop Optimization using LLMs

To let LLMs figure out the best shaping candidates, run: 

```bash
cd envcoder_bench
python auto_shape.py --task Anymal --task_description "Make anymal run as fast as possible" --shaper_types compute_reward compute_observation --llm gpt-4-turbo-preview
```
It supports both OpenAI GPT models and Anthropic Claude models as an LLM agent. 