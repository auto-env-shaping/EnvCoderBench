import isaacgym 
import torch 
from typing import * 

def get_executable(code, function_name):
    local_name_space = {}
    exec(code, local_name_space)
    function = local_name_space[function_name]
    return function


# class CodeHandler: 

#     def __init__(self, env, shaper_types): 
        
#         from envcoder_bench import rollout 
#         rollout(env)

#         self.checker = SanityChecker(env)
#         self.shaper_types = shaper_types

#     def register_shapers(self, shaper_strings, dir = None):

#         self.shaper_strings = shaper_strings
#         if dir is not None:
#             self._write_shapers_to_file(shaper_strings, dir)
#         self.shaper_funcs = self.compile_shapers(shaper_strings)
#         self.shaper_funcs = self._sanity_check_shapers(self.shaper_funcs)

#         print('Number of shapers registered: ', len(self.shaper_strings))
#         print('Number of shapers passed sanity check: ', len(self.shaper_funcs))

#         return self.shaper_funcs

#     def _write_shapers_to_file(self, shaper_strings, dir): 

#         for i, shaper in enumerate(shaper_strings): 
#             with open('{}/shaper_{:03d}.py'.format(dir, i), 'w') as file:
#                 file.write(shaper)
#                 file.write('\n\n')

#     def compile_shapers(self, shaper_strings): 

#         executable_shapers = [] 
#         for i, shaper_string in enumerate(shaper_strings):
#             executable_shaper_dict = {} 
#             for shaper_type in self.shaper_types: 
#                 try:
#                     executable_shaper_dict[shaper_type] = get_executable(shaper_string, shaper_type)
#                 except Exception as e:
#                     print(e)
#                     print('Error in compiling {} shaper in response id {}'.format(shaper_type, i))
#             executable_shapers.append(executable_shaper_dict)

#         return executable_shapers
    
#     def _sanity_check_shapers(self, executable_shapers): 

#         shaper_funcs = []
#         for i, executable_shaper_dict in enumerate(executable_shapers): 
#             sanity_check = self.checker.check(executable_shaper_dict)
#             if sanity_check is not None:
                
#                 if 'compute_torque' in sanity_check:
#                     executable_shaper_dict['policy_dim'] = sanity_check['compute_torque']
#                     print('Compute torque input dimension: ', sanity_check['compute_torque'])

#                 if 'compute_observation' in sanity_check:
#                     executable_shaper_dict['obs_dim'] = sanity_check['compute_observation']
#                     print('Compute observation output dimension: ', sanity_check['compute_observation'])

#                 shaper_funcs.append(executable_shaper_dict)

#         return shaper_funcs


class CodeCompiler:

    def __init__(self, env):

        self.env = env 
        self.state_dict = env.sim_state_dict 
        self.device = env.device
        self.num_envs = env.num_envs
        self.torque_dim = env.cfg["env"]["numActions"]

    def run(self, shaper_string, shaper_types): 

        executable_shaper_dict = {} 
        for shaper_type in shaper_types: 
            try:
                executable_shaper_dict[shaper_type] = get_executable(shaper_string, shaper_type)
            except Exception as e:
                print(e)
                print('Error in compiling {} shaper '.format(shaper_type))

        return executable_shaper_dict


    def extract_device(self, state_dict):
        for key in state_dict.keys(): 
            return state_dict[key].device

    def check_individual(self, fn, type): 

        if type == 'compute_reward':
            try:
                reward = fn(self.state_dict)
                return True if reward.shape ==  (self.num_envs, ) else None
            except Exception as e:
                print(e)
                return None

        elif type == 'compute_torque':

            required_output_dim = self.torque_dim
            for candidate_input_dim in range(100, 0, -1): # count down to prioritize maximal input dimension
                try:
                    output_dim = fn(torch.rand(self.num_envs, candidate_input_dim, device=self.device), self.state_dict).shape
                    if output_dim[1] == required_output_dim:
                        return candidate_input_dim
                except Exception as e:
                    pass

        elif type == 'compute_observation':
            try:
                observation = fn(self.state_dict)
                return observation.shape[1] if observation.dim() == 2 and observation.shape[0] == self.num_envs else None
            except Exception as e:
                print(e)
                return None
        
        elif type == 'compute_terminal':
            
            try:
                terminal = fn(self.state_dict)
                return True if terminal.shape == (self.num_envs, ) else None
            except Exception as e:
                print(e)
                return None


    def check(self, executable_shapers: Dict): 
        
        sanity_check = {}
        for shaper_type, executable_shaper_dict in executable_shapers.items(): 
            sane = self.check_individual(executable_shaper_dict, shaper_type)
            sanity_check[shaper_type] = sane
        
        # if one of them are None, return None 
        # otherwise return sanity_check 
            
        if None in sanity_check.values():
            return None
        else:
            return sanity_check


def extract_num_envs(state_dict): 

    # extract the number of environments from the state_dict
    # check any keys with shape (num_envs, ...)
    # and return num_envs

    for key in state_dict.keys(): 
        if len(state_dict[key].shape) > 1: 
            return state_dict[key].shape[0]
        

if __name__ == "__main__": 
    from envcoder_bench import create_env, rollout, get_state_dict_summary
    from envcoder_bench.utils.llm_utils import LanguageModel

    env, _, _ = create_env(0)
    rollout(env)

    LLM = LanguageModel(task_description = 'make the anymal run as fast as possible',
                        state_dict_summary = env.sim_state_dict,
                        prompt_dir = 'default.yaml',
                        model = 'gpt-4-turbo-preview')
    
    shaper_types = ['compute_reward', 'compute_torque', 'compute_observation', 'compute_terminal']
    results = LLM.query(5, shaper_types)

    code_handler = CodeCompiler(env)
    shaper_dicts = code_handler.register_shapers(results, './gpt_responses')
    print(shaper_dicts)