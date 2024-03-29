from openai import OpenAI
from anthropic import Anthropic 

import yaml 
from pathlib import Path 
from typing import * 
import asyncio
import concurrent
from pprint import pprint
import os 
import ast 
from tqdm import tqdm 
from astroid import parse
from envcoder_bench.scripts.train import get_state_dict_summary
from envcoder_bench.utils.new_code_utils import CodeCompiler

CUR_PATH = Path(__file__).parent.absolute()


def get_executable(code, function_name):
    local_name_space = {}
    exec(code, local_name_space)
    function = local_name_space[function_name]
    return function


class LanguageModel:
    def __init__(self, env, task_description: str, shaper_types: List, \
                 prompt_dir = 'default_prompt.yaml', model = 'gpt-4-turbo-preview'):
        
        if 'gpt' in model:
            self.client = OpenAI()
            self.api = self.client.chat.completions
        else:
            self.client = Anthropic()
            self.api = self.client.messages

        self.shaper_types = shaper_types
        self.task_description = task_description
        self.state_dict_summary = get_state_dict_summary(env)
        self.prompt_dict = self.load_yaml(f'{CUR_PATH}/prompts/{prompt_dir}') # dictionary of prompts 
        self.function_signatures = self.load_yaml(f'{CUR_PATH}/function_signatures.yaml')
        self.model = model 

        self.num_concurrent_query = 1

        self.compiler = CodeCompiler(env)

    def get_helper_functions(self): 

        directory_of_helpers = self.prompt_dict['helper_functions']
        helper_functions = ''
        for helper in directory_of_helpers:
            with open(f'{CUR_PATH}/{helper}', 'r') as file:
                helper_functions += file.read()
                helper_functions += '\n\n'

    def get_function_signatures(self):

        # get the function signatures for the given shaper and return as a single string 
        function_signatures = ''
        for type in self.shaper_types:
            function_signatures += self.function_signatures[type]
            function_signatures += '\n\n'

        return function_signatures
    
    
    def get_tip_prompts(self): 

        # get the tip prompts for the given shaper and return as a single string 
        tip_prompts = ''
        for type in self.shaper_types:
            if f'{type}_tips' in self.prompt_dict.keys():
                tip_prompts += self.prompt_dict[f'{type}_tips']
                tip_prompts += '\n\n'

        return tip_prompts


    def query_single(self):
        '''
        Returns the function blocks for the given task description
        '''
        
        def append_system(message): 

            message.insert(0, {"role": "system", "content": self.system_prompt})
            return message

        if 'gpt' in self.model:
            response = self.api.create(
                    model=self.model,
                    messages = append_system(self.messages), 
                    max_tokens = 4096
                    )
            
        else: 
            response = self.api.create(
                    model=self.model,
                    system = self.system_prompt,
                    messages = self.messages, 
                    max_tokens = 4096
                    )

        python_block = self.extract_python_block(self.response_to_string(response))

        return python_block 
    
    def response_to_string(self, response):

        if 'gpt' in self.model:
            return response.choices[0].message.content
        else:
            return response.content[0].text

    def construct_prompt(self, best_idx, extras):

        if best_idx is None: 

            self.system_prompt = self.prompt_dict['system'].format(shaper = self.shaper_types, \
                                                            function_signatures = self.get_function_signatures()) 
            user_prompt = self.prompt_dict['user'].format(state_dict_summary = self.state_dict_summary, \
                                                        task_description = self.task_description, shaper = self.shaper_types, \
                                                        helper_functions = self.get_helper_functions())
            tip_prompts = self.get_tip_prompts()

            self.messages = [
                            # {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt + tip_prompts },
                        ]
            
        else: 
            reflection_prompt = self.prompt_dict['reflection'].format(result_description = self.convert_log_to_string(extras))

            self.messages = self.messages + [
                {"role": "assistant", "content": self.responses[best_idx] }, 
                {"role": "user", "content": reflection_prompt}
            ]



    def convert_log_to_string(self, extras):
        '''
        Converts the log to a string
        '''
        string = ''
        for key, value in extras.items():
            string += f'{key}: {value}\n'
        return string

    def query(self, n, best_idx = None, extras = None):
        '''
        Returns the function blocks for the given task description, ensuring `n` successful responses.
        '''
        executables = [] 
        responses = []
        trial = 0
        success_count = 0  # Track the number of successful responses
        
        self.construct_prompt(best_idx, extras)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Initially submit a batch of tasks
        #     future_responses = {executor.submit(self.query_single): i for i in range(self.num_concurrent_query)}

        with tqdm(total=n, desc = f'Querying {self.model}') as pbar:
            while success_count < n:
                # Wait for any future to complete
                # done, _ = concurrent.futures.wait(future_responses, return_when=concurrent.futures.FIRST_COMPLETED)
                
                # for future in done:
                #     response = future.result()
                    
                #     if response:
                response = self.query_single()
                trial += 1
                check, executable = self.compile_and_check(response)
                if check is not None:
                    # Process successful response
                    if 'compute_observation' in executable.keys():
                        executable['obs_dim'] = check['compute_observation']
                    if 'compute_torque' in executable.keys():
                        executable['policy_dim'] = check['compute_torque']
                    executables.append(executable)
                    responses.append(response)
                    success_count += 1  # Increment success count
                    pbar.update(1)  # Update the progress bar
                    
                    if success_count >= n:
                        break  # Break the loop if we've collected enough successes
                else:
                    print(f'sanity check failed for response #{trial}')
                # else:
                #     print(f'Query failed for response #{trial}')
                    
                    # # Remove the completed task
                    # future_responses.pop(future)
                pbar.set_description(f'Querying {self.model} | Trial: {trial} | Successes: {success_count} ')
                
                # # Submit new tasks as needed to maintain `n` attempts
                # while len(future_responses) < n - success_count:
                #     new_future = executor.submit(self.query_single)
                #     future_responses[new_future] = trial  # Use trial for identifying future tasks

        self.responses = responses

        return executables
    
    def compile_and_check(self, shaper_string):
        executable_shaper_dict = self.compiler.run(shaper_string, self.shaper_types)
        sanity_check = self.compiler.check(executable_shaper_dict)
        return sanity_check, executable_shaper_dict


    def extract_python_block(self, code):
        '''
        Extracts the python code block from the openai code response
        '''
        return code.split('```python')[1].split('```')[0]


    def load_yaml(self, yaml_path):
        '''
        Loads a yaml file
        '''
        with open(yaml_path, 'r') as stream:
            return yaml.safe_load(stream)
        

    def save_code_to_file(self, codes, dir):
        '''
        Saves the code to a file
        '''
        for i, code in enumerate(codes):
            file_path = '{}/{:03d}.py'.format(dir, i)
            with open(file_path, 'w') as file:
                file.write(code)



if __name__ == "__main__": 

    from envcoder_bench.scripts.train import create_env, rollout, get_state_dict_summary

    env, _, _ = create_env(0)
    rollout(env)

    LLM = LanguageModel(task_description = 'make the anymal run as fast as possible', 
                        state_dict_summary = get_state_dict_summary(env), \
                        prompt_dir = 'default.yaml', model = 'gpt-4-turbo-preview')
    
    results = LLM.query(10)