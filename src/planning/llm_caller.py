from dataclasses import asdict

import jax.numpy
from craftax.craftax.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name
from groq import Groq

from environment.wrapper import SaveStateWrapper
from graphs.move_utils import get_obs_mask


class LLMCaller:
    def __init__(self, api_key: str, task='Craft Iron Sword'):
        self.task = task
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)

        self.history = []

    def reset(self):
        self.history = []
        self.client = Groq(api_key=self.api_key)

    def call_llm(self, msg: str) -> str:
        self.history += [{"role": "user", "content": msg}]
        response = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=self.history
        )
        response_text = response.choices[0].message.content
        self.history += [{"role": "system", "content": response_text}]
        return response_text


    def generate_observation_prompt(self, env) -> str:
        state = env.saved_state
        text = ''
        text += 'Current information about env state:\n'
        text += f'Player level: {state.player_level}\n'

        ## Nearby blocks
        text += f'Nearby blocks:\n'

        mask = get_obs_mask(state)[state.player_level]
        visible_blocks = state.map[state.player_level][jax.numpy.where(mask)]
        unique_blocks = jax.numpy.unique(visible_blocks, return_counts=True)

        for block, count in zip(unique_blocks[0], unique_blocks[1]):
            text += f'\t{BlockType(block.item())}: {count.item()}\n'

        ## Inventory
        text += 'Inventory:\n'
        inventory_dict = asdict(state.inventory)
        for item in inventory_dict:
            text += f'\t{item}: {inventory_dict[item]}\n'

        ## task
        text += f'Task: {self.task}\n'

        print('\n')
        ## Achievements TODO

        return text

    def get_basic_prompt(self, basic_prompt_path='./prompts/basic.txt'):
        with open(basic_prompt_path, 'r') as file:
            basic_prompt = file.read()
        return basic_prompt

    def get_debug_prompt(self, debug_prompt_path='./prompts/debug.txt'):
        with open(debug_prompt_path, 'r') as file:
            debug_prompt = file.read()
        debug_prompt = debug_prompt.format(self.task)
        return debug_prompt

    def get_error_prompt(self, error_string, error_prompt_path='./prompts/error.txt'):
        with open(error_prompt_path, 'r') as file:
            error_prompt = file.read()
        error_prompt = error_prompt.format(error_string, self.task)
        return error_prompt

    def get_main_prompt(self, main_prompt_path='./prompts/main.txt'):
        with open(main_prompt_path, 'r') as file:
            main_prompt = file.read()
        return main_prompt

    def get_response_format_prompt(self, response_format_prompt_path='./prompts/response_format.txt'):
        with open(response_format_prompt_path, 'r') as file:
            response_format_prompt = file.read()
        return response_format_prompt

    def get_full_starting_prompt(self, env):
        return (self.get_basic_prompt() +
                self.get_main_prompt() +
                self.generate_observation_prompt(env) +
                'Current task: ' + self.task + '\n' * 3 +
                self.get_response_format_prompt())


if __name__ == '__main__':
    llm_caller = LLMCaller(api_key='', task='Craft Iron Sword')
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=42, log_dir='../../logs/')
    obs, state = env.reset(seed=42)

    #print(llm_caller.generate_observation_prompt(env))
    # print('=' * 80)
    # print(llm_caller.get_debug_prompt())
    # print('=' * 80)
    # print(llm_caller.get_error_prompt('error'))
    # print('=' * 80)
    # print(llm_caller.get_main_prompt())

    print(llm_caller.get_full_starting_prompt(env))