import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax.numpy as jnp
from dataclasses import asdict
from src.graphs.move_utils import get_obs_mask
from craftax.craftax.constants import OBS_DIM, BlockType, Achievement
from groq import Groq
import ast
import re


def parse_state(state) -> dict:
    stat_list = [
        "player_level", "player_health", "monsters_killed",
        "player_position", "player_food", "player_drink", "player_energy",
        "player_mana"
    ]

    stat_dict = {}
    for name in stat_list:
        try:
            stat_dict[name] = eval(f'state.{name}.value')
        except:
            stat_dict[name] = eval(f'state.{name}.tolist()')

    inventory = asdict(state.inventory)
    inventory_values = {key: (value.tolist() if isinstance(value, jnp.ndarray) else value)
                        for key, value in inventory.items()}

    def num2str_type(num: int, class_) -> str:
        return str(class_(num)).split('.')[1]

    achievements = {num2str_type(i, Achievement): ach for i, ach in enumerate(state.achievements.tolist())}
    stat_dict["achievements"] = achievements

    mask = get_obs_mask(state)[state.player_level]
    obs_map = state.map[state.player_level][mask > 0].reshape(*OBS_DIM)

    unique_elements, counts = jnp.unique(obs_map, return_counts=True)
    blocks_dict = dict(zip([num2str_type(elem, BlockType) for elem in unique_elements], counts.tolist()))

    return stat_dict, inventory_values, blocks_dict


def valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def info2prompt(task, stat_dict, blocks_dict, inventory, last_result='', last_error='') -> str:
    txt = f"""
        Execution error: {last_error}
        Chat log: None
        Level (depth-wise): {stat_dict['player_level']}
        Nearby blocks: {blocks_dict}
        Health: {stat_dict['player_health']}/10
        Hunger: {stat_dict['player_food']}/10
        Water: {stat_dict['player_drink']}/10
        Energy: {stat_dict['player_energy']}/10
        Position: {stat_dict['player_position']}
        Equipment: {inventory}
        Task: {task}
        Critique: {last_result}
        """

    return txt


def gen_code(system_prompt, content_prompt, history: list[dict], save=True) -> str:
    groq_key = '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'
    client = Groq(api_key=groq_key)

    mes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_prompt},
    ]
    if len(history):
        mes.extend(history)

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=mes
    )

    gen_ans = response.choices[0].message.content
    code = re.split('```python\n|```', gen_ans)[1]

    if not valid(code):
        raise ValueError(f'Invalid code: {code}')

    # func_name = code.split()
    # if save:
    #     with open('func_name.py') as f:
    #         print(code, file=f)

    return gen_ans, code
