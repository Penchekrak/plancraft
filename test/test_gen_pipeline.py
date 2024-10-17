import importlib
import sys
import os
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
llm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm')
os.makedirs(log_dir, exist_ok=True)

from src.environment.code_parser import find_most_function_calls

from src.primitives.simple_actions import act_PLACE_TABLE, act_MAKE_WOOD_PICKAXE, act_DO, act_PLACE_FURNACE, \
    act_MAKE_IRON_SWORD, act_MAKE_STONE_PICKAXE, act_MAKE_WOOD_SWORD, act_MAKE_STONE_SWORD, act_MAKE_IRON_PICKAXE

from craftax.craftax.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name
import craftax.craftax.renderer as renderer

# from src.primitives.explore import explore_choose_node
from src.primitives.mine_block import mine_block
from src.environment.gifs import visual_testing
from src.graphs.move_to_node_smart import gen_graph_smart, move_to_pos
from src.graphs.move_utils import find_block_any, find_block_all
from src.environment.wrapper import SaveStateWrapper

from prompt import parse_state, info2prompt, gen_code
import types
from src.primitives.checks import *
import logging
import ast

SEED = 0xBAD_5EED_B00B5
TASK = 'Create iron sword'
TASK_CHECKER = {'Create iron sword': check_inventory_iron_sword}

if __name__ == '__main__':
    # def main():
    importlib.reload(renderer)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=SEED, log_dir=log_dir)
    obs, state = env.reset()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger('main_logger')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    with open(log_dir + '/actions.txt', 'w') as f:
        pass

    with open(f'{llm_dir}/prompt.txt', 'r') as file:
        content_prompt = file.read()

    # #do some stuff
    stat_dict, inventory_values, blocks_dict = parse_state(state)

    info_prompt = info2prompt(TASK, stat_dict, blocks_dict, inventory_values)

    system_prompt = """You are a helpful assistant that writes python code to
        complete any Craftax task specified by me."""

    debug_prompt = """
        The code you provided in the previous answer has some logical errors. Above, I list all achievements and inventory that were obtained from the execution of this generated code:
        Inventory: {}
        Achievements: {} 

        But you need to fix or rewrite this code in order to achieve the given task: {}
        """
    error_prompt = """
        The code you provided in the previous answer has some bugs: {}
        You need to fix or rewrite this code in order to achieve the given task: {}  
    """

    content_prompt += info_prompt

    history = []
    result, error_feedback = None, None
    present_symbols = {}

    for i in range(5):
        if i >= 1:
            env.reset()
            history.extend([
                {
                    'role': 'assistant',
                    'content': ans
                },
                {
                    'role': 'user',
                    'content': debug_prompt.format(inventory_values, stat_dict["achievements"],
                                                   TASK) if error_feedback is None else error_prompt.format(
                        error_feedback, TASK)
                }
            ]
            )

        ans, code = gen_code(system_prompt, content_prompt, history, save=True)
        print("-" * 100)
        print(code)
        print("-" * 100)

        # exec('print(1337)')
        old_symbols = set(locals().keys()).union(set(globals().keys())).union({'old_symbols'})
        print(f"{old_symbols=}")
        exec(code)  # define all generated functions
        new_symbols = set(locals().keys()).union(set(globals().keys())).difference(old_symbols)
        print(f"{new_symbols=}")

        try:
            # eval((func_name := next(iter(new_symbols))) + '(env)')
            func_name, _ = find_most_function_calls(code, new_symbols)
            for func in new_symbols:
                del locals()[func]

        except Exception as e:
            error_feedback = e
            logger.info(error_feedback)
            continue

        stat_dict, inventory_values, blocks_dict = parse_state(env.saved_state)

        visual_testing(SEED, log_dir + f'/actions.txt', log_dir, 1, env, renderer,
                       grid_size=(1, 1), gif_name=f'gif_{TASK}_{i}.gif')

        # delete file actions.txt
        os.remove(log_dir + f'/actions.txt')

        if result := TASK_CHECKER[TASK](env):
            with open(f'{llm_dir}/{TASK}.py', 'w+'):
                print(code, file=f)
            break

        logger.info(result)
        logger.info('%' * 100)
        time.sleep(7)
