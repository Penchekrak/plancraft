from craftax.craftax.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name

from primitives.explore import explore_round
from primitives.gifs import visual_testing
from primitives.move_to_node_smart import gen_graph_smart, move_to_pos
from src.primitives.wrapper import SaveStateWrapper

from src.primitives.utils import find_block_any

import craftax.craftax.renderer as renderer
import importlib

if __name__ == '__main__':
    importlib.reload(renderer)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=0xBAD_5EED_B00B5, log_dir='../logs/')
    obs, state = env.reset()

    prev_pos = env.saved_state.player_position
    for i in range(10):
        while find_block_any(env.saved_state, BlockType.TREE.value) is None:
            G = gen_graph_smart(env.saved_state, True, False)
            pos = explore_round(env, G, prev_pos)
            move_to_pos(env, pos, G, True, False)
            prev_pos = state.player_position

        # print('Got ya! ', find_block_any(env.saved_state, BlockType.TREE.value),
        #       env.saved_state.map[env.saved_state.player_level][find_block_any(env.saved_state, BlockType.TREE.value)])
        move_to_pos(env, find_block_any(env.saved_state, BlockType.TREE.value))
        prev_pos = env.saved_state.player_position


    visual_testing(0xBAD_5EED_B00B5, '../logs/actions.txt', '../logs/', 1, env, renderer,
                   grid_size=(1, 1))
