import importlib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

from craftax.craftax.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name
import craftax.craftax.renderer as renderer

from src.primitives.explore import explore_round
from src.primitives.gifs import visual_testing
from src.primitives.move_to_node_smart import gen_graph_smart, move_to_pos
from src.primitives.utils import find_block_any, find_block_all
from src.primitives.wrapper import SaveStateWrapper

SEED = 0xBAD_5EED_B00B5
BLOCK_TYPE_VALUE = BlockType.IRON.value

if __name__ == '__main__':
    importlib.reload(renderer)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=SEED, log_dir=log_dir)
    obs, state = env.reset()

    with open(log_dir + '/actions.txt', 'w') as f:
        pass

    prev_pos = env.saved_state.player_position
    for i in range(10):
        while find_block_any(env.saved_state, BLOCK_TYPE_VALUE) is None:
            G = gen_graph_smart(env.saved_state, True, False)
            pos = explore_round(env, G, prev_pos)
            move_to_pos(env, pos, G, True, False)
            prev_pos = env.saved_state.player_position
        trees = find_block_all(env.saved_state, BLOCK_TYPE_VALUE)
        closest_index = abs(trees - env.saved_state.player_position).sum(axis=-1).argmin()
        closest_tree = trees[closest_index]
        move_to_pos(env, closest_tree)
        prev_pos = env.saved_state.player_position


    visual_testing(SEED, log_dir + '/actions.txt', log_dir, 1, env, renderer,
                   grid_size=(1, 1))
