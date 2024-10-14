import importlib
import sys
import os

from primitives.simple_actions import act_PLACE_TABLE, act_MAKE_WOOD_PICKAXE, act_DO, act_PLACE_FURNACE, \
    act_MAKE_IRON_SWORD, act_MAKE_STONE_PICKAXE

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

SEED = 0xBAD_5EED_B00B

def explore_and_chop(env: SaveStateWrapper, block_type, max_iter = 25, can_dig=False, can_place=False):
    prev_pos = env.saved_state.player_position

    for i in range(max_iter):
        if find_block_any(env.saved_state, block_type) is not None:
            break
        print(f'exploring for {BlockType(block_type).name}...')
        G = gen_graph_smart(env.saved_state, can_dig, can_place)
        pos = explore_round(env, G, prev_pos)
        move_to_pos(env, pos, G, can_dig, can_place)
        prev_pos = env.saved_state.player_position
    else:
        print('no block found')
        return

    targets = find_block_all(env.saved_state, block_type)
    closest_target_index = abs(targets - env.saved_state.player_position).sum(axis=-1).argmin()
    closest_target = targets[closest_target_index]
    move_to_pos(env, closest_target)
    act_DO(env)

if __name__ == '__main__':
    importlib.reload(renderer)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=SEED, log_dir=log_dir)
    obs, state = env.reset()

    with open(log_dir + '/actions.txt', 'w') as f:
        pass

    #do some stuff
    for i in range(8):
        explore_and_chop(env, BlockType.TREE.value)

    act_PLACE_TABLE(env)
    act_MAKE_WOOD_PICKAXE(env)
    for i in range(5):
        explore_and_chop(env, BlockType.STONE.value, can_dig=True, can_place=False)

    act_PLACE_TABLE(env)
    act_MAKE_STONE_PICKAXE(env)

    for i in range(3):
        explore_and_chop(env, BlockType.COAL.value, can_dig=True, can_place=False)
    for i in range(1):
        explore_and_chop(env, BlockType.IRON.value, can_dig=True, can_place=False)

    act_PLACE_FURNACE(env)
    act_MAKE_IRON_SWORD(env)

    # # Render

    visual_testing(SEED, log_dir + '/actions.txt', log_dir, 1, env, renderer,
                   grid_size=(1, 1))
