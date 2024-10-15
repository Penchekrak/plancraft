import importlib
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

from src.primitives.checks import check_inventory_wood
from src.primitives.mine_block import mine_block
from src.primitives.simple_actions import act_PLACE_TABLE, act_MAKE_WOOD_PICKAXE, act_DO, act_PLACE_FURNACE, \
    act_MAKE_IRON_SWORD, act_MAKE_STONE_PICKAXE, act_MAKE_IRON_PICKAXE

from craftax.craftax.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name
import craftax.craftax.renderer as renderer

from src.environment.gifs import visual_testing
from src.environment.wrapper import SaveStateWrapper

SEED = 0xBAD_5EED_B00B5


if __name__ == '__main__':
    importlib.reload(renderer)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger('main_logger')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=SEED, log_dir=log_dir)
    obs, state = env.reset()

    with open(log_dir + '/actions.txt', 'w') as f:
        pass

    logger.info('Started')

    ### do some stuff
    mine_block(env, BlockType.TREE, 8)
    if not check_inventory_wood(env):
        logging.error('No wood found!')

    act_PLACE_TABLE(env)
    act_MAKE_WOOD_PICKAXE(env)
    mine_block(env, BlockType.STONE, 5)
    act_PLACE_TABLE(env)
    act_MAKE_STONE_PICKAXE(env)
    mine_block(env, BlockType.COAL, 2)
    mine_block(env, BlockType.TREE, 3)
    mine_block(env, BlockType.IRON, 2)
    act_PLACE_FURNACE(env)
    mine_block(env, BlockType.STONE)
    act_PLACE_TABLE(env)
    act_MAKE_IRON_SWORD(env)
    act_MAKE_IRON_PICKAXE(env)

    mine_block(env, BlockType.WATER)

    for i in range(5):
        act_DO(env)

    # # Render
    logger.info('Rendering...')

    visual_testing(SEED, log_dir + '/actions.txt', log_dir, 1, env, renderer,
                   grid_size=(1, 1))

    logger.info('Finished')