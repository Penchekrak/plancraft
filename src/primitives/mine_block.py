import logging

from craftax.craftax.constants import BlockType

from src.graphs.move_to_node_smart import move_to_pos
from src.graphs.move_utils import find_block_all
from src.primitives.checks import check_inventory_stone
from src.primitives.explore_until import explore_until
from src.primitives.simple_actions import act_DO

logger = logging.getLogger()

def mine_block(env, block_type: BlockType, count: int = 1, max_iter = 25):
    logger.info(f'mining {count } of {block_type}...')

    can_dig = env.saved_state.inventory.pickaxe
    can_place = check_inventory_stone(env)

    if block_type == BlockType.WOOD:
        block_type = BlockType.TREE

    for block_iteration in range(count):
        logger.debug(f'iteration {block_iteration}/{count} of find_block...')
        explore_until(env, callback=block_type, max_iter=max_iter)

        targets = find_block_all(env.saved_state, block_type)
        if len(targets) == 0: continue

        closest_target_index = abs(targets - env.saved_state.player_position).sum(axis=-1).argmin()
        closest_target = targets[closest_target_index]
        move_to_pos(env, closest_target, can_dig=can_dig, can_place=can_place)
        act_DO(env)
    logger.debug(f'Finished mining of {block_type}.')
