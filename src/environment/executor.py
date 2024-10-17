from craftax.craftax.constants import Action, CLOSE_BLOCKS, BlockType, DIRECTIONS
import os

from src.primitives.checks import *
from src.environment.wrapper import SaveStateWrapper
import logging

logger = logging.getLogger("Executor")


def check_nearby_table(env):
    for direction in CLOSE_BLOCKS:
        pos = env.saved_state.player_position + direction
        block = BlockType(env.saved_state.map[env.saved_state.player_level][pos[0], pos[1]])
        if block == BlockType.CRAFTING_TABLE:
            return True
    return False

def check_nearby_furnace(env):
    for direction in CLOSE_BLOCKS:
        pos = env.saved_state.player_position + direction
        block = BlockType(env.saved_state.map[env.saved_state.player_level][pos[0], pos[1]])
        if block == BlockType.FURNACE:
            return True
    return False


class MaxStepsException(Exception):
    pass


def executor(env: SaveStateWrapper, action_list: list[Action]):
    # append acton_list to actions.csv
    action_log_file = os.path.join(env.log_dir, 'actions.txt')
    logger.debug(f'acting: {action_list}')

    with open(action_log_file, 'a+') as f:
        # execute actions
        for action in action_list:
            f.write(f'{action.value}\n')

            if action == Action.PLACE_TABLE and check_inventory_wood(env) < 2:
                logger.info("ERROR: Not enough wood to place table!")
            if action == Action.PLACE_FURNACE and check_inventory_stone(env) < 2:
                logger.info("ERROR: Not enough stone to place furnace!")

            if action == Action.MAKE_WOOD_SWORD and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft sword!")
            if action == Action.MAKE_WOOD_PICKAXE and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft pickaxe!")

            if action == Action.MAKE_STONE_SWORD and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft sword!")
            if action == Action.MAKE_STONE_PICKAXE and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft pickaxe!")

            if action == Action.MAKE_IRON_PICKAXE and check_nearby_furnace(env) == False:
                logger.info("ERROR: No furnace nearby to craft pickaxe!")
            if action == Action.MAKE_IRON_PICKAXE and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft pickaxe!")

            if action == Action.MAKE_IRON_SWORD and check_nearby_table(env) == False:
                logger.info("ERROR: No table nearby to craft sword!")
            if action == Action.MAKE_IRON_SWORD and check_nearby_furnace(env) == False:
                logger.info("ERROR: No furnace nearby to craft sword!")

            if action == Action.MAKE_WOOD_PICKAXE and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft wood pickaxe!")

            if action == Action.MAKE_STONE_PICKAXE and check_inventory_stone(env) < 1:
                logger.info("ERROR: Not enough stone to craft stone pickaxe!")
            if action == Action.MAKE_STONE_PICKAXE and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft stone pickaxe!")

            if action == Action.MAKE_IRON_PICKAXE and check_inventory_stone(env) < 1:
                logger.info("ERROR: Not enough stone to craft iron pickaxe!")
            if action == Action.MAKE_IRON_PICKAXE and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft iron pickaxe!")
            if action == Action.MAKE_IRON_PICKAXE and check_inventory_coal(env) < 1:
                logger.info("ERROR: Not enough coal to craft iron pickaxe!")
            if action == Action.MAKE_IRON_PICKAXE and check_inventory_iron(env) < 1:
                logger.info("ERROR: Not enough iron to craft iron pickaxe!")

            if action == Action.MAKE_WOOD_SWORD and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft wood sword!")

            if action == Action.MAKE_STONE_SWORD and check_inventory_stone(env) < 1:
                logger.info("ERROR: Not enough stone to craft stone sword!")
            if action == Action.MAKE_STONE_SWORD and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft stone sword!")

            if action == Action.MAKE_IRON_SWORD and check_inventory_stone(env) < 1:
                logger.info("ERROR: Not enough stone to craft iron sword!")
            if action == Action.MAKE_IRON_SWORD and check_inventory_wood(env) < 1:
                logger.info("ERROR: Not enough wood to craft iron sword!")
            if action == Action.MAKE_IRON_SWORD and check_inventory_coal(env) < 1:
                logger.info("ERROR: Not enough coal to craft iron sword!")
            if action == Action.MAKE_IRON_SWORD and check_inventory_iron(env) < 1:
                logger.info("ERROR: Not enough iron to craft iron sword!")

            if (action == Action.DO and check_forward_block(env) == BlockType.STONE and
                    check_inventory_wood_pickaxe(env) == False):
                logger.info("ERROR: Can not dig stone! Need wood pickaxe!")

            if (action == Action.DO and check_forward_block(env) == BlockType.COAL and
                    check_inventory_stone_pickaxe(env) == False):
                logger.info("ERROR: Can not dig coal! Need stone pickaxe!")

            if (action == Action.DO and check_forward_block(env) == BlockType.IRON and
                    check_inventory_stone_pickaxe(env) == False):
                logger.info("ERROR: Can not dig iron! Need stone pickaxe!")

            obs, state, reward, done, info = env.step(action)

            if env.action_count >= env.max_steps:
                logger.info("ERROR: Max steps reached!")
                raise MaxStepsException("Max steps reached!")