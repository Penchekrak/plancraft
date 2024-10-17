import logging

from craftax.craftax.constants import Action, BlockType, DIRECTIONS
from src.environment.executor import check_nearby_table, check_nearby_furnace
from src.primitives.checks import *
from src.environment.executor import executor
from src.graphs.move_to_node_smart import DIRECTIONS_TO_ACTIONS, NEED_DIG, NEED_CHOP

logger = logging.getLogger()

# UTILS
def act_MOVE_DIRECTION(env, direction):
    executor(env, [DIRECTIONS_TO_ACTIONS[direction]])


# Simple Actions

def act_DO(env):
    action = Action.DO
    executor(env, [action])


def act_PLACE_STONE(env):
    action = Action.PLACE_STONE
    executor(env, [action])


def act_PLACE_TABLE(env):
    action = Action.PLACE_TABLE
    executor(env, [action])
    if check_forward_block(env) == BlockType.CRAFTING_TABLE:
        logger.info('Placed table')
    else:
        logger.info('Could not place table')


def act_PLACE_FURNACE(env):
    # if you can just place it
    if check_forward_block(env) in [BlockType.PATH, BlockType.GRASS, BlockType.SAND]:
        action = Action.PLACE_FURNACE
        executor(env, [action])
        logger.info('Placed furnace')
        return

    # if there already is a furnace
    if check_nearby_furnace(env):
        logger.info('Furnace already exists nearby')
        return

    # if crafting_table is in front of you
    if check_forward_block(env) == BlockType.CRAFTING_TABLE:
        for direction in DIRECTIONS[1: 5]:
            pos = env.saved_state.player_position + direction
            block = BlockType(env.saved_state.map[env.saved_state.player_level][pos[0], pos[1]])
            if block == BlockType.CRAFTING_TABLE:
                continue

            if block in NEED_DIG + NEED_CHOP:
                act_MOVE_DIRECTION(env, direction)
                act_DO(env)
                act_PLACE_FURNACE(env)
                return

            if block in [BlockType.PATH, BlockType.GRASS, BlockType.SAND]:
                opposite_direction = -direction
                opposite_pos = env.saved_state.player_position + opposite_direction
                opposite_block = BlockType(
                    env.saved_state.map[env.saved_state.player_level][opposite_pos[0], opposite_pos[1]]
                )
                # if you can rotate
                if opposite_block in [BlockType.PATH, BlockType.GRASS, BlockType.SAND]:
                    act_MOVE_DIRECTION(env, opposite_direction)
                    act_MOVE_DIRECTION(env, direction)
                    act_PLACE_FURNACE(env)
                    return
                else:
                    continue
    logger.info('Could not place furnace!')


def act_PLACE_PLANT(env):
    action = Action.PLACE_PLANT
    logger.info('Placing plant...')
    executor(env, [action])



def act_MAKE_WOOD_PICKAXE(env):
    action = Action.MAKE_WOOD_PICKAXE
    executor(env, [action])
    if check_inventory_wood_pickaxe(env):
        logger.info('Made wood pickaxe')
    else:
        logger.info('Could not make wood pickaxe')


def act_MAKE_STONE_PICKAXE(env):
    action = Action.MAKE_STONE_PICKAXE
    executor(env, [action])
    if check_inventory_stone_pickaxe(env):
        logger.info('Made stone pickaxe')
    else:
        logger.info('Could not make stone pickaxe')


def act_MAKE_IRON_PICKAXE(env):
    action = Action.MAKE_IRON_PICKAXE
    executor(env, [action])
    if check_inventory_iron_pickaxe(env):
        logger.info('Made iron pickaxe')
    else:
        logger.info('Could not make iron pickaxe')


def act_MAKE_WOOD_SWORD(env):
    action = Action.MAKE_WOOD_SWORD
    executor(env, [action])
    if check_inventory_wood_sword(env):
        logger.info('Made wood sword')
    else:
        logger.info('Could not make wood sword')


def act_MAKE_STONE_SWORD(env):
    action = Action.MAKE_STONE_SWORD
    executor(env, [action])
    if check_inventory_stone_sword(env):
        logger.info('Made stone sword')
    else:
        logger.info('Could not make stone sword')


def act_MAKE_IRON_SWORD(env):
    action = Action.MAKE_IRON_SWORD
    executor(env, [action])
    if check_inventory_iron_sword(env):
        logger.info('Made iron sword')
    else:
        logger.info('Could not make iron sword')


def act_MAKE_DIAMOND_PICKAXE(env):
    action = Action.MAKE_DIAMOND_PICKAXE
    executor(env, [action])
    if check_inventory_diamond_pickaxe(env):
        logger.info('Made diamond pickaxe')
    else:
        logger.info('Could not make diamond pickaxe')


def act_MAKE_DIAMOND_SWORD(env):
    action = Action.MAKE_DIAMOND_SWORD
    executor(env, [action])
    if check_inventory_diamond_sword(env):
        logger.info('Made diamond sword')
    else:
        logger.info('Could not make diamond sword')


def act_MAKE_IRON_ARMOUR(env):
    action = Action.MAKE_IRON_ARMOUR
    executor(env, [action])


def act_MAKE_DIAMOND_ARMOUR(env):
    action = Action.MAKE_DIAMOND_ARMOUR
    executor(env, [action])
