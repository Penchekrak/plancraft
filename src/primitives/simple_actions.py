from craftax.craftax.constants import Action

from .executor import executor

def act_DO(env):
    action = Action.DO.value
    executor(env, [action])

def act_PLACE_STONE(env):
    action = Action.PLACE_STONE.value
    executor(env, [action])

def act_PLACE_TABLE(env):
    action = Action.PLACE_TABLE.value
    executor(env, [action])

def act_PLACE_FURNACE(env):
    action = Action.PLACE_FURNACE.value
    executor(env, [action])

def act_PLACE_PLANT(env):
    action = Action.PLACE_PLANT.value
    executor(env, [action])

def act_MAKE_WOOD_PICKAXE(env):
    action = Action.MAKE_WOOD_PICKAXE.value
    executor(env, [action])

def act_MAKE_STONE_PICKAXE(env):
    action = Action.MAKE_STONE_PICKAXE.value
    executor(env, [action])

def act_MAKE_IRON_PICKAXE(env):
    action = Action.MAKE_IRON_PICKAXE.value
    executor(env, [action])

def act_MAKE_WOOD_SWORD(env):
    action = Action.MAKE_WOOD_SWORD.value
    executor(env, [action])

def act_MAKE_STONE_SWORD(env):
    action = Action.MAKE_STONE_SWORD.value
    executor(env, [action])

def act_MAKE_IRON_SWORD(env):
    action = Action.MAKE_IRON_SWORD.value
    executor(env, [action])

def act_MAKE_DIAMOND_PICKAXE(env):
    action = Action.MAKE_DIAMOND_PICKAXE.value
    executor(env, [action])

def act_MAKE_DIAMOND_SWORD(env):
    action = Action.MAKE_DIAMOND_SWORD.value
    executor(env, [action])

def act_MAKE_IRON_ARMOUR(env):
    action = Action.MAKE_IRON_ARMOUR.value
    executor(env, [action])

def act_MAKE_DIAMOND_ARMOUR(env):
    action = Action.MAKE_DIAMOND_ARMOUR.value
    executor(env, [action])