def collect_stone(env):
    # check how many wood we have in inventory
    wood = check_inventory_wood(env)
    # ensure that we have enough wood in inventory
    # we want to obtain 2 wood for crafting table
    # AND 1 wood for crafting pickaxe
    required_wood = 3
    if wood < required_wood:
        mine_block(env, BlockType.TREE, required_wood - wood)
    # place a crafting table in front of us
    act_PLACE_TABLE(env)
    # craft wood pickaxe
    act_MAKE_WOOD_PICKAXE(env)
    # mine stone
    mine_block(env, BlockType.STONE)

def create_wood_pickaxe(env):
    # check how many wood we have in inventory
    wood = check_inventory_wood(env)
    # ensure that we have enough wood in inventory
    # we want to obtain 2 wood for crafting table
    # AND 1 wood for crafting pickaxe
    required_wood = 3
    if wood < required_wood:
        mine_block(env, BlockType.TREE, required_wood - wood)
    # place a crafting table in front of us
    act_PLACE_TABLE(env)
    # craft wood pickaxe
    act_MAKE_WOOD_PICKAXE(env)