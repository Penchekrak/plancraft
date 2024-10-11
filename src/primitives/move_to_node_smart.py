import jax
import networkx as nx
from craftax.craftax.constants import DIRECTIONS, BlockType, OBS_DIM, Action
from craftax.craftax.craftax_state import EnvState

from .utils import get_obs_mask, is_in_obs
from .executor import executor

DIRECTIONS_TO_ACTIONS = {
    (0, 0): Action.NOOP,
    (0, 1): Action.LEFT,
    (0, -1): Action.RIGHT,
    (1, 0): Action.UP,
    (-1, 0): Action.DOWN
}

def to_node(pos: jax.numpy.ndarray):
    return pos[0].item(), pos[1].item()

INF_WEIGHT = 10 ** 6
BLOCK_WEIGHT = {
    BlockType.INVALID.value: INF_WEIGHT,
    BlockType.OUT_OF_BOUNDS.value: INF_WEIGHT,
    BlockType.GRASS.value: 1,
    BlockType.WATER.value: 15,
    BlockType.STONE.value: 10,
    BlockType.TREE.value: 10,
    BlockType.WOOD.value: 10,
    BlockType.PATH.value: 1,
    BlockType.COAL.value: 10,
    BlockType.IRON.value: 10,
    BlockType.DIAMOND.value: 10,
    BlockType.CRAFTING_TABLE.value: 15,
    BlockType.FURNACE.value: 15,
    BlockType.SAND.value: 1,
    BlockType.LAVA.value: 15,

    # пока не работаем с этим
    BlockType.PLANT.value: INF_WEIGHT,
    BlockType.RIPE_PLANT.value: INF_WEIGHT,
    BlockType.WALL.value: INF_WEIGHT,
    BlockType.DARKNESS.value: INF_WEIGHT,
    BlockType.WALL_MOSS.value: INF_WEIGHT,
    BlockType.STALAGMITE.value: INF_WEIGHT,
    BlockType.SAPPHIRE.value: INF_WEIGHT,
    BlockType.RUBY.value: INF_WEIGHT,
    BlockType.CHEST.value: INF_WEIGHT,
    BlockType.FOUNTAIN.value: INF_WEIGHT,
    BlockType.FIRE_GRASS.value: INF_WEIGHT,
    BlockType.ICE_GRASS.value: INF_WEIGHT,
    BlockType.GRAVEL.value: INF_WEIGHT,
    BlockType.FIRE_TREE.value: INF_WEIGHT,
    BlockType.ICE_SHRUB.value: INF_WEIGHT,
    BlockType.ENCHANTMENT_TABLE_FIRE.value: INF_WEIGHT,
    BlockType.ENCHANTMENT_TABLE_ICE.value: INF_WEIGHT,
    BlockType.NECROMANCER.value: INF_WEIGHT,
    BlockType.GRAVE.value: INF_WEIGHT,
    BlockType.GRAVE2.value: INF_WEIGHT,
    BlockType.GRAVE3.value: INF_WEIGHT,
    BlockType.NECROMANCER_VULNERABLE.value: INF_WEIGHT
}

NEED_DIG = [
    BlockType.STONE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.WOOD.value,
    BlockType.TREE.value
]

NEED_PLACE = [
    BlockType.WATER.value,
    BlockType.LAVA.value
]

def gen_graph_smart(state: EnvState,
                    can_dig=True,
                    can_place=True) -> nx.DiGraph:
    mask = get_obs_mask(state)
    start_pos = state.player_position
    level = state.player_level

    G = nx.DiGraph()
    G.add_node(to_node(start_pos), block_type=state.map[level][start_pos])

    for y_offset in range(-OBS_DIM[0] // 2, OBS_DIM[0] // 2 + 1):
        for x_offset in range(-OBS_DIM[1] // 2, OBS_DIM[1] // 2 + 1):
            cur_pos = start_pos + jax.numpy.array([y_offset, x_offset])
            cur_node = to_node(cur_pos)

            for direction in DIRECTIONS[1:5]:
                neighbor = cur_pos + direction
                neighbor_node = to_node(neighbor)
                neighbor_type = state.map[level][neighbor[0], neighbor[1]].item()

                if not is_in_obs(state, neighbor, mask, level):
                    continue
                G.add_node(neighbor_node, block_type=neighbor_type)
                weight = BLOCK_WEIGHT[neighbor_type]
                if neighbor_type in NEED_DIG and not can_dig:
                    continue
                if neighbor_type in NEED_PLACE and not can_place:
                    continue
                    
                G.add_edge(cur_node, neighbor_node, weight=weight, direction=direction)
    return G


def move_to_node_planner(state: EnvState, G: nx.DiGraph,
                       target_node: tuple[int, int], last_step=True) -> list[Action]:
    if not target_node in G.nodes: return []

    nodes = nx.dijkstra_path(G, source=to_node(state.player_position), target=target_node)

    actions = []
    for i in range(len(nodes) - 1):
        cur_node, next_node = nodes[i], nodes[i + 1]
        if not last_step and i == len(nodes) - 1: break

        direction: tuple[int, int] = G.edges[cur_node, next_node]["direction"].tolist()
        direction = tuple(direction)
        block_type = G.nodes[next_node]['block_type']

        if block_type in NEED_DIG:
            actions.append(DIRECTIONS_TO_ACTIONS[direction])
            actions.append(Action.DO)
        if block_type in NEED_PLACE:
            actions.append(DIRECTIONS_TO_ACTIONS[direction])
            actions.append(Action.PLACE_STONE)

        actions.append(DIRECTIONS_TO_ACTIONS[direction])
        
    return actions

def move_to_pos(env, target_pos: jax.numpy.ndarray, G: nx.DiGraph = None, can_dig=True, can_place=True):
    state = env.saved_state

    if G is None:
        G = gen_graph_smart(state, can_dig, can_place)
    target_node = to_node(target_pos)
    if not target_node in G.nodes:
        return
    act_plan = move_to_node_planner(env.saved_state, G, target_node)
    executor(env, act_plan)
