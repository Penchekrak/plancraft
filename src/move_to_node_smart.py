from queue import Queue

import jax
import networkx as nx
from craftax.craftax.constants import DIRECTIONS, BlockType, OBS_DIM, Action
from craftax.craftax.craftax_state import EnvState

from primitives.move_to_pos import to_node, DIRECTIONS_TO_ACTIONS
from primitives.utils import get_obs_mask, is_in_obs

INF_WEIGHT = 10**6
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


def gen_graph_smart(state: EnvState) -> nx.DiGraph:
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
                neighbor_type = state.map[level][neighbor]

                if not is_in_obs(state, neighbor, mask, level):
                    continue
                G.add_node(neighbor_node, block_type=neighbor_type)
                G.add_edge(cur_node, neighbor_node, weight=BLOCK_WEIGHT[neighbor_type],
                           direction=direction)
    return G


def move_to_node_smart(state: EnvState, G: nx.DiGraph, target_node: tuple[int, int], last_step=True) -> list[Action]:
    if not target_node in G.nodes: return []

    nodes = nx.dijkstra_path(G, source=to_node(state.player_position), target=target_node)

    actions = []
    for i in range(len(nodes) - 1):
        cur_node, next_node = nodes[i], nodes[i + 1]
        if not last_step and i == len(nodes) - 1: break

        direction = tuple(G.edges[cur_node, next_node]['direction'].tolist())
        block_type = G.nodes[next_node]['block_type']

        if block_type in [BlockType.STONE.value,
                          BlockType.COAL.value,
                          BlockType.IRON.value,
                          BlockType.DIAMOND.value,
                          BlockType.CRAFTING_TABLE.value,
                          BlockType.FURNACE.value,
                          BlockType.WOOD.value,
                          BlockType.TREE.value]:
            actions.append(Action.DO.value)
        if block_type in [BlockType.WATER.value,
                          BlockType.LAVA.value]:
            actions.append(Action.PLACE_STONE.value)

        actions.append(DIRECTIONS_TO_ACTIONS(direction))
        
    return actions
