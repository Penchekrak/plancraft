from queue import Queue

import jax
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import DIRECTIONS, COLLISION_LAND_CREATURE, OBS_DIM, Action
from craftax.craftax.game_logic import is_position_in_bounds_not_in_mob_not_colliding

from primitives.utils import get_obs_mask, is_in_obs
import networkx as nx

DIRECTIONS_TO_ACTIONS = {
    (0, 0): Action.NOOP,
    (0, -1): Action.LEFT,
    (0, 1): Action.RIGHT,
    (-1, 0): Action.UP,
    (1, 0): Action.DOWN
}

def to_node(pos: jax.numpy.ndarray):
    return pos[0].item(), pos[1].item()


def valid_block(state, pos, mask=None):
    """
    Checks if a given position is valid in the current state.
    A position is valid if it is within the bounds of the current level,
    is not a mob, and is in the observation area of the player.

    Args:
        state: The current state of the environment.
        pos: The 2D coordinates of the position to check.
        mask: Optionally, a precomputed mask of the observation area. If not given, `get_obs_mask(state)` is used.

    Returns:
        True if the block at `pos` is valid, False otherwise.
    """
    if mask is None: mask = get_obs_mask(state)
    return (is_position_in_bounds_not_in_mob_not_colliding(state, pos, COLLISION_LAND_CREATURE) and
            is_in_obs(state, pos, mask))


def gen_graph(state: EnvState):
    mask = get_obs_mask(state)
    start_pos = state.player_position
    start_node = to_node(start_pos)

    nodes = set()
    edges = set()
    q = Queue()
    q.put(start_pos)
    nodes.add(start_node)

    while not q.empty():
        cur = q.get()
        cur_node = to_node(cur)

        for direction in DIRECTIONS[1:5]:
            neighbor = cur + direction
            neighbor_node = to_node(neighbor)
            if neighbor_node in nodes: continue
            nodes.add(neighbor_node)
            edges.add((cur_node, neighbor_node))
            if valid_block(state, neighbor, mask):
                q.put(neighbor)

    G = nx.Graph(nodes)
    G.add_edges_from(edges)
    return G

def move_to_pos(state: EnvState, target_pos: jax.numpy.ndarray):
    """
    Generates a sequence of actions to move to a target position.
    Graph is generated.

    Args:
        state: The current state of the environment.
        target_pos: The 2D coordinates of the target position.

    Returns:
        A list of actions to move to the target position.
    """
    G = gen_graph(state)
    if not nx.has_path(G, to_node(state.player_position), to_node(target_pos)): return []

    nodes = nx.astar_path(G, source=to_node(state.player_position), target=to_node(target_pos))

    actions = []
    for i in range(len(nodes) - 1):
        actions.append(DIRECTIONS_TO_ACTIONS[
                           (nodes[i + 1][0] - nodes[i][0],
                            nodes[i + 1][1] - nodes[i][1])
                       ])
    return actions


def move_to_node(state: EnvState, G: nx.Graph, target_node: jax.numpy.ndarray):
    if not nx.has_path(G, to_node(state.player_position), target_node): return []

    nodes = nx.astar_path(G, source=to_node(state.player_position), target=target_node)

    actions = []
    for i in range(len(nodes) - 1):
        actions.append(DIRECTIONS_TO_ACTIONS[
                           (nodes[i + 1][0] - nodes[i][0],
                            nodes[i + 1][1] - nodes[i][1])
                       ])
    return actions
