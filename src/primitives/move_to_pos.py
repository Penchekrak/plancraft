from queue import Queue

import jax
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import DIRECTIONS, COLLISION_LAND_CREATURE, OBS_DIM
from craftax.craftax.game_logic import is_position_in_bounds_not_in_mob_not_colliding

from primitives.utils import get_obs_mask, is_in_obs
import networkx as nx

DIRECTIONS_TO_ACTIONS = {
    (0, 0): 0,
    (0, -1): 1,
    (0, 1): 2,
    (-1, 0): 3,
    (1, 0): 4
}


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


def gen_graph(state: EnvState, target_pos: jax.numpy.ndarray):
    """
    Generates a graph of the positions reachable from the player in the current state.
    The graph has nodes for each position reachable from the player and edges between
    positions that are adjacent. The start node is the player's current position and the
    target node is the given target position. Target node is included even if it is not
    valid.

    Args:
        state: The current state of the environment.
        target_pos: The 2D coordinates of the target position.

    Returns:
        A networkx Graph of the positions reachable from the player.
    """
    mask = get_obs_mask(state)
    start_pos = state.player_position
    start_node = (start_pos[0].item(), start_pos[1].item())
    target_node = (target_pos[0].item(), target_pos[1].item())

    nodes = set()
    edges = set()
    q = Queue()
    q.put(start_pos)
    nodes.add(start_node)

    while not q.empty():
        cur = q.get()
        cur_node = (cur[0].item(), cur[1].item())

        for direction in DIRECTIONS[1:5]:
            neighbor = cur + direction
            neighbor_node = (neighbor[0].item(), neighbor[1].item())
            if neighbor_node in nodes: continue
            if valid_block(state, neighbor, mask):
                nodes.add(neighbor_node)
                edges.add((cur_node, neighbor_node))
                q.put(neighbor)
            elif neighbor_node == target_node:
                nodes.add(neighbor_node)
                edges.add((cur_node, neighbor_node))

    G = nx.Graph(nodes)
    G.add_edges_from(edges)
    return G

def move_to_pos(state: EnvState, target_pos: jax.numpy.ndarray):
    """
    Generates a sequence of actions to move to a target position.

    Args:
        state: The current state of the environment.
        target_pos: The 2D coordinates of the target position.

    Returns:
        A list of actions to move to the target position.
    """
    G = gen_graph(state, target_pos)
    nodes = nx.astar_path(G, (state.player_position[0].item(), state.player_position[1].item()),
                            (target_pos[0].item(), target_pos[1].item()))

    actions = []
    for i in range(len(nodes) - 1):
        actions.append(DIRECTIONS_TO_ACTIONS[
                           (nodes[i + 1][0] - nodes[i][0],
                            nodes[i + 1][1] - nodes[i][1])
                       ])
    return actions
