import jax
import networkx as nx
import numpy as np
from craftax.craftax.craftax_state import EnvState


from primitives.move_to_pos import DIRECTIONS_TO_ACTIONS, to_node


def explore_round(state: EnvState, G: nx.Graph, prev_pos: jax.numpy.ndarray, dist = 5):

    """
    Choose a node in the graph to move to.

    Args:
    state: The current state of the environment.
    G: The graph of the environment.
    prev_pos: The position that the player is currently at.
    dist: The maximum distance to move.

    Returns:
    The node to move to as an numpy array of shape (2,).
    """
    if not to_node(prev_pos) in G.nodes: return None
    nodes = jax.numpy.array(list(G.nodes), dtype=jax.numpy.int32)

    direction_vectors = nodes - state.player_position
    prev_direction = state.player_position - prev_pos[:, jax.numpy.newaxis]
    dot_products = direction_vectors.dot(prev_direction)

    # Manhattan distance
    distances = jax.numpy.sum(jax.numpy.abs(direction_vectors), axis=1)

    if jax.numpy.any(dot_products >= 0):
        direction_mask = dot_products >= 0
    else:
        direction_mask = dot_products < 0

    for cool_distance in range(dist, 0, -1):
        indexes = jax.numpy.where(jax.numpy.logical_and(distances == cool_distance, direction_mask))
        if len(indexes) == 0: continue
        return np.random.choice(indexes)
    return None
