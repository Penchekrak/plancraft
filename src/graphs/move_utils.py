import jax
import jax.numpy as jnp
import networkx as nx
from typing import Optional

from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import OBS_DIM, BlockType

def to_node(pos: jax.numpy.ndarray):
    return pos[0].item(), pos[1].item()


def get_obs_mask(state: EnvState):
    """
    Generates a mask of the observation area around the player.

    Args:
    state: The current state of the environment.

    Returns:
    A 3D array of shape (floors, height, width) where values are 1 if the block is in the observation area and 0 otherwise.
    """
    y, x = state.player_position
    floors, h, w = state.map.shape
    mask = jnp.zeros_like(state.map, dtype=jnp.int32)
    mask = mask.at[
           state.player_level,
           max(0, y - OBS_DIM[0] // 2) : min(h, y + OBS_DIM[0] // 2 + 1),
           max(0, x - OBS_DIM[1] // 2) : min(w, x + OBS_DIM[1] // 2 + 1)
           ].set(1)

    return mask


def is_in_obs(state: EnvState, pos: jax.Array, mask = None, level=None):
    """
    Checks if a given position is in the observation area of the player in the given state.

    Args:
        state: The current state of the environment.
        pos: The 2D coordinates of the position to check.
        mask: Optionally, a precomputed mask of the observation area. If not given, `get_obs_mask(state)` is used.
        level: Optionally, the level to check. If not given, `state.player_level` is used.

    Returns:
        True if the block at `pos` is in the observation area, False otherwise.
    """
    mask = mask if mask is not None else get_obs_mask(state)
    level = level if level is not None else state.player_level
    return mask[level, pos[0], pos[1]]


def find_block_any(
        state: EnvState,
        block_type: BlockType,
        level: Optional[int] = None
):
    """
    Finds the coordinates of any block with the given ID in the observation area of the given state.

    Args:
        state: The current state of the environment.
        block_type: The BlockValue of the block to find.
        level: Optionally, the level to search. If not given, `state.player_level` is used.

    Returns:
        The 2D coordinates of the block if found, or None if no such block is in the observation area.
    """
    res = find_block_all(state, block_type, level)

    if res.shape[0] == 0:
        return None
    return res[0]

def find_block_all(
        state: EnvState,
        block_type: BlockType,
        level: Optional[int] = None
) -> jax.numpy.ndarray:
    """
    Finds all blocks with the given ID in the observation area of the given state.

    Args:
        state: The current state of the environment.
        block_type: The BlockType of the block to find.
        level: Optionally, the level to search. If not given, `state.player_level` is used.

    Returns:
        A 2D array of shape (num_blocks, 2) where the first column is the x-coordinate
        and the second column is the y-coordinate.
    """
    level = level if level is not None else state.player_level
    obs_mask = get_obs_mask(state)[level]

    res = jax.numpy.where(
        jax.numpy.logical_and(
            obs_mask,
            state.map[level] == block_type.value
        )
    )

    return jnp.stack(res, axis=1)


def explore_choose_node(env, G: nx.Graph, prev_pos: jax.numpy.ndarray = None, dist = 5):
    state: EnvState = env.saved_state
    if not to_node(prev_pos) in G.nodes: return None
    nodes = jax.numpy.array(list(G.nodes), dtype=jax.numpy.int32)

    direction_vectors = nodes - state.player_position
    distances = jax.numpy.sum(jax.numpy.abs(direction_vectors), axis=1)

    if prev_pos is not None:
        prev_direction = state.player_position - prev_pos[:, jax.numpy.newaxis]

        dot_products = direction_vectors.dot(prev_direction).sum(axis=-1)

        if jax.numpy.any(dot_products >= 0):
            direction_mask = dot_products >= 0
        else:
            direction_mask = dot_products < 0
    else:
        direction_mask = jax.numpy.ones(direction_vectors.shape[0], dtype=bool)

    for cool_distance in range(dist, -1, -1):
        indexes = jax.numpy.where(jax.numpy.logical_and(distances == cool_distance, direction_mask))[0]
        if len(indexes) == 0: continue
        env.rng, rng = jax.random.split(env.rng)
        idx = jax.random.choice(rng, indexes)
        return nodes[idx]
    return None
