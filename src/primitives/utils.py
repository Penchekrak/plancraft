from typing import Optional, Tuple

import jax.numpy
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import SOLID_BLOCK_MAPPING, SOLID_BLOCKS
import jax.numpy as jnp
from jax import Array


def obs_blocks(obs):
    """
    Extracts the visible blocks from a given observation.

    The blocks are represented as a 2D array of shape (9, 11) where each entry is a
    block ID. The block IDs are taken from the constants in craftax.constants.

    Parameters
    ----------
    obs: Array
        The observation to extract the blocks from. This should be a JAX array of
        shape (8217,) or (9, 11, 83).

    Returns
    -------
    visible_blocks: Array
        The visible blocks as a 2D array of shape (9, 11).
    """
    visible_blocks = obs[0:8217].reshape(9, 11, 83)[:, :, 0:37].argmax(axis=-1)
    return visible_blocks


def get_global_from_local(state: EnvState, local: jax.numpy.ndarray) -> jax.numpy.ndarray:
    """
    Converts local coordinates to global coordinates.

    This function takes the current state of the environment and local coordinates
    as input, and returns the corresponding global coordinates.

    Parameters
    ----------
    state: EnvState
        The current state of the environment.
    local: jax.numpy.ndarray
        The local coordinates to be converted.

    Returns
    -------
    global_pos: jax.numpy.ndarray
        The global coordinates corresponding to the input local coordinates.
    """
    # Get the player's current position from the environment state
    player_pos = state.player_position

    # Calculate the local coordinates centered around the player's position
    local_centered = jnp.array([local[0] - 4, local[1] - 5])

    # Calculate the global coordinates by adding the player's position to the centered local coordinates
    global_pos = player_pos + local_centered

    return global_pos


def get_block_offset_any(
    obs: jax.numpy.ndarray, block_id: int
) -> jax.numpy.ndarray | None:
    """
    Given an observation and a block ID, return the coordinates of any block with that ID in the observation.
    Returns local coordinates if the block is found or None if it is not found.
    """
    res = jax.numpy.where(obs_blocks(obs) == block_id)
    if res[0].shape[0] == 0:
        return None
    return jnp.array([res[0][0], res[1][0]])
