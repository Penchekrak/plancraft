import unittest
import gym

import unittest
import gym
import jax
import craftax
from craftax.craftax.game_logic import craftax_step
from craftax.craftax_env import make_craftax_env_from_name

from primitives.utils import *


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Create the environment
        self.rng = jax.random.PRNGKey(0xBAD_5EED_B00B5)
        self.rng, _rng = jax.random.split(self.rng)
        self.rngs = jax.random.split(_rng, 3)

        # Create environment
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.env_params = self.env.default_params

    def one_step(self, action):
        obs, state = self.env.reset(self.rngs[0], self.env_params)
        return self.env.step(self.rngs[2], state, action, self.env_params)

    def get_random_action(self):
        return self.env.action_space(self.env_params).sample(self.rngs[1])



    def test_environment_observation(self):
        action = self.get_random_action()
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)
        res = get_block_offset_any(obs, craftax.craftax.game_logic.BlockType.NECROMANCER.value)
        self.assertIsNone(res)

    def test_get_global_from_local(self):
        action = self.get_random_action()
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)

        local_pos = get_block_offset_any(obs, craftax.craftax.game_logic.BlockType.TREE.value)
        res = get_global_from_local(state, local_pos)
        global_blocks = state.map[state.player_level]
        self.assertEqual(global_blocks[res[0], res[1]], craftax.craftax.game_logic.BlockType.TREE.value)


if __name__ == '__main__':
    unittest.main()