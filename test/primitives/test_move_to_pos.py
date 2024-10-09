import unittest
from time import sleep

import jax
import matplotlib.pyplot as plt
from craftax.craftax_env import make_craftax_env_from_name

from primitives.move_to_pos import move_to_pos
from primitives.utils import *
import craftax.craftax.renderer as renderer
import importlib
from PIL import Image
import numpy as np



class TestModeToPos(unittest.TestCase):
    def setUp(self):
        # Create the environment
        self.rng = jax.random.PRNGKey(0xBAD_5EED5)
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

    def test_move_to_pos(self):
        action = 0
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)

        print(*state.map[state.player_level, 19:30, 19:30].tolist(), sep="\n")

        target = state.player_position + jax.numpy.array([3, 3])
        actions = move_to_pos(state, target)
        print(actions)
        for action in actions:
            obs, state, reward, done, info = self.env.step(self.rngs[2], state, action, self.env_params)

        self.assertEqual(state.player_position.tolist(), target.tolist())


if __name__ == '__main__':
    unittest.main()