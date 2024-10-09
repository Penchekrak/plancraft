import unittest
import jax
from craftax.craftax_env import make_craftax_env_from_name

from primitives.utils import *


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Create the environment
        self.rng = jax.random.PRNGKey(0xBAD_5EED)
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

    def test_get_obs_mask(self):
        action = self.get_random_action()
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)

        # print(get_obs_mask(state)[state.player_level])
        mask = get_obs_mask(state)
        pos = state.player_position
        for i in range(-5, 6):
            for j in range(-6, 7):
                cur = pos + jax.numpy.array([i, j])
                # print(pos, cur, is_in_obs(state, cur, mask))
                if abs(i) != 5 and abs(j) != 6:
                    self.assertTrue(is_in_obs(state, cur, mask))
                else:
                    self.assertFalse(is_in_obs(state, cur, mask))
                self.assertFalse(is_in_obs(state, cur, mask, level=state.player_level + 1))

    def test_find_block_all(self):
        action = self.get_random_action()
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)

        self.assertTrue(find_block_all(state, 2).shape[1] == 2 and
                        find_block_all(state, 2).shape[0] > 0)

    # test for find_block_any
    def test_find_block_any(self):
        action = self.get_random_action()
        obs: jax.numpy.ndarray
        state: EnvState
        obs, state, reward, done, info = self.one_step(action)

        self.assertEqual(find_block_any(state, 2).shape, (2, ))



if __name__ == '__main__':
    unittest.main()