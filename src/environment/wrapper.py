import jax
import numpy as np
from gym import Wrapper
import craftax.craftax.renderer as renderer
from PIL import Image

class SaveStateWrapper(Wrapper):
    def render(self, state):
        return Image.fromarray(np.array(renderer.render_craftax_pixels(state, 16), dtype=np.uint8))

    def reset(self, *args, seed = None, **kwargs):
        if seed is not None:
            self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        obs, state = self.env.reset(self.rng)
        self.images = [self.render(state)]
        self.saved_state = state
        self.action_count = 0
        return obs, state

    def step(self, action):
        self.rng, subkey = jax.random.split(self.rng)
        obs, state, reward, done, info = self.env.step(subkey, self.saved_state, action.value, self.env.default_params)
        self.images.append(self.render(state))
        self.saved_state = state
        self.action_count += 1
        return obs, state, reward, done, info

    def __init__(self, env, seed, log_dir, max_steps=100):
        super().__init__(env)
        self.env = env
        self.seed = seed
        # self.log_dir = 'logs/actions.csv'
        self.log_dir = log_dir
        self.action_count: int = 0
        self.max_steps: int = max_steps

# class SaveStateWrapper(gym.Wrapper):
#     def __init__(self, env, seed=0xBAD_5EED, log_dir: str = '../../logs/'):
#         super().__init__(env)
#         # self.env = env
#         self.rng: jax.random.KeyArray = jax.random.PRNGKey(seed)
#         self.log_dir: str = log_dir
#         self.env_params = self.env.default_params
#
#         os.makedirs(self.log_dir, exist_ok=True)
#
#         self.saved_state: EnvState = None
#
#     def reset(self, rng = None, env_params=None) -> tuple[jax.numpy.ndarray, EnvState]:
#         if env_params is None: env_params = self.env_params
#         if rng is None: rng = self.rng
#
#         obs: jax.numpy.ndarray
#         state: EnvState
#         obs, state = self.env.reset(rng, env_params)
#         self.saved_state = state
#         return obs, state
#
#     def step(self, action: int, state=None):
#         if state is None: state = self.saved_state
#
#         rng, self.rng = jax.random.split(self.rng)
#         obs, state, reward, done, info = self.env.step(self.rng, state, action, self.env_params)
#         self.saved_state = state
#         return obs, state, reward, done, info
