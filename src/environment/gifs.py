import os
import tqdm
import numpy as np
import jax
from PIL import Image
from craftax.craftax.constants import Action

from src.environment.wrapper import SaveStateWrapper


def process_environment_step(env: SaveStateWrapper, renderer, action, img_array):
    """Processes an environment step and adds the rendered image to img_array."""
    obs, state, reward, done, info = env.step(action)

    image_array = renderer.render_craftax_pixels(state, 16)  # Assuming '16' is pixel size
    img = Image.fromarray(np.array(image_array, dtype=np.uint8))
    img_array.append(img)
    return obs, state, reward, done, info


def create_gif_grid(gif_arrays, save_path, grid_size, file_name="grid_output.gif"):
    """Creates a video"""
    def write_video(file_name, images, slide_time=5, FPS=10):
        shape = images[0].size
        fourcc = cv2.VideoWriter.fourcc(*'MJPG')
        out = cv2.VideoWriter(file_name, fourcc, FPS, shape)
    
        for image in images:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for _ in range(slide_time * FPS):
                cv_img = cv2.resize(image, shape)
                out.write(cv_img)
    
        out.release()

    # Save the grid as a GIF
    gif_save_path = f"{save_path}"
    os.makedirs(gif_save_path, exist_ok=True)
    write_video(os.path.join(gif_save_path, file_name), gif_arrays[0])


def visual_testing(env, file_path: str, path_to_save: str, num_tries, renderer, grid_size=(2, 2),
                   gif_name="grid_output.gif"):
    """Tests a function in the environment and generates a GIF grid from the steps."""
    # all_gif_arrays = []
    # print('started visual testing')
    # img_array = []
    # try:
    #     done = False
    #
    #     # actions = function_to_test(state, **function_parameters)
    #     if not os.path.exists(file_path):
    #         raise ValueError(f"No such file in given path: {file_path}")
    #
    #     with open(file_path, 'r') as f:
    #         actions = []
    #         for line in f:
    #             actions.append(Action(int(line.strip())))
    #
    #     # assert len(actions) != 0, 'No actions found for this test function'
    #
    #     i = 0
    #     while not done:
    #         # action = function_to_test(obs, **function_parameters)
    #         obs, state, reward, done, info = process_environment_step(
    #             env, renderer, actions[i], img_array
    #         )
    #
    #         i += 1
    #         if i >= len(actions):
    #             done = True
    #
    # except Exception as e:
    #     print(f"Error during testing: {e}")
    #     return
    #
    # # Store each gif array to be used for grid creation later
    # all_gif_arrays.append(img_array)

    # Generate the grid of GIFs
    # print(path_to_save)
    create_gif_grid([env.images], path_to_save.replace('.gif', '.mp4'), grid_size, gif_name)
