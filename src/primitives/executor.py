from craftax.craftax.constants import Action

def executor(env, action_list: list[Action]):
    # append acton_list to actions.csv
    with open(env.log_file, 'a+') as f:
        for action in action_list:
            print(type(action), action)
            f.write(f'{action.value}\n')

    # execute actions
    for action in action_list:
        obs, state, reward, done, info = env.step(action)