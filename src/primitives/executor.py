from craftax.craftax.constants import Action

def executor(env, action_list: list[Action], log_file='../logs/actions.csv', seed=0xBAD_5EED_B00B5):
    # append acton_list to actions.csv
    with open(log_file, 'a') as f:
        for action in action_list:
            f.write(f'{action.value},{seed}\n')

    # execute actions
    for action in action_list:
        pass
        # TODO: work with seed