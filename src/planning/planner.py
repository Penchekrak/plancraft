import ast
import logging
import re
import sys
from typing import Tuple, Type

from craftax.craftax_env import make_craftax_env_from_name

from environment.wrapper import SaveStateWrapper
from src.primitives.checks import *
from src.environment.code_parser import find_most_function_calls
from src.planning.llm_caller import LLMCaller

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('MAIN_PLANNING_LOGGER')

TASKS = [
    'Collect wood',
    'Make wood pickaxe',
    'Collect stone',
    'Make stone pickaxe',
    'Collect coal',
    'Collect iron',
    'Make iron pickaxe',
    'Collect diamond'
]

CHECKERS = [
    check_inventory_wood,
    check_inventory_wood_pickaxe,
    check_inventory_stone,
    check_inventory_stone_pickaxe,
    check_inventory_coal,
    check_inventory_iron,
    check_inventory_iron_pickaxe,
    check_inventory_diamond
]
TASK_CHECKER = {
    task: checker
    for task, checker in zip(TASKS, CHECKERS)
}


def valid(code: str) -> tuple[bool, Type[SyntaxError] | None]:
    try:
        ast.parse(code)
    except SyntaxError:
        return False, SyntaxError
    return True, None

class Planner:
    def __init__(self, task: str, api_key: str, main_seed=42):
        self.task = task
        self.api_key = api_key
        self.llm_caller = LLMCaller(api_key=api_key, task=task)
        self.main_seed = main_seed
        self.task_checker = TASK_CHECKER[task]

    def play_code(self, code, env) -> str | None:
        try:
            new_symbols = {}
            exec(code, globals(), new_symbols)  # define all generated functions

            func_name, _ = find_most_function_calls(code, set(new_symbols.keys()))
            exec(f"{func_name}(env)", globals(), locals() | new_symbols)
        except Exception as error_feedback:
            return str(error_feedback)

        return None


    def parse_code_from_response(self, full_response) -> Tuple[str | None, bool]:
        code = '\n' + re.split('```python\n|```', full_response)[1]

        code_is_valid, error = valid(code)
        if not code_is_valid:
            return str(error), False
        return code, True


    def get_plan(self, env, num_tries_replanning=3):
        logger.info('Generating plan...')
        self.llm_caller.reset()

        # создаем среду, чтобы прокинуть в промпт obs
        env.reset(self.main_seed)
        starting_prompt = self.llm_caller.get_full_starting_prompt(env)

        # получаем первичный код
        logger.info('Calling LLM...')
        response = self.llm_caller.call_llm(starting_prompt)
        logger.info('validating code...')
        code, code_is_valid = self.parse_code_from_response(response)

        replan_prompt = ''
        if not code_is_valid:
            # если первичный код сгенерирован с ошибкой
            logger.info(f'Первичный код сгенерирован с ошибкой: {code}')
            replan_prompt = self.llm_caller.get_error_prompt(code)
        else:
            # играем первую траекторию
            logger.info(f'Играем первый промпт: {code}\n\n')
            play_result = self.play_code(code, env)
            if play_result is not None:
                # если первичный код падает с ошибкой
                logger.info(f'Первичный код падает с ошибкой: {play_result}')
                replan_prompt = self.llm_caller.get_error_prompt(play_result)
            elif not self.task_checker(env):
                # если первичный код не выполняет задачу
                logger.info('Первичный код не выполняет задачу')
                replan_prompt = self.llm_caller.get_debug_prompt()
            else:
                logger.info('Первичный код выполняет задачу')
                return code  # код сразу сгенерирован удачно


        # иначе начинаем цикл репланнинга
        for planning_iteration in range(num_tries_replanning):
            env.reset(self.main_seed)

            # получаем новый код
            logger.info(f'Попытка {planning_iteration + 1} из {num_tries_replanning}')
            logger.info(f'Перепланнируем код...\n')
            response = self.llm_caller.call_llm(replan_prompt)
            code, code_is_valid = self.parse_code_from_response(response)

            # если код невалидный
            if not code_is_valid:
                # если первичный код сгенерирован с ошибкой
                logger.info(f'Первичный код сгенерирован с ошибкой: {code}')
                replan_prompt = self.llm_caller.get_error_prompt(code)
                continue
            else:
                # играем траекторию
                logger.info(f'Играем код: {code}\n\n')
                play_result = self.play_code(code, env)
                if play_result is not None:
                    # если код падает с ошибкой
                    logger.info(f'Код падает с ошибкой: {play_result}')
                    replan_prompt = self.llm_caller.get_error_prompt(play_result)
                elif not self.task_checker(env):
                    # если код не выполняет задачу
                    logger.info('Код не выполняет задачу')
                    replan_prompt = self.llm_caller.get_debug_prompt()
                else:
                    logger.info('Код выполняет задачу')
                    return code  # код сгенерирован удачно

        print('число попыток превышено.')
        return None

    def estimate_success_rate(self, env, code, num_iter=10):
        logger.info('Estimating success rate...')
        results = []
        for iteration in range(num_iter):
            seed = self.main_seed + iteration
            env.reset(seed)

            logger.info(f'Estimating success rate: {iteration + 1} из {num_iter}')
            result = self.play_code(code, env)
            if result is None:
                logger.info('Failed to play code')
                results.append(self.task_checker(env))
            else:
                logger.info(f'result: {result}')
                results.append(0)
        return (1. * sum(results)) / len(results)


if __name__ == '__main__':
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = SaveStateWrapper(env, seed=42, log_dir='../../logs/')
    obs, state = env.reset(seed=42)

    planner = Planner(task='Make wood pickaxe',
                      api_key='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')
    code = planner.get_plan(env)
    if code is not None:
        print('success rate: ', planner.estimate_success_rate(env, code))
