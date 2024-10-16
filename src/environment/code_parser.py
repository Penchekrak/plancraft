import ast


class FunctionCallCounter(ast.NodeVisitor):
    def __init__(self, function_set):
        self.function_calls = {}
        self.current_function = None
        self.function_set = function_set

    def visit_FunctionDef(self, node):
        if node.name in self.function_set:
            self.current_function = node.name
            self.function_calls[self.current_function] = 0
            self.generic_visit(node)
            self.current_function = None
        else:
            pass

    def visit_Call(self, node):
        if self.current_function and isinstance(node.func, ast.Name):
            if node.func.id in self.function_set:
                self.function_calls[self.current_function] += 1
        self.generic_visit(node)


def find_most_function_calls(code_str, function_set):
    tree = ast.parse(code_str)

    counter = FunctionCallCounter(function_set)
    counter.visit(tree)

    if counter.function_calls:
        most_called_func = max(counter.function_calls, key=counter.function_calls.get)
        return most_called_func, counter.function_calls[most_called_func]
    else:
        return None, 0
