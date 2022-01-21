from importlib import import_module
import json


_STRATEGY_CFG_DIR = "config/test/strategy/"

def load(module_name: str, class_name: str, args=None):
    module = import_module(f"{module_name}.{class_name.lower()}")
    class_loaded = getattr(module, f"{class_name}{module_name.capitalize()}")
    if args:
        instance = class_loaded(args)
    else:
        instance = class_loaded()

    return instance

def load_strategy(strategy_name: str):
    strategy_args = load_json(f"{_STRATEGY_CFG_DIR + strategy_name.lower()}.json")
    strategy = load("strategy", strategy_name, strategy_args)
    return strategy


def load_json(json_file_path: str):
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    return json_data

