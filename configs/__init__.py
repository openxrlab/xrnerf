import importlib


def load_configs(name):
    modellib = importlib.import_module(name)
    # print(configs.hmr_configs)
    return modellib

# load_configs("train_configs")
