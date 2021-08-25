import os
import importlib.machinery as Fun_imp


def make_network(cfg, is_train):
    module = '.'.join(['lib.net', cfg.task])
    path = os.path.join('lib/net', cfg.task, '__init__.py')
    return Fun_imp.SourceFileLoader(module, path).load_module().get_network(cfg, is_train)

