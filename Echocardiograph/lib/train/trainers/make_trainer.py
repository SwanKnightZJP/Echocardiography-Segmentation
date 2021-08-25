from .trainer import Trainer
import importlib.machinery as Fun_imp
# import imp
import os


def _wrapper_factory(cfg, network):
    module = '.'.join(['lib.train.trainers', cfg.task])
    # ## --- module = "lib.evaluators.DataID.TaskName" -- FunName_str
    path = os.path.join('lib/train/trainers', cfg.task+'.py')
    # ## --- path = "lib/evaluators/DataID/TaskName.py"  -- FunPth
    # network_wrapper = imp.load_source(module, path).NetworkWrapper(cfg, network)  # change the imp
    network_wrapper = Fun_imp.SourceFileLoader(module, path).load_module().NetworkWrapper(cfg, network)
    # ## --- instantiate FunPth.NetworkWrapper in FunName_str

    return network_wrapper


def make_trainer(cfg, network):
    train_stage = _wrapper_factory(cfg, network)
    return Trainer(train_stage)
