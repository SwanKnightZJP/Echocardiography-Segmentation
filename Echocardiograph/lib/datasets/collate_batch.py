import torch
from torch.utils.data.dataloader import default_collate


def defined_collator(batch):
    ret = batch
    return ret


_collators = {
    'defined': defined_collator
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate
