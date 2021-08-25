import torch
# import imp
import importlib.machinery as Fun_imp
import os
import torch.utils.data
from . import samplers
import torch.multiprocessing

from .transforms import make_transforms
from .dataset_catalog import DatasetCatalog


torch.multiprocessing.set_sharing_strategy('file_system')


def _data_factory(data_source, dataBuilder):
    module = '.'.join(['lib.datasets', data_source, dataBuilder])
    path = os.path.join('lib/datasets', data_source, dataBuilder+'.py')
    # dataset_C = imp.load_source(module, path).Dataset
    dataset_C = Fun_imp.SourceFileLoader(module, path).load_module().Dataset
    return dataset_C


def make_dataset(cfg, dataset_name, transform, is_train=True):
    args = DatasetCatalog.get(dataset_name)

    dataBuilder = cfg.data.Constractioner
    data_source = args['id']

    if dataBuilder == 'RandomPatches_xyz' or dataBuilder == 'RandomPatches_xy':
        img_dir = args['enhanced_dir'] if cfg.data.img_type == 'E' else args['img_dir']
        label_dir = args['label_f_dir']
        mini_img = args['mini_img']
        mini_label = args['mini_label']
        del args['img_dir']
        del args['enhanced_dir']
        del args['label_f_dir']
        del args['mini_img']
        del args['mini_label']

    elif dataBuilder == 'Total_img':        # 06 27 added
        img_dir = args['enh_dir'] if cfg.data.img_type == 'E' else args['ori_dir']
        label_dir = args['label_o_dir']
        mini_img = args['mini_img']
        mini_label = args['mini_label']
        del args['img_dir']
        del args['enhanced_dir']
        del args['label_o_dir']
        del args['mini_img']
        del args['mini_label']

    elif dataBuilder == 'RandomPatch_DTM':  # 06 30 added
        img_dir = args['ori_dir']
        label_dir = args['label_dir']
        mini_img = args['mini_img']
        mini_label = args['mini_label']
        del args['ori_dir']
        del args['label_dir']
        del args['mini_img']
        del args['mini_label']

    else:  # -------------------- need to be adjust ------------------------------------------------------------
        img_dir = args['seq_enhanced_dir'] if cfg.data.img_type == 'E' else args['seq_img_dir']
        label_dir = args['seq_label_f_dir'] if cfg.data.label_type == 'FLOW' else args['seq_label_s_dir']
        mini_img = args['mini_img']
        mini_label = args['mini_label']
        del args['seq_img_dir']
        del args['seq_enhanced_dir']
        del args['seq_label_f_dir']
        del args['seq_label_s_dir']

    dataset_C = _data_factory(data_source, dataBuilder)
    #  --- to instantiate lib.datasets.data_source.task.py// Dataset
    dataset = dataset_C(cfg, transform, img_dir, label_dir, is_train, mini_img, mini_label)
    del args['id']

    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:  # to randomly sort the data in the dataset
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter):
    # to combing the data according to the size of the batch
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset
    transform = make_transforms(cfg, is_train)
    dataset = make_dataset(cfg, dataset_name, transform, is_train)

    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter)

    num_workers = cfg.train.num_workers
    # collator = make_collator(cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,  # batch_size etc.
        num_workers=num_workers,
        # collate_fn=collator           # if memory is enough -- set pin_memory=True
    )

    return data_loader
