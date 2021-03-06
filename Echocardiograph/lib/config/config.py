from .yacs import CfgNode as CN  # lib.config/config.py
import argparse
import os

# ------------------- default init -------------------- #
cfg = CN()

# model
cfg.model = 'hello'           # init
cfg.model_dir = '../model'  # define the dir of the model

# network
cfg.network = 'ResUnet'        # def the name of the network

# network heads
cfg.heads = CN()

# task
cfg.task = 'default'

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True

cfg.dpr = 0.0  # dropout rate

# ---------------------------------------------------------------
# train
# ---------------------------------------------------------------

cfg.train = CN()

cfg.train.dataset = 'default_train_data_folder'
cfg.train.epoch = 300
cfg.train.num_workers = 4

# optimizer

cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# ---------------------------------------------------------------
# test
# ---------------------------------------------------------------

cfg.test = CN()
cfg.test.dataset = 'default_test_data_folder'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# ---------------------------------------------------------------
# data
# ---------------------------------------------------------------

cfg.data = CN()
cfg.data.img_type = ' '        # 'E': enhance, 'O': original
cfg.data.label_type = ' '
cfg.data.DataTransType = ' '
cfg.data.Act_type = ' '
cfg.data.Constractioner = ' '
cfg.data.patch_size = (112, 112, 112)
cfg.data.clip_size = (112, 112, 112)


# ------------------ recorder ------------------ #
cfg.record_dir = 'data/record'

# ------------------  result  ------------------ #
cfg.result_dir = 'data/result'

# ----------------- evaluation ----------------- #
cfg.skip_eval = False
cfg.save_ep = 5
cfg.demo_ep = 5

cfg.demo_path = ''


# -------------- build the cfg_file ------------ #
def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.data_type, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.data_type, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
# parser.add_argument("--cfg_file", default='/media/zdc/zjp/Projects/Echocardiography/configs/BScan04.yaml', type=str)
parser.add_argument("--cfg_file", default='configs/BScan/BScan32.yaml', type=str)
# parser.add_argument("--cfg_file", default='configs/OrganSeg/OrganSeg01.yaml', type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)







