"""

    To adjust the default sampler from pytorch

"""

from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
import math
import torch.distributed as dist


# class ImageSizeBatchSampler(Sampler):
#     def __int__(self):
#
#     def __iter__(self):
#
#     def __len__(self):
#
#
# class IterationBasedBatchSampler(BatchSampler):
#     def __int__(self):
#
#     def __iter__(self):
#
#     def __len__(self):