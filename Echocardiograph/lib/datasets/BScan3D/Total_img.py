"""

for the full size image

"""

import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset as dataset
import nibabel as nib
import numpy as np
import random
import gc
import math


class Dataset(dataset):
    def __init__(self, cfg, transform, img_dir, label_dir, is_train):
        self.transform = transform
        self.patch_size = cfg.data.patch_size
        self.clip_size = cfg.data.clip_size
        self.label_list = os.listdir(label_dir)
        self.label_dir_list = list(map(lambda x: os.path.join(label_dir, x), self.label_list))
        self.cfg = cfg
        self.train_tag = is_train

    def __getitem__(self, item):

        single_img_dir = self.label_dir_list[item]
        img_array = nib.load(single_img_dir).get_fdata()
        label_array = nib.load(self.label_dir_list[item]).get_fdata().astype(np.uint8)

        patch_img_array = img_array
        patch_label_array = label_array

        patch_img_tensor = torch.FloatTensor(patch_img_array).unsqueeze(0)
        patch_label_tensor = torch.FloatTensor(patch_label_array).unsqueeze(0)

        del patch_img_array
        del patch_label_array
        gc.collect()

        ret = {'input': patch_img_tensor}       # shape = {1, x, y, z}
        #  -----------  eight random mini patches  ---------- obtained right here ---------- #
        # {'mini_patch': cat_patch_tensor}      # shape = {8, 112, 112, 160}  use this eight to enhance the batch size?
        # might over the ability of our cards
        seg = {'label': patch_label_tensor}     # shape = {1, 112, 112, 160}
        #  -----------  correlated
        # {'mini_label': cat_label_tensor}      #
        ret.update(seg)

        return ret

    def __len__(self):
        return len(self.label_list)
