"""
    write for distance transform map img
"""
import os
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset as dataset
import nibabel as nib
import numpy as np
import random
import gc
import math


class Dataset(dataset):
    def __init__(self, cfg, transform, img_dir, label_dir, is_train, mini_img, mini_label):
        self.transform = transform
        self.patch_size = cfg.data.patch_size
        self.clip_size = cfg.data.clip_size
        self.img_list = os.listdir(label_dir)
        self.img_dir_list = list(map(lambda x: os.path.join(img_dir, x), self.img_list))
        self.label_list = os.listdir(label_dir)
        self.label_dir_list = list(map(lambda x: os.path.join(label_dir, x), self.label_list))
        self.cfg = cfg
        self.train_tag = is_train

    def __getitem__(self, item):
        img_array = nib.load(self.label_dir_list[item].replace('LABEL', 'IMG')).get_fdata()
        label_array = nib.load(self.label_dir_list[item]).get_fdata().astype(np.uint8)

        if self.train_tag:
            start_x = random.randint(0, label_array.shape[0] - self.clip_size[0])
            start_y = random.randint(0, label_array.shape[1] - self.clip_size[1])
            # start_z = random.randint(0, label_array.shape[2] - self.clip_size)
        else:
            start_x = math.ceil((label_array.shape[0] - self.clip_size[0]) / 2)
            start_y = math.ceil((label_array.shape[1] - self.clip_size[1]) / 2)
            # start_z = random.randint(0, label_array.shape[2] - self.clip_size)

        patch_img_array = img_array[start_x: start_x + self.clip_size[0], start_y: start_y + self.clip_size[1], 0: 160]
        patch_label_array = label_array[start_x: start_x + self.clip_size[0], start_y: start_y + self.clip_size[1], 0: 160]

        # test_A = patch_label_array.copy() * 125
        # cv2.imshow('sagittal', test_A[74, :, :])
        # cv2.imshow('coronal', test_A[:, 98, :])
        # cv2.imshow('axial', test_A[:, :, 104])
        # cv2.waitKey()

        patch_img_tensor = torch.FloatTensor(patch_img_array).unsqueeze(0)  # torch.float32
        patch_label_tensor = torch.FloatTensor(patch_label_array).unsqueeze(0)

        del patch_img_array
        del patch_label_array
        gc.collect()

        ret = {'input': patch_img_tensor}       # shape = {1, 112, 112, 160}
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
