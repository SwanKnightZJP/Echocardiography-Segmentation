import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset as dataset
import nibabel as nib
import numpy as np
import random
import gc
import SimpleITK as sitk
import scipy.ndimage as ndimage
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

        # restrict the boundary
        start_z = random.randint(0, label_array.shape[2] - self.clip_size[2])

        mid_x = int(img_array.shape[0]/2)
        mid_y = int(img_array.shape[1]/2)
        sagittal_nib = img_array[mid_x, :, :]
        list_z = sagittal_nib[1, :].tolist()
        mid_z = list_z.index(max(list_z))

        if start_z < mid_z:
            x_index = -(mid_x / mid_z) * (start_z + self.clip_size[2] / 2) + mid_x
            y_index = -(mid_y / mid_z) * (start_z + self.clip_size[2] / 2) + mid_y
        else:
            x_index = (mid_x / (self.clip_size[2] - mid_z)) * (start_z + self.clip_size[2] / 2) + mid_x * mid_z / (self.clip_size[2] - mid_z)
            y_index = (mid_y / (self.clip_size[2] - mid_z)) * (start_z + self.clip_size[2] / 2) + mid_y * mid_z / (self.clip_size[2] - mid_z)

        tmp_x_start, tmp_x_end = x_index - self.clip_size[0] / 2 - 1, img_array.shape[0] - x_index - self.clip_size[0] / 2 - 1
        tmp_y_start, tmp_y_end = y_index - self.clip_size[0] / 2 - 1, img_array.shape[1] - y_index - self.clip_size[0] / 2 - 1

        if tmp_x_start < 0:
            x_start_bound = int(self.clip_size[0] / 2 + 1)
            x_end_bound = int(img_array.shape[0] - self.clip_size[0] / 2 - 1)
        else:
            x_start_bound = int(x_index)
            x_end_bound = int(img_array.shape[0] - x_index)

        if tmp_y_start < 0:
            y_start_bound = int(self.clip_size[1] / 2 + 1)
            y_end_bound = int(img_array.shape[1] - self.clip_size[1] / 2 - 1)
        else:
            y_start_bound = int(y_index)
            y_end_bound = int(img_array.shape[1] - y_index)

        start_x = random.randint(x_start_bound, x_end_bound)
        start_y = random.randint(y_start_bound, y_end_bound)  # need to correlated with z

        patch_img_array = img_array[start_x - int(self.clip_size[0] / 2): start_x - int(self.clip_size[0] / 2) + self.clip_size[0],
                          start_y - int(self.clip_size[1] / 2): start_y - int(self.clip_size[1] / 2) + self.clip_size[1],
                          start_z: start_z + self.clip_size[2]]
        patch_label_array = label_array[start_x - int(self.clip_size[0] / 2): start_x - int(self.clip_size[0] / 2) + self.clip_size[0],
                            start_y - int(self.clip_size[1] / 2): start_y - int(self.clip_size[1] / 2) + self.clip_size[1],
                            start_z: start_z + self.clip_size[2]]

        patch_up_img_array = (ndimage.zoom(patch_img_array, (2, 2, 2), order=3)).astype(np.uint8)
        patch_up_label_array = (ndimage.zoom(patch_img_array, (2, 2, 2), order=1)).astype(np.uint8)

        patch_img_tensor = torch.FloatTensor(patch_up_img_array).unsqueeze(0)
        patch_label_tensor = torch.FloatTensor(patch_up_label_array).unsqueeze(0)

        del patch_img_array
        del patch_label_array
        gc.collect()

        ret = {'input': patch_img_tensor}       # shape = {1, 112, 112, 160}
        seg = {'label': patch_label_tensor}    # shape = {1, 112, 112, 160}
        ret.update(seg)

        return ret

    def __len__(self):
        return len(self.label_list)
