"""

  SegCombine: single_nii
  nii_shape:
    x = 0 - max = from left to right
    y = 0 - max = from P to A
    z = 0 - max = from bottom to surface

  input: data_pth
  output: seg_result

  split_mod:
      -- 1: whatever the shape of the input image -- get 18 Cubes
      -- 2: the num of the cubes depends on the shape of the image without overlapping

  fuse_mod:
      -- 1: without overlapping
      -- 2: with overlap01
      -- 3: with overlap02

  test: --2 --1

"""
import torch.nn as nn
import torch
import nibabel as nib
import math
import numpy as np
# from torchvision import transforms
import os

from lib.net.BScanSeg.VnetOri import VNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def img_split(input_array, patch_size=(112, 112, 112), split_model=1):
    """
    :param input_array:  nii_array, (X, Y, 208)
    :param patch_size:   the patch_size of the network default = 112
    :param fuse_mod:   the patch_size of the network default = 112
    :return: array_patcher: list contains 8-18 112*112*112cubes?
    """
    tensor_patches = []
    max_x, max_y, max_z = input_array.shape[0], input_array.shape[1], input_array.shape[2]
    if split_model == 1:
        num_x, num_y, num_z = 3, 3, 2
        # ------ x list ------ #
        x_mid = (max_x - patch_size[0]) // 2
        x_list = [0, x_mid, (max_x - patch_size[0])]
        # ------ y list ------ #
        y_mid = (max_y - patch_size[1]) // 2
        y_list = [0, y_mid, (max_y - patch_size[1])]
        # ------ z list ------ #
        z_list = [0, (max_z - patch_size[2])]
        '''
            xyz:
            000 001 010 011 020 021
            100 101 110 111 120 121
            200 201 210 211 220 221
        '''

    elif split_model == 2:
        num_x = math.ceil(max_x/patch_size[0])
        num_y = math.ceil(max_y/patch_size[1])
        num_z = 1
        # ------ x list ------ #
        if num_x == 3:
            x_mid = (max_x - patch_size[0]) // 2
            x_list = [0, x_mid, (max_x - patch_size[0])]
        else:
            x_list = [0, (max_x - patch_size[0])]
        # ------ y list ------ #
        if num_y == 3:
            y_mid = (max_y - patch_size[1]) // 2
            y_list = [0, y_mid, (max_y - patch_size[1])]
        else:
            y_list = [0, (max_y - patch_size[1])]
        # ------ z list ------ #
        z_list = [0, (max_z - patch_size[2])]
        '''
            xyz:
            000 001 010 011 | 020 021
            100 101 110 111 | 120 121
            --- --- --- --- | --- --- 
            200 201 210 211 | 220 221
        '''

    elif split_model == 3:  # center crop
        num_x = 1
        num_y = 1
        num_z = 2
        x_list = [(max_x - patch_size[0]) // 2]
        y_list = [(max_y - patch_size[0]) // 2]
        z_list = [0, (max_z - patch_size[2])]

    elif split_model == 4:
        num_x = 1
        num_y = 1
        num_z = 1
        x_list = [(max_x - patch_size[0]) // 2]
        y_list = [(max_y - patch_size[0]) // 2]
        z_list = [0]

    elif split_model == 5:
        num_x = math.ceil(max_x/patch_size[0])
        num_y = math.ceil(max_y/patch_size[1])
        num_z = 1
        # ------ x list ------ #
        if num_x == 3:
            x_mid = (max_x - patch_size[0]) // 2
            x_list = [0, x_mid, (max_x - patch_size[0])]
        else:
            x_list = [0, (max_x - patch_size[0])]
        # ------ y list ------ #
        if num_y == 3:
            y_mid = (max_y - patch_size[1]) // 2
            y_list = [0, y_mid, (max_y - patch_size[1])]
        else:
            y_list = [0, (max_y - patch_size[1])]
        # ------ z list ------ #
        z_list = [0]
        '''
            xyz:
            000 001 010 011 | 020 021
            100 101 110 111 | 120 121
            --- --- --- --- | --- --- 
            200 201 210 211 | 220 221
        '''

    else:
        raise KeyError("wrong input")

    for i in range(num_x):  # num_x = 3
        for j in range(num_y):  # num_y = 3
            for k in range(num_z):  # num_z = 2
                tmp_array = input_array[x_list[i]: x_list[i] + patch_size[0], y_list[j]: y_list[j] + patch_size[1],
                            z_list[k]: z_list[k] + patch_size[2]]
                tmp_copy = tmp_array.copy()
                tmp_tensor = torch.FloatTensor(tmp_copy).unsqueeze(0).type(torch.FloatTensor)  # (112,112,112) - (1,112,112,112)

                # # #-------------- warning !!!!! do not play fire !!!!!-----------------#
                tmp_tensor = tmp_tensor.float().cuda().half()
                # # #-------------- warning !!!!! do not play fire !!!!!-----------------#

                tensor_patches.append(tmp_tensor)

    split_list = {'x': x_list, 'y': y_list, 'z': z_list}

    return tensor_patches, split_list


def img_stack(input_array):
    tensor_patches = []
    tmp_tensor = torch.FloatTensor(input_array).unsqueeze(0).type(torch.FloatTensor)
    tensor_patches.append(tmp_tensor)
    return tensor_patches


def net_val(patch_list, network):
    out_list = []
    for index in range(len(patch_list)):
        with torch.no_grad():
            input_tensor = patch_list[index].unsqueeze(dim=0).cuda()  # tensor([1, 1, 112, 112, 112])
            output_dict = network(input_tensor)

        output_tensor = output_dict['map']

        test_a = output_tensor[:, 1, :, :, :]
        test_b = output_tensor[:, 2, :, :, :]
        array_r = test_a.squeeze().cpu().detach().numpy()
        array_b = test_b.squeeze().cpu().detach().numpy()
        pre_l = (array_b * 125).astype(np.uint8)
        pre_r = (array_r * 125).astype(np.uint8)
        output_array = output_tensor.squeeze().cpu().detach().numpy()
        out_array = np.argmax(output_array, axis=0)
        out_array = (out_array * 125).astype(np.uint8)
        import cv2
        cv2.imshow('r', pre_r[56, :, :])
        cv2.imshow('l', pre_l[56, :, :])
        cv2.imshow("out", out_array[56, :, :])
        cv2.imshow('r_S', pre_r[:, :, 80])
        cv2.imshow('l_S', pre_l[:, :, 80])
        cv2.imshow("out_S", out_array[:, :, 80])
        cv2.waitKey()

        output_array = output_tensor.squeeze().cpu().detach().numpy()          # ndarray([c, 112, 112, 112])
        # -- achieve class -- #
        pre_seg_array = np.argmax(output_array, axis=0).astype(np.uint8)  # ndarray([112, 112, 112])
        out_list.append(pre_seg_array)
    return out_list, output_array[1:2, :, :, :]  #


def fuse_no_overlap(array_list, img_shape, s_dict):
    """
    :param array_list:  the pred result from the net
    :param img_shape:   the shape of the ori_img -- used to init the array space
    :param s_dict:      {'x': x_list, 'y': y_list, 'z': z_list}
    :param patch_size:  the trained patchsize
    :return:
    """
    # defined empty array
    final_seg_array = np.ones(img_shape).astype(np.uint8)
    x_list = s_dict['x']  # x_start_index(s)
    y_list = s_dict['y']  # y_start_index(s)
    z_list = s_dict['z']  # z_start_index(s)
    # patch_size = array_list[0].shape  # default (112, 112, 112)

    if len(x_list) == 3 & len(y_list) == 3:
        final_seg_array[x_list[2]:x_list[2]+112, y_list[2]:y_list[2]+112, z_list[1]:z_list[1]+160] = array_list[17] # 221 - 17
        final_seg_array[x_list[0]:x_list[0]+112, y_list[2]:y_list[2]+112, z_list[1]:z_list[1]+160] = array_list[5] # 021 - 5
        final_seg_array[x_list[2]:x_list[2]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[13] # 201 - 13
        final_seg_array[x_list[0]:x_list[0]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[1] # 001 - 1
        final_seg_array[x_list[1]:x_list[1]+112, y_list[2]:y_list[2]+112, z_list[1]:z_list[1]+160] = array_list[11] # 121 - 11
        final_seg_array[x_list[2]:x_list[2]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[15] # 211 - 15
        final_seg_array[x_list[0]:x_list[0]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[3] # 011 - 3
        final_seg_array[x_list[1]:x_list[1]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[7] # 101 - 7
        final_seg_array[x_list[1]:x_list[1]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[9] # 111 - 9

        final_seg_array[x_list[2]:x_list[2]+112, y_list[2]:y_list[2]+112, z_list[0]:z_list[0]+160] = array_list[16] # 220 - 16
        final_seg_array[x_list[0]:x_list[0]+112, y_list[2]:y_list[2]+112, z_list[0]:z_list[0]+160] = array_list[4] # 020 - 4
        final_seg_array[x_list[2]:x_list[2]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[12] # 200 - 12
        final_seg_array[x_list[0]:x_list[0]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[0] # 000 - 0
        final_seg_array[x_list[1]:x_list[1]+112, y_list[2]:y_list[2]+112, z_list[0]:z_list[0]+160] = array_list[10] # 120 - 10
        final_seg_array[x_list[2]:x_list[2]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[14] # 210 - 14
        final_seg_array[x_list[0]:x_list[0]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[2] # 010 - 2
        final_seg_array[x_list[1]:x_list[1]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[6] # 100 - 6
        final_seg_array[x_list[1]:x_list[1]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[8] # 110 - 8

    if len(x_list) == 3 & len(y_list) == 2:
        final_seg_array[x_list[2]:x_list[2]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[11]  # 211 - 11
        final_seg_array[x_list[0]:x_list[0]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[3]  # 011 - 3
        final_seg_array[x_list[2]:x_list[2]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[9]  # 201 - 9
        final_seg_array[x_list[0]:x_list[0]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[1]  # 001 - 1
        final_seg_array[x_list[1]:x_list[1]+112, y_list[1]:y_list[1]+112, z_list[1]:z_list[1]+160] = array_list[7]  # 111 - 7
        final_seg_array[x_list[1]:x_list[1]+112, y_list[0]:y_list[0]+112, z_list[1]:z_list[1]+160] = array_list[5]  # 101 - 5

        final_seg_array[x_list[2]:x_list[2]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[10]  # 210 - 10
        final_seg_array[x_list[0]:x_list[0]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[2]  # 010 - 2
        final_seg_array[x_list[2]:x_list[2]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[8]  # 200 - 8
        final_seg_array[x_list[0]:x_list[0]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[0]  # 000 - 0
        final_seg_array[x_list[1]:x_list[1]+112, y_list[1]:y_list[1]+112, z_list[0]:z_list[0]+160] = array_list[6]  # 110 - 6
        final_seg_array[x_list[1]:x_list[1]+112, y_list[0]:y_list[0]+112, z_list[0]:z_list[0]+160] = array_list[4]  # 100 - 4

    if len(x_list) == 2 & len(y_list) == 3:
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[2]:y_list[2] + 112, z_list[1]:z_list[1] + 112] = array_list[11]  # 121 - 11
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[2]:y_list[2] + 112, z_list[1]:z_list[1] + 112] = array_list[5]  # 021 - 5
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[0]:y_list[0] + 112, z_list[1]:z_list[1] + 112] = array_list[7]  # 101 - 7
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[0]:y_list[0] + 112, z_list[1]:z_list[1] + 112] = array_list[1]  # 001 - 1
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[1]:y_list[1] + 112, z_list[1]:z_list[1] + 112] = array_list[9]  # 111 - 9
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[1]:y_list[1] + 112, z_list[1]:z_list[1] + 112] = array_list[3]  # 011 - 3

        final_seg_array[x_list[1]:x_list[1] + 112, y_list[2]:y_list[2] + 112, z_list[0]:z_list[0] + 112] = array_list[10]  # 120 - 10
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[2]:y_list[2] + 112, z_list[0]:z_list[0] + 112] = array_list[4]  # 020 - 4
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[0]:y_list[0] + 112, z_list[0]:z_list[0] + 112] = array_list[6]  # 100 - 6
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[0]:y_list[0] + 112, z_list[0]:z_list[0] + 112] = array_list[00]  # 000 - 0
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[1]:y_list[1] + 112, z_list[0]:z_list[0] + 112] = array_list[8]  # 110 - 8
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[1]:y_list[1] + 112, z_list[0]:z_list[0] + 112] = array_list[2]  # 010 - 2

    if len(x_list) == 2 & len(y_list) == 2:
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[1]:y_list[1] + 112, z_list[1]:z_list[1] + 112] = array_list[7]  # 111 - 7
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[1]:y_list[1] + 112, z_list[1]:z_list[1] + 112] = array_list[3]  # 011 - 3
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[0]:y_list[0] + 112, z_list[1]:z_list[1] + 112] = array_list[5]  # 101 - 5
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[0]:y_list[0] + 112, z_list[1]:z_list[1] + 112] = array_list[1]  # 001 - 1

        final_seg_array[x_list[1]:x_list[1] + 112, y_list[1]:y_list[1] + 112, z_list[0]:z_list[0] + 112] = array_list[6]  # 110 - 6
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[1]:y_list[1] + 112, z_list[0]:z_list[0] + 112] = array_list[2]  # 010 - 2
        final_seg_array[x_list[1]:x_list[1] + 112, y_list[0]:y_list[0] + 112, z_list[0]:z_list[0] + 112] = array_list[4]  # 100 - 4
        final_seg_array[x_list[0]:x_list[0] + 112, y_list[0]:y_list[0] + 112, z_list[0]:z_list[0] + 112] = array_list[0]  # 000 - 0

    return final_seg_array


def fuse_with_overlap(array_list, img_shape, s_dict):
    """
        :param array_list:  the pred result from the net
        :param img_shape:   the shape of the ori_img -- used to init the array space
        :param s_dict:      {'x': x_list, 'y': y_list, 'z': z_list}
        :param patch_size:  the trained patchsize
        :return:
        """
    # defined empty array
    final_seg_array = np.zeros(img_shape).astype(np.uint8)
    x_list = s_dict['x']  # x_start_index(s)
    y_list = s_dict['y']  # y_start_index(s)
    z_list = s_dict['z']  # z_start_index(s)
    patch_size = array_list[0].shape  # default (112, 112, 112)
    index = 0
    for i in range(len(x_list)):
        for j in range(len(y_list)):
            # for k in range(len(z_list)):
            final_seg_array[x_list[i]: x_list[i] + patch_size[0], y_list[j]: y_list[j] + patch_size[1],
            z_list[0]: z_list[0] + patch_size[2]] = array_list[index]
            index += 1
    # final_seg_array[final_seg_array > 0] = 1

    return final_seg_array
