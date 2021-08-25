import torch.nn as nn
import torch
import numpy as np
import SimpleITK as sitk
from time import time

import math
import os
import scipy.ndimage as ndimage

from Echocardiograph.lib.net.BScanSeg.Vnet8x import VNet
from Echocardiograph.utils.evaluate_utils import cal_dice
from Echocardiograph.utils.ImgResample import nrrd_spacing_resample, nii_spacing_resample, img_padding


def build_model(pth, net, para_half):
    if para_half:
        net.half()   # from double to float
    net = nn.DataParallel(net).cuda()
    saved_model = torch.load(pth)
    net.load_state_dict({('module.' + k): v for k, v in saved_model['net'].items()})
    net.eval()

    return net


def single_img_splitx(img_array, split_size, para_half):
    """
    :param img_array:       rotated img array, x y x
    :param split_size:      ()
    :return:
    """
    x0, x1, x2 = 0, int(split_size[0] / 2), img_array.shape[0] - split_size[0]
    y0, y1, y2 = 0, int(split_size[1] / 2), img_array.shape[1] - split_size[1]
    # x_list = [x0, x1, x1, x2, x1]
    # y_list = [y1, y0, y2, y1, y1]
    x_list = [x0, x2, x0, x2, x0, x1, x1, x2, x1]
    y_list = [y0, y0, y2, y2, y1, y0, y2, y1, y1]
    z_list = [0]
    tensor_patches = []
    for i in range(len(x_list)):  # num_x = 3
        # for j in range(len(y_list)):  # num_y = 3
        tmp_array = img_array[x_list[i]: x_list[i] + patch_size[0], y_list[i]: y_list[i] + patch_size[1],
                    z_list[0]: z_list[0] + patch_size[2]]
        tmp_copy = tmp_array.copy()
        tmp_tensor = torch.FloatTensor(tmp_copy).unsqueeze(0).type(torch.FloatTensor)
        # (112,112,112) - (1,112,112,112)
        if para_half:
            tmp_tensor = tmp_tensor.float().cuda().half()
        tensor_patches.append(tmp_tensor)

    split_list = {'x': x_list, 'y': y_list, 'z': z_list}
    return tensor_patches, split_list


def fuse_predict(label_list, prob_list, target_shape, index_list):
    """
    :param label_list:     [ [1 x y z] ... ]
    :param prob_list:      [ [3 x y z] ... ]
    :param target_shape:   [ [224 224 208] ]
    :param index_list:     {'x': [x0 x1 x2 x3 x4 x5], 'y': ... }
    :return:  final_label_array ( x y z ), final_prob_array (3, x, y, z)
    """
    fuse_bound = 2
    final_label_array = np.zeros(target_shape).astype(np.uint8)
    final_prob_array = np.zeros((3, target_shape[0], target_shape[1], target_shape[2]))
    x_list = index_list['x']
    y_list = index_list['y']
    z_list = index_list['z']
    patch_size = label_list[0].shape
    x_length = patch_size[0] - fuse_bound
    y_length = patch_size[1] - fuse_bound
    index = 0
    for i in range(len(x_list)):
        # for j in range(len(y_list)):
        final_label_array[
        x_list[i] + fuse_bound: x_list[i] + patch_size[0] - fuse_bound,
        y_list[i] + fuse_bound: y_list[i] + patch_size[1] - fuse_bound,
        z_list[0]: z_list[0] + patch_size[2]] \
            = label_list[index][fuse_bound:x_length, fuse_bound:y_length, :]

        final_prob_array[
        :, x_list[i] + fuse_bound: x_list[i] + patch_size[0] - fuse_bound,
        y_list[i] + fuse_bound: y_list[i] + patch_size[1] - fuse_bound,
        z_list[0]: z_list[0] + patch_size[2]] \
            = prob_list[index][:, fuse_bound:x_length, fuse_bound:y_length, :]

        index += 1

    return final_label_array, final_prob_array


def net_val(input_list, pred_model):
    out_prob_list = []
    out_label_list = []
    for index in range(len(input_list)):
        with torch.no_grad():
            input_tensor = input_list[index].unsqueeze(dim=0).cuda()  # tensor([1, 1, x, y, z])
            output_dict = pred_model(input_tensor)
        output_tensor = output_dict['map']  # 1,3,x,y,z
        out_prob_array = output_tensor.squeeze().cpu().detach().numpy()  # ndarray([c, 112, 112, 112])
        out_label_array = np.argmax(out_prob_array, axis=0)
        out_prob_list.append(out_prob_array)
        out_label_list.append(out_label_array)
    return out_prob_list, out_label_list


def single_predict(img_array, size, half, made_model):
    # --- input full_size x y z array, return 5 splited image xyz arrays
    patch_list, bound_list = single_img_splitx(img_array, size, half)  # the input_img.size=224.224.160
    # --- input splitted array lists, val them all
    predict_prob_list, predict_label_list = net_val(patch_list, made_model)  # 112 112 160
    # --- fuse the predict lists to the original size
    predict_label, predict_prob = \
        fuse_predict(predict_label_list, predict_prob_list, img_array.shape, bound_list)  # 224 224 160
    return predict_label, predict_prob


# def smooth(img_array):
#     """
#     :param img_array:  predicted image array
#     :return:  smoothed image array
#
#     to create a boundary([0,1]) and time the predicted image
#     """


if __name__ == '__main__':
    mod_pth = 'C:\\Users\\mnrsm\\Desktop\\Exploer\\B-scan\\Echocardiograph\\model\\BScanSeg\\Data_112_112_160_V5' \
              '\\SVnet_ELU_+GDiceBG+RandomPatches_Ori_Padding\\3499.pth'

    # ---------------------- define parameters ------------------------------- #
    half_para = False
    val = True
    save = True
    save_posb = False
    save_delta = False
    visualize = False
    post_process = True
    num_class = 2
    patch_size = (112, 112, 160)  # x y z
    united_shape = (208, 224, 224)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # -------------------------- build the net ------------------------------- #
    model = VNet(elu=True, nll=False, out_channel=num_class + 1)
    predict_model = build_model(mod_pth, model, half_para)

    total_folder = 'C:\\Users\\mnrsm\\Downloads\\NRRD\\test\\'
    for j, folder_name in enumerate(os.listdir(total_folder)):
        folder_pth = os.path.join(total_folder, folder_name)
        for n, file_name in enumerate(os.listdir(folder_pth)):
            label_folder = ''
            save_folder = ''
            img_name = file_name
            label_name = img_name.replace('IMG_', 'LABEL_') if img_name.split('_')[0] == 'IMG' else ''
            img_pth, label_pth = os.path.join(folder_pth, img_name), os.path.join(label_folder, label_name)
            save_pth = img_pth.replace('Seq_', 'Pred_Seq_9_')
            print('current img =', file_name)
            # ------------------------ data preprocess ------------------------------- #
            ori_img = sitk.ReadImage(img_pth)
            ori_spacing = ori_img.GetSpacing()
            ori_img_array = sitk.GetArrayFromImage(ori_img)

            if len(ori_img_array.shape) == 3:   # is nrrd or single image
                print('The input image is a 3D image')

                if ori_spacing[0] != ori_spacing[1]:  # united spacing  z y x
                    print('The image with single frame need to be resample')
                    space_united_array, new_spacing = nii_spacing_resample(ori_img_array, ori_spacing)
                else:
                    print('The image with single frame no need to be resample')
                    space_united_array = ori_img_array
                    new_spacing = ori_spacing

                if space_united_array.shape != united_shape:
                    print('The image need to be padded to fit the input size of our model')
                    val = False
                    save_delta = False
                    size_padding_array = img_padding(space_united_array, target_shape=(208, 224, 224))
                    united_img_array = np.einsum('zyx->xyz', size_padding_array)  # 224 224 208
                else:
                    print('The image.shape() do fit the input size of our model')
                    united_img_array = np.einsum('zyx->xyz', space_united_array)  # 224 224 208
                predict_label_array, predice_prob_array = single_predict(united_img_array, patch_size, half_para, predict_model)
                prob_array = (predice_prob_array * 125).astype(np.uint8)

            elif len(ori_img_array.shape) == 4:
                print('The input image is a 4D sequence image')
                val = False
                visualize = False
                save_posb = False
                save_delta = False
                if ori_spacing[0] != ori_spacing[1]:  # united spacing  z y x t
                    print('The image with single frame need to be resample')
                    space_united_array, new_spacing = nrrd_spacing_resample(ori_img_array, ori_spacing)
                    print('------------------------ nrrd resample finished ----------------------')
                else:
                    print('The image with single frame no need to be resample')
                    space_united_array = ori_img_array
                    new_spacing = ori_spacing

                saved_array = np.zeros((united_shape[0], united_shape[1], united_shape[2],
                                        space_united_array.shape[3])).astype(np.uint8)     # z y x t

                for i in range(space_united_array.shape[3]):
                    start_time = time()
                    single_frame_array = space_united_array[:, :, :, i]     # z 208 y 224 x 224
                    if single_frame_array.shape != united_shape:
                        size_padding_array = img_padding(single_frame_array, target_shape=united_shape)
                        united_img_array = np.einsum('zyx->xyz', size_padding_array)  # 224 224 208
                    else:
                        united_img_array = np.einsum('zyx->xyz', single_frame_array)  # 224 224 208

                    predict_label_array, _ = single_predict(united_img_array, patch_size, half_para, predict_model)
                    print('Predict Frame_No', i, '/', space_united_array.shape[3], 'costs', "%.2f" % (time() - start_time), 'seconds')
                    saved_array[:, :, :, i] = np.einsum('xyz->zyx', predict_label_array)

                print('---------- Sequence image prediction finished! ------------------')

            if val:
                ori_label = sitk.ReadImage(label_pth)
                ori_label_array = sitk.GetArrayFromImage(ori_label)
                united_label_array = np.einsum('zyx->xyz', ori_label_array)
                dice_total, dice_class = cal_dice(predict_label_array, united_label_array, num_class=2)
                print('dice_total:', dice_total)
                [print('dice_class', i, '=', dice_class[i]) for i in range(len(dice_class))]

            if visualize:
                label_array = (predict_label_array * 125).astype(np.uint8)
                import cv2

                cv2.imshow('r', prob_array[1, 112, :, :])
                cv2.imshow('l', prob_array[2, 112, :, :])
                cv2.imshow("out", label_array[112, :, :])
                cv2.imshow('r_S', prob_array[1, :, :, 80])
                cv2.imshow('l_S', prob_array[2, :, :, 80])
                cv2.imshow("out_S", label_array[:, :, 80])
                cv2.waitKey()

            if save_posb:
                save_pos_l = save_pth.replace('Predict', 'left_posb')
                save_pos_r = save_pth.replace('Predict', 'right_posb')
                posb_left = sitk.GetImageFromArray(np.einsum('xyz->zyx', prob_array[2, :, :, :]))
                posb_right = sitk.GetImageFromArray(np.einsum('xyz->zyx', prob_array[1, :, :, :]))
                posb_left.SetDirection(ori_img.GetDirection())
                posb_left.SetOrigin(ori_img.GetOrigin())
                posb_left.SetSpacing(ori_img.GetSpacing())
                sitk.WriteImage(posb_left, save_pos_l)
                posb_right.SetDirection(ori_img.GetDirection())
                posb_right.SetOrigin(ori_img.GetOrigin())
                posb_right.SetSpacing(ori_img.GetSpacing())
                sitk.WriteImage(posb_right, save_pos_r)

            if save_delta:
                save_delta_pth = save_pth.replace('Predict', 'Delta')
                total_delta = (united_label_array + 1) * 3 - predict_label_array
                total_delta[total_delta == 3] = 0
                total_delta[total_delta == 5] = 0
                total_delta[total_delta == 7] = 0
                # --  gt/pred  1: 0/2  2: 0/1  3: 2/1  4: 1/2  5: 2/0  6: 1/0
                total_delta[total_delta == 8] = 3
                total_delta[total_delta == 9] = 5

                total_delta_img = sitk.GetImageFromArray(np.einsum('xyz->zyx', total_delta))
                total_delta_img.SetDirection(ori_img.GetDirection())
                total_delta_img.SetOrigin(ori_img.GetOrigin())
                total_delta_img.SetSpacing(new_spacing)
                sitk.WriteImage(total_delta_img, save_delta_pth)

            if save:
                if len(ori_img_array.shape) == 3:
                    predict_img = sitk.GetImageFromArray(np.einsum('xyz->zyx', predict_label_array))
                    predict_img.SetDirection(ori_img.GetDirection())
                    predict_img.SetOrigin(ori_img.GetOrigin())
                    predict_img.SetSpacing(new_spacing)
                    sitk.WriteImage(predict_img, save_pth)
                    print('Single image saved at', save_pth)

                if len(ori_img_array.shape) == 4:
                    predict_img = sitk.GetImageFromArray(saved_array)
                    predict_img.SetDirection(ori_img.GetDirection())
                    predict_img.SetOrigin(ori_img.GetOrigin())
                    predict_img.SetSpacing(new_spacing)
                    sitk.WriteImage(predict_img, save_pth)
                    print('Sequence image saved at', save_pth)

    print('end')





