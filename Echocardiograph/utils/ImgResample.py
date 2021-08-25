"""

the input nrrd sequenceimg with spacing (0.652, 0.863, 0.4)  where the shape = (224, 176, 208)
to normalized the input shape, the spacing should be unified to (0.652 0.652 0.4) with the shape = (224, 231, 208)

--------

note ndimage.output.shape is rounding
while np.uint8 is

"""
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
import SimpleITK as sitk
from time import time
import os


def img_padding(image_array, target_shape):
    # adjust the Z
    ori_shape = image_array.shape
    padding_img = np.zeros(target_shape).astype(np.uint8)
    """
    如果原Z轴过大，那么 新数据的 区间将被填满 左右维度应当相同
    如果原Y轴过大，那么 新数据的 区间将被填满
    如果原X轴过大，那么 新数据的 区间将被填满
    """
    if ori_shape[0] >= target_shape[0]:          #
        if ori_shape[1] >= target_shape[1]:      #
            if ori_shape[2] >= target_shape[2]:  # Z> Y> X>
                padding_img[0:target_shape[0], :, :] = image_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2),
                    int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:                               # Z> Y> X<
                padding_img[0:target_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]
        else:                                   # Z> Y< X>
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = image_array[0:target_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:                               # Z> Y< X<
                padding_img[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = image_array[0:target_shape[0], :, :]
    else:
        if ori_shape[1] >= target_shape[1]:
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:ori_shape[0], :, :] = image_array[0:ori_shape[0],
                                                       int(ori_shape[1] / 2 - target_shape[1] / 2):int(
                                                           ori_shape[1] / 2 + target_shape[1] / 2),
                                                       int(ori_shape[2] / 2 - target_shape[2] / 2):int(
                                                           ori_shape[2] / 2 + target_shape[2] / 2)]
            else:
                padding_img[0:ori_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:ori_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]
        else:
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = image_array[0:ori_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:
                padding_img[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = image_array[0:ori_shape[0], :, :]

    return padding_img


def img_padding_i_gt(ori_shape, target_shape, image_array, gt_array):
    # adjust the Z
    padding_img = np.zeros(target_shape).astype(np.uint8)
    padding_label = padding_img.copy()
    """
    如果原Z轴过大，那么 新数据的 区间将被填满 左右维度应当相同
    如果原Y轴过大，那么 新数据的 区间将被填满
    如果原X轴过大，那么 新数据的 区间将被填满
    """
    if ori_shape[0] >= target_shape[0]:          #
        if ori_shape[1] >= target_shape[1]:      #
            if ori_shape[2] >= target_shape[2]:  # Z> Y> X>
                padding_img[0:target_shape[0], :, :] = image_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2),
                    int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]

                padding_label[0:target_shape[0], :, :] = gt_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2),
                    int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:                               # Z> Y> X<
                padding_img[0:target_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]

                padding_label[0:target_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:target_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]
        else:                                   # Z> Y< X>
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = image_array[0:target_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
                padding_label[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = gt_array[0: target_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:                               # Z> Y< X<
                padding_img[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = image_array[0:target_shape[0], :, :]
                padding_label[0:target_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = gt_array[0: target_shape[0], :, :]
    else:
        if ori_shape[1] >= target_shape[1]:
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:ori_shape[0], :, :] = image_array[0:ori_shape[0],
                                                       int(ori_shape[1] / 2 - target_shape[1] / 2):int(
                                                           ori_shape[1] / 2 + target_shape[1] / 2),
                                                       int(ori_shape[2] / 2 - target_shape[2] / 2):int(
                                                           ori_shape[2] / 2 + target_shape[2] / 2)]

                padding_label[0:target_shape[0], :, :] = gt_array[0:target_shape[0] - 1,
                                                         int(ori_shape[1] / 2 - target_shape[1] / 2):int(
                                                             ori_shape[1] / 2 + target_shape[1] / 2),
                                                         int(ori_shape[2] / 2 - target_shape[2] / 2):int(
                                                             ori_shape[2] / 2 + target_shape[2] / 2)]
            else:
                padding_img[0:ori_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:ori_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]

                padding_label[0:ori_shape[0], :,
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] = \
                    image_array[0:ori_shape[0],
                    int(ori_shape[1] / 2 - target_shape[1] / 2):int(ori_shape[1] / 2 + target_shape[1] / 2), :]
        else:
            if ori_shape[2] >= target_shape[2]:
                padding_img[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = image_array[0:ori_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
                padding_label[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2), :] \
                    = gt_array[0: ori_shape[0], :,
                      int(ori_shape[2] / 2 - target_shape[2] / 2):int(ori_shape[2] / 2 + target_shape[2] / 2)]
            else:
                padding_img[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = image_array[0:ori_shape[0], :, :]
                padding_label[0:ori_shape[0],
                int(target_shape[1] / 2 - ori_shape[1] / 2):int(target_shape[1] / 2 + ori_shape[1] / 2),
                int(target_shape[2] / 2 - ori_shape[2] / 2):int(target_shape[2] / 2 + ori_shape[2] / 2)] \
                    = gt_array[0: ori_shape[0], :, :]

    return padding_img, padding_label


def img_resample(ori_shape, target_shape, image_array, gt_array, spacing):
    """
    :param ori_shape:    z y x
    :param target_shape: z y x
    :param image_array:  the img_array.shape(z y x)
    :param gt_array:
    :param spacing:      spacing = img_spacing ()
    :return:
    """
    x_rate = target_shape[2] / ori_shape[2]
    y_rate = target_shape[1] / ori_shape[1]
    resample_img_array = (zoom(image_array, (1, y_rate, x_rate), order=3)).astype(np.uint8)
    resample_label_array = (zoom(gt_array, (1, y_rate, x_rate), order=0)).astype(np.uint8)
    new_spacing = (spacing[0] / x_rate, spacing[1] / y_rate, spacing[2])

    return resample_img_array, resample_label_array, new_spacing


def nii_spacing_resample(input_array, spacing_tuple):
    img_array = np.einsum('zyx->xyz', input_array)

    if spacing_tuple[1] > spacing_tuple[0]:  # rate must > 1 (Y > X) - UPSAMPLE y
        tmp_array = (ndimage.zoom(img_array, (1, (spacing_tuple[1] / spacing_tuple[0]), 1), order=3)).astype(np.uint8)
        new_spacing = (spacing_tuple[0], spacing_tuple[0], spacing_tuple[2])
    else:
        tmp_array = (ndimage.zoom(img_array, ((spacing_tuple[0] / spacing_tuple[1]), 1, 1), order=3)).astype(np.uint8)
        new_spacing = (spacing_tuple[1], spacing_tuple[1], spacing_tuple[2])

    resampled_array = np.einsum('xyz->zyx', tmp_array)  # (t x y z) --> (z y x t)

    return resampled_array, new_spacing


def nrrd_spacing_resample(input_array, spacing_tuple):
    img_array = np.einsum('ijkl->lkji', input_array)  # (z y x t) --> (t x y z)
    seq_length = img_array.shape[0]
    print('---------------- start resampling the nrrd data ----------------------')
    if spacing_tuple[1] > spacing_tuple[0]:  # rate must > 1 (Y > X) - upsample Y
        tmp_array = np.zeros((img_array.shape[0],
                              img_array.shape[1],
                              np.uint8(np.round(img_array.shape[2] * (spacing_tuple[1] / spacing_tuple[0]))),
                              img_array.shape[3])).astype(np.uint8)

        for i in range(img_array.shape[0]):
            single_start = time()
            resample_img_array_i = (
                ndimage.zoom(img_array[i, :, :, :], (1, (spacing_tuple[1] / spacing_tuple[0]), 1), order=3)).astype(
                np.uint8)
            tmp_array[i, :, :, :] = resample_img_array_i
            print('Resample Frame_No', i, '/', seq_length, 'costs', "%.3f" % (time() - single_start), 'seconds')
        new_spacing = (spacing_tuple[0], spacing_tuple[0], spacing_tuple[2])

    else:
        tmp_array = np.zeros((img_array.shape[0],
                              np.uint8(img_array.shape[1] * (spacing_tuple[0] / spacing_tuple[1])),
                              img_array.shape[2],
                              img_array.shape[3])).astype(np.uint8)

        for i in range(img_array.shape[0]):
            single_start = time()
            resample_img_array_i = (
                ndimage.zoom(img_array[i, :, :, :], ((spacing_tuple[0] / spacing_tuple[1]), 1, 1), order=3)).astype(
                np.uint8)
            tmp_array[i, :, :, :] = resample_img_array_i
            print('Resample Frame_No.', i, '/', seq_length, 'costs', "%.3f" % (time() - single_start), 'seconds')
        new_spacing = (spacing_tuple[1], spacing_tuple[1], spacing_tuple[2])

    resampled_array = np.einsum('ijkl->lkji', tmp_array)  # (t x y z) --> (z y x t)

    return resampled_array, new_spacing


if __name__ == '__main__':
    folder_pth = '/media/zdc/zjp/datasets/data_Bscan/nii_files/val/sequence_img/sequence_img_enhanced/'
    save_folder_pth = '/media/zdc/zjp/datasets/data_Bscan/nii_files/val/sequence_img/sequence_img_enhanced_resampled/'

    single_name = '15-231--1-FuYaPingZhiNu.seq.nrrd'

    mode = 1
    # -------------- one folder ------------------#
    if mode == 0:
        ori_time = time()
        for index, filename in enumerate(os.listdir(folder_pth)):
            start_time = time()

            file_pth = os.path.join(folder_pth, filename)
            save_pth = os.path.join(save_folder_pth, filename)
            single_img = sitk.ReadImage(file_pth)
            spacing_tuple = single_img.GetSpacing()
            single_img_array = sitk.GetArrayFromImage(single_img)  # (z y x t)

            resample_array = nrrd_spacing_resample(single_img_array, spacing_tuple)

            # resampled_img_array = np.einsum('ijkl->lkji', resampled_array)  # (z y x t) --> (t x y z)
            resampled_img = sitk.GetImageFromArray(resample_array)  # (z y x t) return to the shape just like the input

            resampled_img.SetDirection(single_img.GetDirection())
            resampled_img.SetOrigin(single_img.GetOrigin())
            resampled_img.SetSpacing((spacing_tuple[0], spacing_tuple[0], spacing_tuple[2]))

            sitk.WriteImage(resampled_img, save_pth)
            cost_one_nrrd = time() - start_time
            print(filename, 'cost: ', cost_one_nrrd, 'seconds')

        total_cost = time() - ori_time
        print('total cost', total_cost, 'seconds')

    # -------------- one img ------------------#
    if mode == 1:
        start_time = time()
        filename = single_name
        file_pth = os.path.join(folder_pth, filename)
        save_pth = os.path.join(save_folder_pth, filename)
        single_img = sitk.ReadImage(file_pth)
        spacing_tuple = single_img.GetSpacing()
        single_img_array = sitk.GetArrayFromImage(single_img)  # (z y x t)

        resample_array = nrrd_spacing_resample(single_img_array, spacing_tuple)  # (z y x t )

        # resampled_img_array = np.einsum('ijkl->lkji', resampled_array)  # (z y x t) --> (t x y z)
        resampled_img = sitk.GetImageFromArray(resample_array)

        resampled_img.SetDirection(single_img.GetDirection())
        resampled_img.SetOrigin(single_img.GetOrigin())
        resampled_img.SetSpacing((spacing_tuple[0], spacing_tuple[0], spacing_tuple[2]))

        sitk.WriteImage(resampled_img, save_pth)
        cost_one_nrrd = time() - start_time
        print(filename, 'cost: ', cost_one_nrrd, 'seconds')
