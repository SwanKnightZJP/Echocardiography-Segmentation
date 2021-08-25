"""

    pre-process codes for the original data set

"""


import cv2
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils.sobel_edge import sobel


def single_enhance(single_img):

    img_spacing = single_img.GetSpacing()
    single_image_array = sitk.GetArrayFromImage(single_img)
    test_ds_zyx = single_image_array
    test_ds_xyz = np.einsum('zyx->xyz', test_ds_zyx)
    test_ds_xyz = test_ds_xyz.astype(np.uint8)

    edg_yz = np.zeros(test_ds_xyz.shape).astype(np.uint8)
    edg_xz = np.zeros(test_ds_xyz.shape).astype(np.uint8)
    edg_xy = np.zeros(test_ds_xyz.shape).astype(np.uint8)

    mid_yz = np.zeros(test_ds_xyz.shape).astype(np.uint8)
    mid_xz = np.zeros(test_ds_xyz.shape).astype(np.uint8)
    mid_xy = np.zeros(test_ds_xyz.shape).astype(np.uint8)

    # ------------------------------ ehnaced ----------------------------------- #
    for i in range(test_ds_xyz.shape[0]):
        array_tmp = cv2.medianBlur((test_ds_xyz[i, :, :]), 9)
        _, _, _, sobel_median = sobel(array_tmp)
        edg_yz[i, :, :] = sobel_median
        mid_yz[i, :, :] = array_tmp
    # obtain edg_xz
    for j in range(test_ds_xyz.shape[1]):
        array_tmp = cv2.medianBlur((test_ds_xyz[:, j, :]), 9)
        _, _, _, sobel_median = sobel(array_tmp)
        edg_xz[:, j, :] = sobel_median
        mid_xz[:, j, :] = array_tmp
    # obtain edg_xy
    for k in range(test_ds_xyz.shape[2]):
        array_tmp = cv2.medianBlur((test_ds_xyz[:, :, k]), 9)
        _, _, _, sobel_median = sobel(array_tmp)
        edg_xy[:, :, k] = sobel_median
        mid_xy[:, :, k] = array_tmp

    fuse01 = (edg_yz / 3 + edg_xz / 3 + edg_xy / 3).astype(np.uint8)
    fuse02 = (mid_yz / 3 + mid_xz / 3 + mid_xy / 3).astype(np.uint8)

    fuse03 = fuse01 + fuse02 / 2
    fuse03[fuse03 > 255] = 255
    fuse03 = fuse03.astype(np.uint8)  # xyz

    fuse04 = np.einsum('xyz->zyx', fuse03)
    save_img = sitk.GetImageFromArray(fuse04)

    save_img.SetDirection(single_img.GetDirection())
    save_img.SetOrigin(single_img.GetOrigin())
    save_img.SetSpacing(img_spacing)

    return save_img


def single_resample_label(single_img):

    img_spacing = single_img.GetSpacing()  #

    single_image_array = sitk.GetArrayFromImage(single_img)  # zyx
    test_ds_xyz = np.einsum('zyx->xyz', single_image_array)

    # ---------------- resample ------------------------------------------------------ #
    if img_spacing[1] > img_spacing[0]:
        new_spacing = (img_spacing[0] * 1.25, img_spacing[0] * 1.25, img_spacing[2])
        resample_tmp = (ndimage.zoom(test_ds_xyz, (0.8, (0.8 * img_spacing[1] / img_spacing[0]), 1), order=0)).astype(np.uint8)

    else:
        new_spacing = (img_spacing[0] * 1.25, img_spacing[0] * 1.25, img_spacing[2])
        resample_tmp = (ndimage.zoom(test_ds_xyz, ((0.8 * img_spacing[0] / img_spacing[1]), 0.8, 1), order=0)).astype(np.uint8)

    # if img_spacing[1] > img_spacing[0]:
    #     new_spacing = (img_spacing[0] * 1.6, img_spacing[0] * 1.6, img_spacing[2])
    #     resample_tmp = (ndimage.zoom(test_ds_xyz, (0.625, (0.625 * img_spacing[1] / img_spacing[0]), 1), order=1)).astype(np.uint8)
    #
    # else:
    #     new_spacing = (img_spacing[0] * 1.6, img_spacing[0] * 1.6, img_spacing[2])
    #     resample_tmp = (ndimage.zoom(test_ds_xyz, ((0.625 * img_spacing[0] / img_spacing[1]), 0.625, 1), order=1)).astype(np.uint8)

    save_array = np.einsum('xyz->zyx', resample_tmp)
    save_img = sitk.GetImageFromArray(save_array)

    save_img.SetDirection(single_img.GetDirection())
    save_img.SetOrigin(single_img.GetOrigin())
    save_img.SetSpacing(new_spacing)

    return save_img


def single_resample_img(single_img):

    img_spacing = single_img.GetSpacing()  #

    single_image_array = sitk.GetArrayFromImage(single_img).astype(np.uint8)  # zyx
    test_ds_xyz = np.einsum('zyx->xyz', single_image_array)

    # ---------------- resample ------------------------------------------------------ #
    if img_spacing[1] > img_spacing[0]:
        new_spacing = (img_spacing[0] * 1.25, img_spacing[0] * 1.25, img_spacing[2])
        resample_tmp = (ndimage.zoom(test_ds_xyz, (0.8, (0.8 * img_spacing[1] / img_spacing[0]), 1), order=3)).astype(np.uint8)

    else:
        new_spacing = (img_spacing[0] * 1.25, img_spacing[0] * 1.25, img_spacing[2])
        resample_tmp = (ndimage.zoom(test_ds_xyz, ((0.8 * img_spacing[0] / img_spacing[1]), 0.8, 1), order=3)).astype(np.uint8)

    # if img_spacing[1] > img_spacing[0]:
    #     new_spacing = (img_spacing[0] * 1.6, img_spacing[0] * 1.6, img_spacing[2])
    #     resample_tmp = (ndimage.zoom(test_ds_xyz, (0.625, (0.625 * img_spacing[1] / img_spacing[0]), 1), order=3)).astype(np.uint8)
    #
    # else:
    #     new_spacing = (img_spacing[0] * 1.6, img_spacing[0] * 1.6, img_spacing[2])
    #     resample_tmp = (ndimage.zoom(test_ds_xyz, ((0.625 * img_spacing[0] / img_spacing[1]), 0.625, 1), order=3)).astype(np.uint8)

    save_array = np.einsum('xyz->zyx', resample_tmp)
    # cv2.imshow('axial', save_array[74, :, :])
    # cv2.imshow('coronal', save_array[:, 98, :])
    # cv2.imshow('sagittal', save_array[:, :, 104])
    # cv2.waitKey()
    save_img = sitk.GetImageFromArray(save_array)

    save_img.SetDirection(single_img.GetDirection())
    save_img.SetOrigin(single_img.GetOrigin())
    save_img.SetSpacing(new_spacing)

    return save_img