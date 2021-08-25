import numpy as np
import torch


def cal_dice(pred, gt, num_class):   # hard dice
    epsilon = np.spacing(1)
    if torch.is_tensor(pred):
        pred = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.data.cpu().numpy()

    intersection = []
    union = []
    dice_score_class = []
    for i in range(0, num_class + 1):  # range 1 2   # without bg
        tmp_label = np.zeros(gt.shape).astype(np.uint8)
        tmp_pred = np.zeros(pred.shape).astype(np.uint8)
        tmp_label[gt == i] = 1
        tmp_pred[pred == i] = 1
        intersection.append(np.sum(tmp_label * tmp_pred))
        union.append(np.sum(tmp_label) + np.sum(tmp_pred))
        dice_score_class.append(2 * intersection[i] / (union[i] + epsilon))

    forward_dice_score = 2 * sum(intersection[1:3]) / (sum(union[1:3]) + epsilon)
    totoal_dice_score = 2 * sum(intersection) / (sum(union) + epsilon)
    dice_score_class.append(totoal_dice_score)
    return forward_dice_score, dice_score_class


def cal_iou(pred, gt, num_class):
    epsilon = np.spacing(1)
    if torch.is_tensor(pred):
        pred = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.data.cpu().numpy()

    intersection = []
    union = []
    # for i in range(num_class):






