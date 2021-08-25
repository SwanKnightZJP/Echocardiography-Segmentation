"""
    evaluator.evaluate
    evaluator.summerize
"""
import os
import cv2
import numpy as np
# from lib.utils.eval_utils import evaluate_single_prediction

# --- #
from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.loss.Loss_function import DiceLoss, CELoss, GDiceLoss, Dice_CE_Loss
import tqdm
import torch
import torch.nn as nn
# --- #

# _loss_factory = {
#     'GDice': GDiceLoss,
#     'Dice': DiceLoss,
#     'CE': CELoss,
#     'Dice_CE': Dice_CE_Loss,
# }


class Evaluator:
    def __init__(self, cfg):
        self.results = []
        self.num_class = cfg.num_class


    def evaluate(self, output, batch):
        pre_tensor = output["map"]  # B c x y z
        gt_tensor = batch["label"]  # b 1 x y z

        # comp_pred_tensor = output["compress"]  # B nxyz
        # comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(
        #     dim=1).long()

        gt_squeezed = gt_tensor[:, 0].long()

        defined_loss = nn.CrossEntropyLoss()
        CE_loss = defined_loss(pre_tensor, gt_squeezed)

        # output_array = pre_tensor.squeeze().cpu().detach().numpy()          # ndarray([2, 112, 112, 112])
        # pre_seg_array = np.argmax(output_array, axis=0).astype(np.uint8)  # ndarray([112, 112, 112])
        # gt_array = gt_tensor.squeeze().squeeze().cpu().detach().numpy()  # x y z  [1,112,112,112]
        output_tensor = pre_tensor.squeeze()  # ndarray([c, 112, 112, 112])
        pre_seg_array = torch.argmax(output_tensor, axis=0)  # ndarray([112, 112, 112])
        gt_tmp_tensor = gt_tensor.squeeze().squeeze()  # x y z  [1,112,112,112]

        pre_array_c0 = pre_tensor[:, 0, :, :, :].squeeze()  # float
        pre_array_c1 = pre_tensor[:, 1, :, :, :].squeeze()  # float
        pre_array_c2 = pre_tensor[:, 2, :, :, :].squeeze()  # float

        gt_00, gt_01, gt_02 = torch.zeros(gt_tmp_tensor.shape).cuda(), torch.zeros(gt_tmp_tensor.shape).cuda(), torch.zeros(gt_tmp_tensor.shape).cuda()
        gt_01[gt_tmp_tensor == 1] = 1
        gt_02[gt_tmp_tensor == 2] = 1
        gt_00[gt_tmp_tensor == 0] = 1

        epsilon = np.spacing(1)
        intersection = []
        union = []
        for i in range(cfg.num_class + 1):    # range 1, 2
            tmp_label = torch.zeros(gt_tmp_tensor.shape).cuda()
            tmp_pred = torch.zeros(pre_seg_array.shape).cuda()
            tmp_label[gt_tmp_tensor == i] = 1
            tmp_pred[pre_seg_array == i] = 1
            intersection.append(torch.sum(tmp_label * tmp_pred))
            union.append(torch.sum(tmp_label) + torch.sum(tmp_pred))

        dice_score_ftotal = 2 * sum(intersection) / (sum(union) + epsilon)  # int without BG

        dice_f01 = 2 * intersection[1] / (union[1] + epsilon)
        dice_f02 = 2 * intersection[2] / (union[2] + epsilon)

        tp1 = pre_array_c1 * gt_01
        fp1 = pre_array_c1 * (1 - gt_01)
        fn1 = (1 - pre_array_c1) * gt_01
        tp1 = tp1.sum()
        fp1 = fp1.sum()
        fn1 = fn1.sum()
        dice01 = (2 * tp1 + epsilon) / (2 * tp1 + fp1 + fn1 + epsilon)  # float

        tp2 = pre_array_c2 * gt_02
        fp2 = pre_array_c2 * (1 - gt_02)
        fn2 = (1 - pre_array_c2) * gt_02
        tp2 = tp2.sum()
        fp2 = fp2.sum()
        fn2 = fn2.sum()
        dice02 = (2 * tp2 + epsilon) / (2 * tp2 + fp2 + fn2 + epsilon)

        tp0 = pre_array_c0 * gt_00
        fp0 = pre_array_c0 * (1 - gt_00)
        fn0 = (1 - pre_array_c0) * gt_00
        tp0 = tp0.sum()
        fp0 = fp0.sum()
        fn0 = fn0.sum()
        dice00 = (2 * tp0 + epsilon) / (2 * tp0 + fp0 + fn0 + epsilon)

        self.results.append((CE_loss, dice_score_ftotal, dice01, dice02, dice00, dice_f01, dice_f02))

    def summarize(self):
        total_ce = total_dice = dice_01 = dice_02 = dice_00 = dice_f01 = dice_f02 = 0
        num = len(self.results)
        for cell in self.results:
            total_ce += cell[0]
            total_dice += cell[1]
            dice_01 += cell[2]
            dice_02 += cell[3]
            dice_00 += cell[4]
            dice_f01 += cell[5]
            dice_f02 += cell[6]

        ce_score = total_ce / num
        dice_score = total_dice / num

        dice_score_01 = dice_01 / num
        dice_score_02 = dice_02 / num
        dice_score_00 = dice_00 / num

        dice_f01 = dice_f01 / num
        dice_f02 = dice_f02 / num


        return {'ce_loss': ce_score, 'dice_score': dice_score, 'dice_01': dice_score_01, 'dice_02': dice_score_02,
                'dice_bg': dice_score_00, 'dice_f_01': dice_f01, 'dice_f02': dice_f02}
