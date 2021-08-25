"""

    linked at lib/train/trainers/task.py

    input: pred & batch
      batch: {"input": input_tensor(B, 1, x, y, z), "label": label_batch(B, 1, x, y, z)}
      pre: {"map": pre_tensor(B, n, x, y, z), "compress": compress_pre_tensor(B*z*y*z, n)}


"""
import torch
import torch.nn as nn
import numpy as np
from nnunet.utilities.tensor_utilities import sum_tensor

'''
sample :

class DefinedLossFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self,):
        Loss = "defined"
        return Loss
        
'''


def get_tp_fp_fn_tn(pred, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(pred.size())))

    shp_x = pred.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(pred.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if pred.device.type == "cuda":
                y_onehot = y_onehot.cuda(pred.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = pred * y_onehot
    fp = pred * (1 - y_onehot)
    fn = (1 - pred) * y_onehot
    tn = (1 - pred) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        #  # if batch_dice: [batch_TP_class01, batch_TP_class02, batch_TP_class03]
        #  # else: [[b0_TP_class01, b0_TP_class02, b0_class03],[b1_TP_class01, b1_TP_class02, b1_class03] ...]
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


# GDice with BG + CE  rewritten @ 0712
class GDiceBG_CE_Loss(nn.Module):
    # ---------------------- old version before 0712 ----------------------- #
    # def __init__(self, cfg):
    #     super().__init__()
    #     self.num_class = cfg.num_class
    #     self.size_x = cfg.data.patch_size[0]
    #     self.size_y = cfg.data.patch_size[1]
    #     self.size_z = cfg.data.patch_size[2]
    #
    # def forward(self, pred, batch):
    #     """
    #     :param pred: "compress": compress_pre_tensor(B*112*112*112, 2)
    #     :param batch: "label": label_batch(B, 1, 112, 112, 112)
    #     :return: loss([]) singel value
    #     """
    #
    #     comp_pred_tensor = pred["compress"]
    #     gt_tensor = batch["label"]
    #     pre_tensor = pred["map"]  # [1,3,x,y,z]
    #     epsilon = np.spacing(1)
    #
    #     # Dice Loss
    #     class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
    #     class_weight = []
    #
    #     for class_index in range(self.num_class + 1):
    #         temp_label = torch.zeros(gt_tensor.size())
    #         temp_label[gt_tensor == class_index] = 1
    #         class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
    #         w_tmp = 1 / ((class_gt[:, class_index, :, :, :].sum().item())**2 + epsilon)
    #         class_weight.append(w_tmp)
    #
    #     class_gt = class_gt.cuda()  # (B, 2, W, H, Z)
    #
    #     tp = pre_tensor * class_gt  # [1,3,x,y,z]
    #     fp = pre_tensor * (1 - class_gt)
    #     fn = (1 - pre_tensor) * class_gt
    #
    #     tp = torch.sum(tp[:, 0, :, :, :]/class_weight[0] + tp[:,1,:,:,:]/class_weight[1] + tp[:,2,:,:,:]/class_weight[2])
    #     fp = torch.sum(fp[:, 0, :, :, :]/class_weight[0] + fp[:,1,:,:,:]/class_weight[1] + fp[:,2,:,:,:]/class_weight[2])
    #     fn = torch.sum(fn[:, 0, :, :, :]/class_weight[0] + fn[:,1,:,:,:]/class_weight[1] + fn[:,2,:,:,:]/class_weight[2])
    #
    #     dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    #     loss_dice = 1 - dice
    #
    #     # CE loss
    #     CE_loss = nn.CrossEntropyLoss()
    #     comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(
    #         dim=1).long()
    #     loss_ce = CE_loss(comp_pred_tensor, comp_gt_tensor)
    #
    #     total_loss = loss_ce + loss_dice
    def __init__(self, cfg, square_volume=False, batch_dice=False):
        super().__init__()
        self.square_volume = square_volume
        self.batch_dice = batch_dice
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, batch):
        """
        :param pred: "map": pre_tensor(B, 2, x, y, z)
        :param batch: "label": label_batch(B, 1, x, y, z)
        :return: loss([]) singel value
        """
        pre_tensor = pred["map"]
        gt_tensor = batch["label"]

        gt_squeezed = gt_tensor[:, 0].long()
        ce_loss = self.loss(pre_tensor, gt_squeezed)

        epsilon = np.spacing(1)
        class_gt = torch.zeros(pre_tensor.size()).cuda()
        class_gt = class_gt.scatter_(1, gt_tensor.long(), 1).cuda()   # gt_one_hot [b c x y z]
        if self.batch_dice:
            axes = [0] + list(range(2, len(pre_tensor.size())))  # [0 2 3 4] = [b x y z]  -- the num_tp = Class
        else:
            axes = list(range(2, len(pre_tensor.size())))        # [2 3 4]   = [x y z]    -- the num_tp = batch * class

        tp, fp, fn, _ = get_tp_fp_fn_tn(pre_tensor, class_gt, axes)  # [class0 class1 class2]

        volumes = sum_tensor(class_gt, axes) + epsilon    #
        if self.square_volume:
            volumes = volumes ** 2

        tp = tp / volumes  # if batch: [c], else: [b, c]
        fp = fp / volumes
        fn = fn / volumes

        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)   # teturn one value
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        dice = dice.mean()
        dice_loss = 1 - dice

        total_loss = ce_loss + dice_loss   # Gdice_loss range 0.7 - 0.04  CE loss range 1.01 - 0.56
        return total_loss


# class GDiceBG_GCE_Loss(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.num_class = cfg.num_class
#         self.size_x = cfg.data.patch_size[0]
#         self.size_y = cfg.data.patch_size[1]
#         self.size_z = cfg.data.patch_size[2]
#
#     def forward(self, pred, batch):
#         """
#         :param pred: "compress": compress_pre_tensor(B*112*112*112, 2)
#         :param batch: "label": label_batch(B, 1, 112, 112, 112)
#         :return: loss([]) singel value
#         """
#
#         comp_pred_tensor = pred["compress"]
#         gt_tensor = batch["label"]
#         pre_tensor = pred["map"]  # [1,3,x,y,z]
#         epsilon = np.spacing(1)
#
#         # Dice Loss
#         class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
#         class_weight = []
#
#         for class_index in range(self.num_class + 1):
#             temp_label = torch.zeros(gt_tensor.size())
#             temp_label[gt_tensor == class_index] = 1
#             class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
#             w_tmp = 1 / ((class_gt[:, class_index, :, :, :].sum().item())**2 + epsilon)
#             class_weight.append(w_tmp)
#
#         # import cv2
#         # tmp_bg_array = class_gt[0, 0, :, :, 80].detach().cpu().numpy()
#         # tmp_lv_array = class_gt[0, 1, :, :, 80].detach().cpu().numpy()
#         # tmp_rv_array = class_gt[0, 2, :, :, 80].detach().cpu().numpy()
#         # cv2.imshow('tmp_bg_array', (tmp_bg_array * 127).astype(np.uint8))
#         # cv2.imshow('tmp_lv_array', (tmp_lv_array * 127).astype(np.uint8))
#         # cv2.imshow('tmp_rv_array', (tmp_rv_array * 127).astype(np.uint8))
#         # cv2.waitKey()
#
#         class_gt = class_gt.cuda()  # (B, 2, W, H, Z)
#
#         tp = pre_tensor * class_gt  # [1,3,x,y,z]
#         fp = pre_tensor * (1 - class_gt)
#         fn = (1 - pre_tensor) * class_gt
#
#         # import cv2
#         # tmp_bg_tp = tp[0, 0, :, :, 80].detach().cpu().numpy()
#         # tmp_lv_tp = tp[0, 1, :, :, 80].detach().cpu().numpy()
#         # tmp_rv_tp = tp[0, 2, :, :, 80].detach().cpu().numpy()
#         #
#         # tmp_bg_fp = fp[0, 0, :, :, 80].detach().cpu().numpy()
#         # tmp_lv_fp = fp[0, 1, :, :, 80].detach().cpu().numpy()
#         # tmp_rv_fp = fp[0, 2, :, :, 80].detach().cpu().numpy()
#         #
#         # tmp_bg_fn = fn[0, 0, :, :, 80].detach().cpu().numpy()
#         # tmp_lv_fn = fn[0, 1, :, :, 80].detach().cpu().numpy()
#         # tmp_rv_fn = fn[0, 2, :, :, 80].detach().cpu().numpy()
#         #
#         # cv2.imshow('tmp_bg_tp', (tmp_bg_tp * 127).astype(np.uint8))
#         # cv2.imshow('tmp_lv_tp', (tmp_lv_tp * 127).astype(np.uint8))
#         # cv2.imshow('tmp_rv_tp', (tmp_rv_tp * 127).astype(np.uint8))
#         #
#         # cv2.imshow('tmp_bg_fp', (tmp_bg_fp * 127).astype(np.uint8))
#         # cv2.imshow('tmp_lv_fp', (tmp_lv_fp * 127).astype(np.uint8))
#         # cv2.imshow('tmp_rv_fp', (tmp_rv_fp * 127).astype(np.uint8))
#         #
#         # cv2.imshow('tmp_bg_fn', (tmp_bg_fn * 127).astype(np.uint8))
#         # cv2.imshow('tmp_lv_fn', (tmp_lv_fn * 127).astype(np.uint8))
#         # cv2.imshow('tmp_rv_fn', (tmp_rv_fn * 127).astype(np.uint8))
#         #
#         # cv2.waitKey()
#
#         tp = torch.sum(tp[:, 0, :, :, :]/class_weight[0] + tp[:,1,:,:,:]/class_weight[1] + tp[:,2,:,:,:]/class_weight[2])
#         fp = torch.sum(fp[:, 0, :, :, :]/class_weight[0] + fp[:,1,:,:,:]/class_weight[1] + fp[:,2,:,:,:]/class_weight[2])
#         fn = torch.sum(fn[:, 0, :, :, :]/class_weight[0] + fn[:,1,:,:,:]/class_weight[1] + fn[:,2,:,:,:]/class_weight[2])
#
#         dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
#         loss_dice = 1 - dice
#
#         # CE loss
#         CE_loss = nn.CrossEntropyLoss(weight=torch.tensor([class_weight[0], class_weight[1], class_weight[2]]).float().cuda())
#         comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(
#             dim=1).long()
#         loss_ce = CE_loss(comp_pred_tensor, comp_gt_tensor)
#
#         total_loss = loss_ce + loss_dice
#
#         # test_a = pre_tensor[:, 1, :, :, :]
#         # test_b = pre_tensor[:, 2, :, :, :]
#         # array_r = test_a.squeeze().cpu().detach().numpy()
#         # array_b = test_b.squeeze().cpu().detach().numpy()
#         # pre_l = (array_b * 125).astype(np.uint8)
#         # pre_r = (array_r * 125).astype(np.uint8)
#         #
#         # output_array = pre_tensor.squeeze().cpu().detach().numpy()
#         # gt_array = gt_tensor.squeeze().squeeze().cpu().detach().numpy()
#         # out_array = np.argmax(output_array, axis=0)
#         #
#         # out_array = (out_array * 125).astype(np.uint8)
#         # gt_array = (gt_array * 125).astype(np.uint8)
#         # import cv2
#         # cv2.imshow('r', pre_r[:, :, 80])
#         # cv2.imshow('l', pre_l[:, :, 80])
#         # cv2.imshow("out", out_array[:, :, 80])
#         # cv2.imshow("gt", gt_array[:, :, 80])
#         # cv2.waitKey()
#
#         return total_loss

# Dice_score + CE loss with background
class DiceBG_CE_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.loss = nn.CrossEntropyLoss()
        self.size_x = cfg.data.patch_size[0]
        self.size_y = cfg.data.patch_size[1]
        self.size_z = cfg.data.patch_size[2]

    def forward(self, pred, batch):
        """
        :param pred: "compress": compress_pre_tensor(B*112*112*112, 2)
        :param batch: "label": label_batch(B, 1, 112, 112, 112)
        :return: loss([]) singel value
        """
        # 计算交叉熵损失值
        comp_pred_tensor = pred["compress"]
        gt_tensor = batch["label"]
        comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(dim=1).long()
        loss_ce = self.loss(comp_pred_tensor, comp_gt_tensor)

        pre_tensor = pred["map"]
        epsilon = np.spacing(1)
        class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
        for class_index in range(self.num_class + 1):
            temp_label = torch.zeros(gt_tensor.size())
            temp_label[gt_tensor == class_index] = 1
            class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
        class_gt = class_gt.cuda()  # (B, 2, W, H, Z)

        tp = pre_tensor * class_gt
        fp = pre_tensor * (1 - class_gt)
        fn = (1 - pre_tensor) * class_gt

        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        loss_dice = 1 - dice

        total_loss = loss_ce + loss_dice

        return total_loss


# Dice_score + CE loss without background
class Dice_CE_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.loss = nn.CrossEntropyLoss()
        self.size_x = cfg.data.patch_size[0]
        self.size_y = cfg.data.patch_size[1]
        self.size_z = cfg.data.patch_size[2]

    def forward(self, pred, batch):
        """
        :param pred: "compress": compress_pre_tensor(B*112*112*112, 2)
        :param batch: "label": label_batch(B, 1, 112, 112, 112)
        :return: loss([]) singel value
        """
        # 计算交叉熵损失值
        comp_pred_tensor = pred["compress"]
        gt_tensor = batch["label"]
        comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(dim=1).long()
        loss_ce = self.loss(comp_pred_tensor, comp_gt_tensor)

        pre_tensor = pred["map"]
        epsilon = np.spacing(1)
        class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
        for class_index in range(1, self.num_class + 1):
            temp_label = torch.zeros(gt_tensor.size())
            temp_label[gt_tensor == class_index] = 1
            class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
        class_gt = class_gt.cuda()  # (B, 2, W, H, Z)

        tp = pre_tensor * class_gt
        fp = pre_tensor * (1 - class_gt)
        fn = (1 - pre_tensor) * class_gt

        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        loss_dice = 1 - dice

        total_loss = loss_ce + loss_dice

        # test_a = pre_tensor[:, 1, :, :, :]
        # test_b = pre_tensor[:, 2, :, :, :]
        # array_r = test_a.squeeze().cpu().detach().numpy()
        # array_b = test_b.squeeze().cpu().detach().numpy()
        # pre_l = (array_b * 125).astype(np.uint8)
        # pre_r = (array_r * 125).astype(np.uint8)
        #
        # output_array = pre_tensor.squeeze().cpu().detach().numpy()
        # gt_array = gt_tensor.squeeze().squeeze().cpu().detach().numpy()
        # out_array = np.argmax(output_array, axis=0)
        #
        # out_array = (out_array * 125).astype(np.uint8)
        # gt_array = (gt_array * 125).astype(np.uint8)
        # import cv2
        # cv2.imshow('r', pre_r[:, :, 80])
        # cv2.imshow('l', pre_l[:, :, 80])
        # cv2.imshow("out", out_array[:, :, 80])
        # cv2.imshow("gt", gt_array[:, :, 80])
        # cv2.waitKey()

        return total_loss


# CrossEntropy Loss
class CELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, batch):
        """
        :param pred: "compress": compress_pre_tensor(B*112*112*112, C)
        :param batch: "label": label_batch(B, 1, X, Y, Z)
            gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(dim=1).long()
            1. turn B 1 X Y Z into [B X Y Z 1]  (value range(0-2))
            2. turn [B X Y Z 1] into [B*X*Y*Z 1]
            3. turn [B*X*Y*Z 1] into [B*X*Y*Z]
        :return: loss([]) singel value
        """
        # 计算交叉熵损失值
        gt_tensor = batch["label"]  # [b 1 x y z]
        pre_tensor = pred["map"]  # [b c x y z]

        # ----- old ------------------------------ #
        # comp_pred_tensor = pred["compress"]  # [2007040, 3]
        # comp_gt_tensor = gt_tensor.permute(0, 2, 3, 4, 1).contiguous().view(gt_tensor.numel() // 1, 1).squeeze(
        #     dim=1).long()

        # loss = self.loss(comp_pred_tensor, comp_gt_tensor)
        # ----- new ------------------------------ #
        gt_squeezed = gt_tensor[:, 0].long()
        loss = self.loss(pre_tensor, gt_squeezed)

        return loss
# class DiceLoss(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.num_class = cfg.num_class
#         self.size_x = cfg.data.patch_size[0]
#         self.size_y = cfg.data.patch_size[1]
#         self.size_z = cfg.data.patch_size[2]
#
#     def forward(self, pred, batch):
#         """
#         :param pred: "map": pre_tensor(B, n, x, y, z)
#         :param batch: "label": label_batch(B, 1, x, y, z)
#         :return: loss([]) singel value
#         """
#         pre_tensor = pred["map"]  # b class x y z
#         pred_tensor = torch.argmax(pre_tensor, axis=1).unsqueeze(dim=0)  #  b 1 x y z
#         gt_tensor = batch["label"]
#         epsilon = np.spacing(1)
#         class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
#         class_pre = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
#         for class_index in range(self.num_class + 1):
#             temp_gt = torch.zeros(gt_tensor.size())
#             temp_gt[gt_tensor == class_index] = 1
#             class_gt[:, class_index, :, :, :] = temp_gt[:, 0, :, :, :]
#
#             temp_pre = torch.zeros(gt_tensor.size())
#             temp_pre[pred_tensor == class_index] = 1
#             class_pre[:, class_index, :, :, :] = temp_pre[:, 0, :, :, :]
#
#         class_gt = class_gt.cuda()  # (B, 2, W, H, Z)
#         class_pre = class_pre.cuda()
#         numerator = 0.0
#         denominator = 0.0
#         for i in range(self.num_class + 1):
#             # w_tmp = 1 / ((class_gt[:, i, :, :, :].sum(dim=1).sum(dim=1).sum(dim=0).item())**2 + epsilon)
#             numerator_tmp = (class_pre[:, i, :, :, :] * class_gt[:, i, :, :, :]).sum()
#             denominator_tmp = class_pre[:, i, :, :, :].sum() + class_gt[:, i, :, :, :].sum()
#             numerator += numerator_tmp
#             denominator += denominator_tmp
#         dice_score = 2 * numerator / (denominator + epsilon)
#         dice_loss = 1 - dice_score
#         return dice_loss


# Dice_score with background

class DiceLoss(nn.Module):   # Dice_score with background
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.size_x = cfg.data.patch_size[0]
        self.size_y = cfg.data.patch_size[1]
        self.size_z = cfg.data.patch_size[2]

    def forward(self, pred, batch):
        """
        :param pred: "map": pre_tensor(B, n, x, y, z)
        :param batch: "label": label_batch(B, 1, x, y, z)
        :return: loss([]) singel value
        """
        pre_tensor = pred["map"]
        gt_tensor = batch["label"]
        epsilon = np.spacing(1)

        class_gt = torch.zeros(pre_tensor.size()).cuda()
        class_gt = class_gt.scatter_(1, gt_tensor.long(), 1)

        # class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
        # for class_index in range(self.num_class + 1):  # even from 1 but the label from 0 to 2
        #     temp_label = torch.zeros(gt_tensor.size())
        #     temp_label[gt_tensor == class_index] = 1
        #     class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
        # class_gt = class_gt.cuda()  # (B, 2, W, H, Z)

        tp = pre_tensor * class_gt
        fp = pre_tensor * (1 - class_gt)
        fn = (1 - pre_tensor) * class_gt

        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        dice_loss = 1 - dice
        # test_a = pre_tensor[:, 1, :, :, :]
        # test_b = pre_tensor[:, 2, :, :, :]
        # array_r = test_a.squeeze().cpu().detach().numpy()
        # array_b = test_b.squeeze().cpu().detach().numpy()
        # pre_l = (array_b * 125).astype(np.uint8)
        # pre_r = (array_r * 125).astype(np.uint8)
        #
        # output_array = pre_tensor.squeeze().cpu().detach().numpy()
        # out_array = np.argmax(output_array, axis=0)
        #
        # out_array = (out_array*125).astype(np.uint8)
        #
        # import cv2
        # cv2.imshow('r', pre_r[:, :, 80])
        # cv2.imshow('l', pre_l[:, :, 80])
        # cv2.imshow("out", out_array[:, :, 80])
        # cv2.waitKey()
        return dice_loss


class DiceLossNoBG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.size_x = cfg.data.patch_size[0]
        self.size_y = cfg.data.patch_size[1]
        self.size_z = cfg.data.patch_size[2]

    def forward(self, pred, batch):
        """
        :param pred: "map": pre_tensor(B, n, x, y, z)
        :param batch: "label": label_batch(B, 1, x, y, z)
        :return: loss([]) singel value
        """
        pre_tensor = pred["map"][:, 1:3, :, :, :]
        gt_tensor = batch["label"]
        epsilon = np.spacing(1)
        class_gt = torch.zeros((gt_tensor.size(0), self.num_class, self.size_x, self.size_y, self.size_z))
        for class_index in range(self.num_class):  # 1 2
            temp_label = torch.zeros(gt_tensor.size())
            temp_label[gt_tensor == class_index + 1] = 1
            class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
        class_gt = class_gt.cuda()  # (B, 2, W, H, Z)

        tp = pre_tensor * class_gt
        fp = pre_tensor * (1 - class_gt)
        fn = (1 - pre_tensor) * class_gt
        tn = (1 - pre_tensor) * (1 - class_gt)

        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        dice_loss = 1 - dice

        return dice_loss


class GDiceLoss(nn.Module):
    """
    Generalized Dice Loss
    """
    def __init__(self, cfg, square_volume=False, batch_dice=False):
        super().__init__()
        self.square_volume = square_volume
        self.batch_dice = batch_dice

    def forward(self, pred, batch):
        """
        :param pred: "map": pre_tensor(B, 2, x, y, z)
        :param batch: "label": label_batch(B, 1, x, y, z)
        :return: loss([]) singel value
        """
        pre_tensor = pred["map"]
        gt_tensor = batch["label"]
        epsilon = np.spacing(1)

        # ----------------------- 0707 ------------------------#
        class_gt = torch.zeros(pre_tensor.size()).cuda()
        class_gt = class_gt.scatter_(1, gt_tensor.long(), 1).cuda()   # gt_one_hot [b c x y z]
        # numerator = 0.0
        # denominator = 0.0
        # for i in range(self.num_class + 1):
        #     w_tmp = 1 / ((class_gt[:, i, :, :, :].sum().item())**2 + epsilon)
        #     numerator_tmp = (pre_tensor[:, i, :, :, :] * class_gt[:, i, :, :, :]).sum()
        #     denominator_tmp = pre_tensor[:, i, :, :, :].sum() + class_gt[:, i, :, :, :].sum()
        #     numerator += w_tmp * numerator_tmp
        #     denominator += w_tmp * denominator_tmp
        # dice_score = 2 * numerator / (denominator + epsilon)
        # dice_loss = 1 - dice_score
        # class_gt = torch.zeros((gt_tensor.size(0), self.num_class + 1, self.size_x, self.size_y, self.size_z))
        # for class_index in range(self.num_class + 1):
        #     temp_label = torch.zeros(gt_tensor.size())
        #     temp_label[gt_tensor == class_index] = 1
        #     class_gt[:, class_index, :, :, :] = temp_label[:, 0, :, :, :]
        # class_gt = class_gt.cuda()  # (B, 2, W, H, Z)
        # ----------------------- ---- ------------------------#
        if self.batch_dice:
            axes = [0] + list(range(2, len(pre_tensor.size())))  # [0 2 3 4] = [b x y z]  -- the num_tp = Class
        else:
            axes = list(range(2, len(pre_tensor.size())))        # [2 3 4]   = [x y z]    -- the num_tp = batch * class

        tp, fp, fn, _ = get_tp_fp_fn_tn(pre_tensor, class_gt, axes)  # [class0 class1 class2]

        volumes = sum_tensor(class_gt, axes) + epsilon    #
        if self.square_volume:
            volumes = volumes ** 2

        tp = tp / volumes  # if batch: [c], else: [b, c]
        fp = fp / volumes
        fn = fn / volumes

        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)   # teturn one value
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        dice = dice.mean()
        dice_loss = 1 - dice

        return dice_loss


'''
class GDiceLossFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_class = cfg.num_class
        self.size = cfg.data.patch_size
        # self.class_weight = [0.0001, 10.0, 6.0, 3.0, 3.0, 1.0, 0.2]

    def forward(self, output, batch):
        """
        c = 6 (1,2,3,4,5,6) calculate the loss with the bg
        :param pre:  predicted output (B, c, 401, 401)
        :param bone_mask: the attention of the bone (B, 1, 401, 401)
        :param gt: the label of the predict image (B, 1, 401, 401)
        :return: dice distance
        """
        pre1 = output['pre1']
        bone_mask, gt = batch['bone_mask'], batch['gt']
        epsilon = np.spacing(1)
        class_gt = torch.zeros((gt.size(0), self.num_class + 1, self.size, self.size))
        for class_index in range(self.num_class + 1):
            temp_label = torch.zeros(gt.size())
            temp_label[gt == class_index] = 1
            class_gt[:, class_index, :, :] = temp_label[:, 0, :, :]
        class_gt = class_gt.cuda()  # (B, 2, W, H)
        class_gt *= bone_mask
        # pre1 *= bone_mask
        numerator = 0.0
        denominator = 0.0

        for i in range(self.num_class + 1):
            w_tmp = 1 / ((class_gt[:, i, :, :].sum(dim=1).sum(dim=1).sum(dim=0).item())**2 + epsilon)
            # w_tmp = ((class_gt[:, i, :, :].sum(dim=1).sum(dim=1).sum(dim=0).item())**2 + epsilon) / 160801
            numerator_tmp = (pre1[:, i, :, :] * class_gt[:, i, :, :]).sum(dim=1).sum(dim=1).sum(dim=0)
            # denominator_tmp = pre1[:, i, :, :].sum(dim=1).sum(dim=1).sum(dim=0) + class_gt[:, i, :, :].sum(
            #     dim=1).sum(dim=1).sum(dim=0)
            denominator_tmp = pre1[:, i, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=0) + class_gt[:, i, :, :].sum(
                dim=1).sum(dim=1).sum(dim=0)
            numerator += w_tmp * numerator_tmp
            denominator += w_tmp * denominator_tmp
        # w1 = 1 / ((class_gt[:, 1, :, :].sum(dim=1).sum(dim=1).sum(dim=0).item())**2 + epsilon)
        # numerator1 = (pre1[:, 1, :, :] * class_gt[:, 1, :, :]).sum(dim=1).sum(dim=1).sum(dim=0)
        # denominator1 = pre1[:, 1, :, :].sum(dim=1).sum(dim=1).sum(dim=0) + class_gt[:, 1, :, :].sum(
        #     dim=1).sum(dim=1).sum(dim=0)
        #
        # w0 = 1 / ((class_gt[:, 0, :, :].sum(dim=1).sum(dim=1).sum(dim=0).item())**2 + epsilon)
        # numerator0 = (pre1[:, 0, :, :] * class_gt[:, 0, :, :]).sum(dim=1).sum(dim=1).sum(dim=0)
        # denominator0 = pre1[:, 0, :, :].sum(dim=1).sum(dim=1).sum(dim=0) + class_gt[:, 0, :, :].sum(
        #     dim=1).sum(dim=1).sum(dim=0)

        # dice_stage1 = 0.0
        # for class_index in range(1, self.num_class):
        #     dice_stage1_numerator = self.class_weight[class_index] * 2 * (
        #                 pre1[:, class_index, :, :] * class_gt[:, class_index, :, :]).sum(dim=1).sum(dim=1).sum(dim=0)
        #     dice_stage1_denominator = pre1[:, class_index, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=0) + class_gt[:,
        #                                                                                                    class_index,
        #                                                                                                    :, :].pow(
        #         2).sum(dim=1).sum(dim=1).sum(dim=0)
        #
        #     dice_stage1 += dice_stage1_numerator / (dice_stage1_denominator + 1e-5)
        #
        # dice_stage1 = dice_stage1 / (self.num_class - 1)
        # dice = 1 - dice_stage1
        # dice_score = 2 * (w0 * numerator0 + w1 * numerator1) / (w0 * denominator0 + w1 * denominator1)
        dice_score = 2 * numerator / (denominator + epsilon)
        dice_loss = 1 - dice_score

        # pre_test0 = pre1[0, 0, :, :]
        # pre_test0 = np.uint8(pre_test0.cpu().detach().numpy() * 250)
        # cv.imshow('pre_test0', pre_test0)
        # cv.waitKey()
        #
        # pre_test1 = pre1[0, 1, :, :]
        # pre_test1 = np.uint8(pre_test1.cpu().detach().numpy() * 250)
        # cv.imshow('pre_test1', pre_test1)
        # cv.waitKey()

        # pre_test2 = pre1[0, 2, :, :]
        # pre_test2 = np.uint8(pre_test2.cpu().detach().numpy() * 250)
        # cv.imshow('pre_test2', pre_test2)
        # cv.waitKey()

        # pre_test3 = pre1[0, 3, :, :]
        # pre_test3 = pre_test3.cpu().detach().numpy() * 250
        # cv.imshow('pre_test3', pre_test3)
        # cv.waitKey()

        # gt_test0 = class_gt[0, 0, :, :]
        # gt_test0 = gt_test0.cpu().detach().numpy() * 250
        # cv.imshow('gt_test0', gt_test0)
        # cv.waitKey()
        #
        # gt_test1 = class_gt[0, 1, :, :]
        # gt_test1 = gt_test1.cpu().detach().numpy() * 250
        # cv.imshow('gt_test1', gt_test1)
        # cv.waitKey()

        # gt_test2 = class_gt[0, 2, :, :]
        # gt_test2 = gt_test2.cpu().detach().numpy() * 250
        # cv.imshow('gt_test2', gt_test2)
        # cv.waitKey()

        # gt_test3 = class_gt[0, 3, :, :]
        # gt_test3 = gt_test3.cpu().detach().numpy() * 250
        # cv.imshow('gt_test3', gt_test3)
        # cv.waitKey()

        return dice_loss
'''

