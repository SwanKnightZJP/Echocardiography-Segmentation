import torch.nn as nn
from lib.utils import net_utils
from lib.loss.Loss_function import DiceLoss, CELoss, GDiceLoss, Dice_CE_Loss, DiceBG_CE_Loss, GDiceBG_CE_Loss
import gc


_loss_factory = {
    'GDice': GDiceLoss,
    'Dice': DiceLoss,
    'CE': CELoss,
    'Dice_CE': Dice_CE_Loss,
    'DiceBG_CE': DiceBG_CE_Loss,
    'GDiceBG_CE': GDiceBG_CE_Loss,
}


class NetworkWrapper(nn.Module):
    def __init__(self, cfg, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.loss = _loss_factory[cfg.loss](cfg)

    # def forward(self, inputs, mask, target):
    def forward(self, batch):
        pred = self.net(batch['input'])  #
        loss = self.loss(pred, batch)

        # cv2.imshow('sagittal', patch_label_array[74, :, :])
        # cv2.imshow('coronal', patch_label_array[:, 98, :])
        # cv2.imshow('axial', patch_label_array[:, :, 104])
        # cv2.waitKey()
        del batch['input']
        gc.collect()

        scalar_stats = {}
        scalar_stats.update({'loss': loss})
        image_stats = {}
        # scalar_stats.update({'stat': img_stat})

        return pred, loss, scalar_stats, image_stats
