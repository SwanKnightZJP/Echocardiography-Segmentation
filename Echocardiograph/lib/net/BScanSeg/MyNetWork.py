import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
import numpy as np


class MyNetWork(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, inputs):
        outputs = inputs
        return outputs


class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()

    def forward(self,):
        pre = 'model_stracture'
        return pre


def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, 0.25)
        nn.init.constant_(module.bias, 0)


def get_network():
    network = Net()
    network.apply(init)
    return network


# --- debug_test --- #
if __name__ == '__main__':

    net = Net(num_class=1, training=True, dropout_rate=0.5)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    bone_mask_1 = np.ones((1, 512, 512))
    summary(net, (1, 512, 512))

    # def init(module):
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
    #         nn.init.kaiming_normal_(module.weight, 0.25)
    #         nn.init.constant_(module.bias, 0)
    #
    # net.apply(init)

    Final = 'the end'
    print(Final)
