"""
    BaseLine Vnet:
        3D version

        dilateConv / defaultConv

    input_img:
        shape: 112 * 112 * 112
    input_tensor:
        shape: B 1 112 112 112

    torchsummary.test
        2 1 112 112 112 -- at single card

        ================================================================
        Total params: 45,600,316
        Trainable params: 45,600,316
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 5.36
        Forward/backward pass size (MB): 5112.17
        Params size (MB): 173.95
        Estimated Total Size (MB): 5291.48
        ----------------------------------------------------------------

    linked at net/task/__init__.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
import numpy as np


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)  ###############

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(8)
        self.relu1 = ELUCons(elu, 8)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))   # x from c=1 to c=16
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x), 1)  # repeat 16 times
        out = self.relu1(torch.add(out, x16))         # then add channel wised X and 16times x
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    """
    input:  [B 32 112 112 112]
     -- conv1(32,2,k=5,p=2) -- [B 2 112 112 112]
     -- BN1 -- [ 2 ]
     -- conv2(2,2,k=1) -- [B 2 112 112 112]
     -- ELU -- [2]
     -- permute(0, 2, 3, 4, 1).contiguous() -- [B 112 112 112 2]
     -- view     -- [B*W*H*Z 2]
     -- softmax()
    """
    def __init__(self, inChans, elu, nll, out_channel):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, out_channel, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1)
        self.relu1 = ELUCons(elu, out_channel)
        self.out_channel = out_channel
        if nll:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # convolve 32 down to n channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()  # from [B C W H Z] -- [B W H Z C]
        # flatten
        out = out.view(out.numel() // self.out_channel, self.out_channel)  # dimation change from [B C W H Z] -- [ B*W*H*Z, C ]
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class OutputTransitionMap(nn.Module):
    """
    input:  [B 32 112 112 112]
     -- conv1(32,n,k=5,p=2) -- [B n 112 112 112]
     -- BN1 -- [ n ]
     -- conv2(n,n,k=1) -- [B n 112 112 112]
     -- ELU -- [ n ]
     -- softmax(dim=1)
    """
    def __init__(self, inChans, elu, nll, out_channel):
        super(OutputTransitionMap, self).__init__()
        self.conv1 = nn.Conv3d(inChans, out_channel, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1)
        self.relu1 = ELUCons(elu, out_channel)
        if nll:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # convolve 32 down to 3 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False, out_channel=2):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(8, elu)  # size = input_X
        self.down_tr16 = DownTransition(8, 1, elu)  # 1 layer in 16 out 32 size=input_X/2
        self.down_tr32 = DownTransition(16, 2, elu)  # 2 layers in 32 out 64 size=input_X/4
        self.down_tr64 = DownTransition(32, 3, elu, dropout=True)  # 3 layers in 64 out 128 size=input_X/8
        # -------------- warning !!! ---------------------- #
        self.down_tr128 = DownTransition(64, 2, elu, dropout=True)  # 2 layers in 128 out 256 size=input_X/16
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)  # 2 layers in 256 out 256 size=input_X/2
        self.up_tr64 = UpTransition(128, 64, 2, elu, dropout=True)
        # self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        # -------------- warning !!! ---------------------- #
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.up_tr16 = UpTransition(32, 16, 1, elu)
        self.out_tr = OutputTransition(16, elu, nll, out_channel)  # out channel = ? num_channel
        self.out_map = OutputTransitionMap(16, elu, nll, out_channel)  # out channel = ?  num_channel

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out8 = self.in_tr(x)
        out16 = self.down_tr16(out8)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        # torch.save(out8, 'out8E.pt')
        # torch.save(out16, 'out16E.pt')
        # torch.save(out32, 'out32E.pt')
        # torch.save(out64, 'out64E.pt')
        # torch.save(out128, 'out128E.pt')
        out = self.up_tr128(out128, out64)
        # torch.save(out, 'out128D.pt')
        out = self.up_tr64(out, out32)
        # torch.save(out64, 'out64D.pt')
        out = self.up_tr32(out, out16)
        # torch.save(out32, 'out32D.pt')
        out = self.up_tr16(out, out8)  # [2,32,112,112,112] (B,C,W H Z)
        # torch.save(out16, 'out16D.pt')

        out02 = self.out_map(out)
        # torch.save(out02, 'outFin.pt')
        # out01 = self.out_tr(out)   # [2, 32->n, 112, 112, 112] --> [2*112*112*112, 2]  warning !!!
         # [B, n, 112, 112, 112]   warning !!!

        output = {'map': out02}  # shape = {B, n, 112, 112, 112} -- Dice
        # outLine = {'compress': out01}  # shape = {n, BWHZ}} -- CrossEntropy

        # output.update(outLine)

        return output


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight, 0.25)
        nn.init.constant_(module.bias, 0)


def get_network(cfg, is_train):
    net = VNet(elu=cfg.elu, nll=cfg.nll, out_channel=cfg.num_class+1)
    net.apply(init)
    return net


# --- debug_test --- #
if __name__ == '__main__':

    test_net = VNet(elu=True, nll=False, out_channel=2+1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = test_net.to(device)

    summary(test_net, (1, 128, 128, 160), batch_size=1)  # channel/ depth / width / height

    Final = 'the end'
    print(Final)
