import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from pytorch_quantization import quant_modules
#from pytorch_quantization import nn as quant_nn
class GetKeysDict:
    def __init__(self):
        self.hand_palm_dict = [

            'backbone1.backbone1.0.conv.weight',
            'backbone1.backbone1.0.conv.bias',
            'backbone1.backbone1.1.weight',
            'backbone1.backbone1.2.convs.0.conv.weight',
            'backbone1.backbone1.2.convs.0.conv.bias',
            'backbone1.backbone1.2.convs.1.conv.weight',
            'backbone1.backbone1.2.convs.1.conv.bias',

            'backbone1.backbone1.3.weight',
            'backbone1.backbone1.4.convs.0.conv.weight',
            'backbone1.backbone1.4.convs.0.conv.bias',
            'backbone1.backbone1.4.convs.1.conv.weight',
            'backbone1.backbone1.4.convs.1.conv.bias',

            'backbone1.backbone1.5.weight',
            'backbone1.backbone1.6.convs.0.conv.weight',
            'backbone1.backbone1.6.convs.0.conv.bias',
            'backbone1.backbone1.6.convs.1.conv.weight',
            'backbone1.backbone1.6.convs.1.conv.bias',

            'backbone1.backbone1.7.weight',
            'backbone1.backbone1.8.convs.0.conv.weight',
            'backbone1.backbone1.8.convs.0.conv.bias',
            'backbone1.backbone1.8.convs.1.conv.weight',
            'backbone1.backbone1.8.convs.1.conv.bias',

            'backbone1.backbone1.9.weight',
            'backbone1.backbone1.10.convs.0.conv.weight',
            'backbone1.backbone1.10.convs.0.conv.bias',
            'backbone1.backbone1.10.convs.1.conv.weight',
            'backbone1.backbone1.10.convs.1.conv.bias',

            'backbone1.backbone1.11.weight',
            'backbone1.backbone1.12.convs.0.conv.weight',
            'backbone1.backbone1.12.convs.0.conv.bias',
            'backbone1.backbone1.12.convs.1.conv.weight',
            'backbone1.backbone1.12.convs.1.conv.bias',

            'backbone1.backbone1.13.weight',
            'backbone1.backbone1.14.convs.0.conv.weight',
            'backbone1.backbone1.14.convs.0.conv.bias',
            'backbone1.backbone1.14.convs.1.conv.weight',
            'backbone1.backbone1.14.convs.1.conv.bias',

            'backbone1.backbone1.15.weight',
            'backbone1.backbone1.16.convs.0.conv.weight',
            'backbone1.backbone1.16.convs.0.conv.bias',
            'backbone1.backbone1.16.convs.1.conv.weight',
            'backbone1.backbone1.16.convs.1.conv.bias',

            'backbone1.backbone1.17.weight',
            'backbone1.backbone1.18.convs.0.conv.weight',
            'backbone1.backbone1.18.convs.0.conv.bias',
            'backbone1.backbone1.18.convs.1.conv.weight',
            'backbone1.backbone1.18.convs.1.conv.bias',

            'backbone1.backbone1.19.weight',
            'backbone1.backbone1.20.convs.0.conv.weight',
            'backbone1.backbone1.20.convs.0.conv.bias',
            'backbone1.backbone1.20.convs.1.conv.weight',
            'backbone1.backbone1.20.convs.1.conv.bias',

            'backbone1.backbone1.21.weight',
            'backbone1.backbone1.22.convs.0.conv.weight',
            'backbone1.backbone1.22.convs.0.conv.bias',
            'backbone1.backbone1.22.convs.1.conv.weight',
            'backbone1.backbone1.22.convs.1.conv.bias',

            'backbone1.backbone1.23.weight',

            'backbone2.path1.0.convs.0.conv.weight',
            'backbone2.path1.0.convs.0.conv.bias',

            'backbone2.path1.0.convs.1.conv.weight',
            'backbone2.path1.0.convs.1.conv.bias',

            'backbone2.path1.1.weight',
            'backbone2.path1.2.convs.0.conv.weight',
            'backbone2.path1.2.convs.0.conv.bias',
            'backbone2.path1.2.convs.1.conv.weight',
            'backbone2.path1.2.convs.1.conv.bias',

            'backbone2.path1.3.weight',
            'backbone2.path1.4.convs.0.conv.weight',
            'backbone2.path1.4.convs.0.conv.bias',
            'backbone2.path1.4.convs.1.conv.weight',
            'backbone2.path1.4.convs.1.conv.bias',

            'backbone2.path1.5.weight',
            'backbone2.path1.6.convs.0.conv.weight',
            'backbone2.path1.6.convs.0.conv.bias',
            'backbone2.path1.6.convs.1.conv.weight',
            'backbone2.path1.6.convs.1.conv.bias',

            'backbone2.path1.7.weight',

            'backbone2.path2.0.convs.0.conv.weight',
            'backbone2.path2.0.convs.0.conv.bias',

            'backbone2.path2.0.convs.1.conv.weight',
            'backbone2.path2.0.convs.1.conv.bias',

            'backbone2.path2.1.weight',
            'backbone2.path2.2.convs.0.conv.weight',
            'backbone2.path2.2.convs.0.conv.bias',
            'backbone2.path2.2.convs.1.conv.weight',
            'backbone2.path2.2.convs.1.conv.bias',

            'backbone2.path2.3.weight',
            'backbone2.path2.4.convs.0.conv.weight',
            'backbone2.path2.4.convs.0.conv.bias',
            'backbone2.path2.4.convs.1.conv.weight',
            'backbone2.path2.4.convs.1.conv.bias',

            'backbone2.path2.5.weight',
            'backbone2.path2.6.convs.0.conv.weight',
            'backbone2.path2.6.convs.0.conv.bias',
            'backbone2.path2.6.convs.1.conv.weight',
            'backbone2.path2.6.convs.1.conv.bias',

            'backbone2.path2.7.weight',
            'backbone2.path2.9.conv.weight',
            'backbone2.path2.9.conv.bias',
            'backbone2.path2.10.weight',
            'backbone2.path3.0.convs.0.conv.weight',
            'backbone2.path3.0.convs.0.conv.bias',
            'backbone2.path3.0.convs.1.conv.weight',
            'backbone2.path3.0.convs.1.conv.bias',

            'backbone2.path3.1.weight',
            'backbone2.path3.2.convs.0.conv.weight',
            'backbone2.path3.2.convs.0.conv.bias',
            'backbone2.path3.2.convs.1.conv.weight',
            'backbone2.path3.2.convs.1.conv.bias',

            'backbone2.path3.3.weight',

            'backbone2.path4.1.conv.weight',

            'backbone2.path4.1.conv.bias',

            'backbone2.path4.2.weight',

            'backbone2.path5.0.convs.0.conv.weight',

            'backbone2.path5.0.convs.0.conv.bias',
            #
            'backbone2.path5.0.convs.1.conv.weight',
            #
            'backbone2.path5.0.convs.1.conv.bias',

            'backbone2.path5.1.weight',
            'backbone2.path5.2.convs.0.conv.weight',
            'backbone2.path5.2.convs.0.conv.bias',
            'backbone2.path5.2.convs.1.conv.weight',
            'backbone2.path5.2.convs.1.conv.bias',

            'backbone2.path5.3.weight',

            'classifier1.weight',
            'classifier1.bias',

            'classifier2.weight',
            'classifier2.bias',
            'regressor1.weight',
            'regressor1.bias',
            'regressor2.weight',
            'regressor2.bias'
        ]

# class ConvTF(nn.Conv2d):
#     def __init__(self, in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding='same',
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  quantize: bool = False
#                  ):
#         super(ConvTF, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#
#         if padding.lower() not in ('valid', 'same'):
#             raise ValueError("padding must be 'same' or 'valid'")
#         self.pad = padding
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#
#     def compute_valid_shape(self, in_shape):
#
#         in_shape = np.asarray(in_shape).astype('int32')
#         stride = np.asarray(self.stride).astype('int32')
#         kernel_size = np.asarray(self.kernel_size).astype('int32')
#         stride = np.concatenate([[1, 1], stride])
#         kernel_size = np.concatenate([[1, 1], kernel_size])
#         dilation = np.asarray(self.dilation).astype('int32')
#         dilation = np.concatenate([[1, 1], dilation])
#         if self.pad == 'same':
#             out_shape = (in_shape + stride - 1) // stride
#         else:
#             out_shape = (in_shape - dilation * (kernel_size - 1) - 1) // stride + 1
#         valid_input_shape = (out_shape - 1) * stride + 1 + dilation * (kernel_size - 1)
#
#         return valid_input_shape
#
#     def forward(self, input):
#
#         in_shape = np.asarray(input.shape).astype('int32')
#         valid_shape = self.compute_valid_shape(in_shape)
#         pad = []
#         for x in valid_shape - in_shape:
#             if x == 0:
#                 continue
#             pad_left = x // 2
#             pad_right = x - pad_left
#             # pad right should be larger than pad left
#             pad.extend((pad_left, pad_right))
#         if np.not_equal(pad, 0).any():
#             padded_input = F.pad(input, pad, "constant", 0)
#         else:
#             padded_input = input
#         return self.conv(padded_input)
#         # return super(ConvTF, self).forward(padded_input)


class ConvTF(nn.Module):
    """Tensorflow convolution has a different padding than PyTorch.
    When using padding='SAME' in tf, it will pad right bottom first, while
    pt pads left top first.
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ConvTF, self).__init__()
        if padding.lower() not in ('valid', 'same'):
            raise ValueError("padding must be 'same' or 'valid'")
        self.pad = padding
        self.stride=(stride,stride)
        self.kernel_size=(kernel_size,kernel_size)
        self.dilation= (dilation,dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def compute_valid_shape(self, in_shape):
        in_shape = np.asarray(in_shape).astype('int32')
        stride = np.asarray(self.stride).astype('int32')
        kernel_size = np.asarray(self.kernel_size).astype('int32')
        stride = np.concatenate([[1, 1], stride])
        kernel_size = np.concatenate([[1, 1], kernel_size])
        dilation = np.asarray(self.dilation).astype('int32')
        dilation = np.concatenate([[1, 1], dilation])
        if self.pad == 'same':
            out_shape = (in_shape + stride - 1) // stride
        else:
            out_shape = (in_shape - dilation * (kernel_size - 1) - 1) // stride + 1
        valid_input_shape = (out_shape - 1) * stride + 1 + dilation * (kernel_size - 1)
        return valid_input_shape

    def forward(self, input):
        in_shape = np.asarray(input.shape).astype('int32')
        valid_shape = self.compute_valid_shape(in_shape)
        pad = []
        for x in valid_shape - in_shape:
            if x == 0:
                continue
            pad_left = x // 2
            pad_right = x - pad_left
            # pad right should be larger than pad left
            pad.extend((pad_left, pad_right))
        if np.not_equal(pad, 0).any():
            padded_input = F.pad(input, pad, "constant", 0)
        else:
            padded_input = input
        result = self.conv(padded_input)
        return result
        # return super(ConvTF, self).forward(padded_input)

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        # self.quant = torch.quantization.QuantStub()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride > 1:
            self.max_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=[stride, stride], padding=[0, 0], dilation=1,
                                               ceil_mode=False)

        self.convs = nn.Sequential(
            ConvTF(in_channels=in_channels, out_channels=in_channels,
                   kernel_size=kernel_size, stride=stride, padding='same',
                   groups=in_channels, bias=True),
            ConvTF(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=1, stride=1, padding='valid', bias=True),
        )
        # self.dequant = torch.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()  ##因为int8数值进行加法运算容易超出数值范围，所以不是直接进行计算，而是进行反量化->计算->量化的操作
    def forward(self, x):
        # x=self.quant(x)
        if self.stride > 1:
            h = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            h = F.pad(h, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
            # h = torch.cat([h, h], dim=1)
        # x = self.dequant(x)
        return self.skip_add.add(self.convs(x),h)####


class BackBone1(nn.Module):
    def __init__(self):
        super(BackBone1, self).__init__()
        # self.quant = torch.quantization.QuantStub()
        self.backbone1 = nn.ModuleList([
            ConvTF(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.PReLU(32),  # [1,32,96,96]

            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.PReLU(64),  # [1,64,48,48]

            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.PReLU(128),  # [1,128,24,24]

            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),
            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),
            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),  # 该层需要输出 #[1,128,24,24]
        ])
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x=self.quant(x)
        y = x
        for fn in self.backbone1:
            y = fn(y)
        # y=self.dequant(y)
        return y


class BackBone2(nn.Module):
    def __init__(self):
        super(BackBone2, self).__init__()
        # self.quant = torch.quantization.QuantStub()
        self.path1 = nn.Sequential(
            BlazeBlock(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256))
        self.path2 = nn.Sequential(
            BlazeBlock(256, 256, 5, 2),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvTF(256, 256, 1),
            nn.PReLU(256))
        self.path3 = nn.Sequential(
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256))
        self.path4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvTF(256, 128, 1),
            nn.PReLU(128))
        self.path5 = nn.Sequential(
            BlazeBlock(128, 128, 5),
            nn.PReLU(128),
            BlazeBlock(128, 128, 5),
            nn.PReLU(128))
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        p1 = self.path1(x)
        p2 = self.path2(p1)
        p3 = self.path3(p1 + p2)
        p4 = self.path4(p3)
        p5 = self.path5(x + p4)
        # p3 = self.dequant(p3)
        # p5 = self.dequant(p5)
        return p3, p5


class BlazePalm_QAT(nn.Module):
    KEY_POINTS_NUMBER = 7
    NUM_PER_KEYPOINT = 2
    NUM_PER_BOX = 4

    def __init__(self):
        super(BlazePalm_QAT, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.backbone1 = BackBone1()
        self.backbone2 = BackBone2()
        self.classifier1 = nn.Conv2d(256, 6, 1)
        self.classifier2 = nn.Conv2d(128, 2, 1)
        self.regressor1 = nn.Conv2d(256, 108, 1)
        self.regressor2 = nn.Conv2d(128, 36, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, image):

        image = self.quant(image)
        b1 = self.backbone1(image)

        f1, f2 = self.backbone2(b1)

        c1 = self.classifier1(f1)
        c2 = self.classifier2(f2)
        r1 = self.regressor1(f1)
        r2 = self.regressor2(f2)

        regression_channels = self.NUM_PER_BOX + self.KEY_POINTS_NUMBER * self.NUM_PER_KEYPOINT
        c1 = c1.permute(0, 2, 3, 1).reshape(-1, c1.shape[1] * c1.shape[2] * c1.shape[3], 1)  # 864
        c2 = c2.permute(0, 2, 3, 1).reshape(-1, c2.shape[1] * c2.shape[2] * c2.shape[3], 1)
        r1 = r1.permute(0, 2, 3, 1).reshape(-1, r1.shape[2] * r1.shape[3] * 6, regression_channels)
        r2 = r2.permute(0, 2, 3, 1).reshape(-1, r2.shape[2] * r2.shape[3] * 2, regression_channels)

        c = torch.cat((c2, c1), dim=1)
        r = torch.cat((r2, r1), dim=1)
        c = self.dequant(c)
        r = self.dequant(r)

        return c, r


