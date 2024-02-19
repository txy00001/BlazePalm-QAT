from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models

class SEModule(nn.Module):##注意力机制
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#####自适应平均池化
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return inputs * outputs


class DAN(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(DAN, self).__init__()

        resnet = models.resnet18(pretrained)

        # if pretrained:
        #     checkpoint = torch.load('./models/resnet18-f37072fd.pth')
        #     resnet.load_state_dict(checkpoint, strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)
        print(heads.sum(dim=1).shape)
        out1 = self.fc(heads.sum(dim=1))
        out1 = self.bn(out1)

        return out1


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out


class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y

        return out



class MeshBlock(nn.Module):#匹配输出输入通道
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(MeshBlock, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = quant_nn.QuantMaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.conv0 = quant_nn.QuantConv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=in_channels, bias=True)
        self.conv1 = quant_nn.QuantConv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0, bias=True)

        self.act = nn.ReLU(out_channels)


    def forward(self, x):
        x = self.quant(x)
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        x = self.conv0(h)
        x = self.conv1(h)
        x = self.dequant(x)
        return self.act(self.convs(h) + x)

