import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet50, resnet101
BN = nn.BatchNorm2d

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            BN(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(ASPP, self).__init__()
        self.c0 = _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        self.c1 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=pyramids[0], dilation=pyramids[0])
        self.c2 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=pyramids[1], dilation=pyramids[1])
        self.c3 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=pyramids[2], dilation=pyramids[2])
        self.imagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1),
        )

    def forward(self, x):
        _,_,h,w = x.size()
        out_imagepool = self.imagepool(x)
        out_imagepool = F.upsample(out_imagepool, size=(h,w), mode='bilinear')

        out_c0 = self.c0(x)
        out_c1 = self.c1(x)
        out_c2 = self.c2(x)
        out_c3 = self.c3(x)
        out = torch.cat([out_c0, out_c1, out_c2, out_c3, out_imagepool], dim=1)

        return out

class DeepLabv3Plus(nn.Module):

    def __init__(self, channel, num_classes, backbone='res50', pretrained=True):
        super(DeepLabv3Plus, self).__init__()

        if backbone == 'res50':
            self.backbone = resnet50(pretrained=pretrained)
        elif backbone == 'res101':
            self.backbone = resnet101(pretrained=pretrained)
        pyramids = [4,6,12]
        self.aspp = ASPP(2048, 256, pyramids)

        self.f1 = _ConvBatchNormReLU(256*(len(pyramids)+2), 256, 1, 1, 0, 1)
        self.literal = _ConvBatchNormReLU(256, 48, 1, 1, 0, 1) # reduce c1

        self.f2 = nn.Sequential(
            _ConvBatchNormReLU(48+256, 256, 3, 1, 1, 1),
            _ConvBatchNormReLU(256, 256, 3, 1, 1, 1)
        )

        self.out_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        o = self.backbone.conv1(x)
        o = self.backbone.bn1(o)
        o = self.backbone.relu(o)
        o = self.backbone.maxpool(o) # [#bs, 64, 60, 60]

        c1 = self.backbone.layer1(o)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3) # [#bs, 2048, 30, 30]

        out_aspp = self.aspp(c4)
        out_f1 = self.f1(out_aspp)

        out_reduce = self.literal(c1)
        out_f1 = F.upsample(out_f1, size=c1.shape[2:], mode='bilinear')

        out = torch.cat((out_f1, out_reduce), dim=1)
        out = self.f2(out)
        out = self.out_conv(out)
        out = F.upsample(out, size=x.shape[2:], mode='bilinear')

        return out

