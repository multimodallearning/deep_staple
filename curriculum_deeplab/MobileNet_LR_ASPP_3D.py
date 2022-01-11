"""
Implements a Lite R-ASPP_3d Network for semantic segmentation from
`"Searching for MobileNetV3"
<https://arxiv.org/abs/1905.02244>`_.
Args:
    low_channels (int): the number of channels of the low level features.
    high_channels (int): the number of channels of the high level features.
    num_classes (int): number of output classes of the model (including the background).
    inter_channels (int, optional): the number of channels for intermediate computations.
ctx:
    ctx should be a dictionary like this: {'low': low_level_feature_map, 'high': high_level_feature_map}
    with feature maps coming from a backbone network used to compute the features for the model
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LRASPPHead_3d(nn.Module):
    # Credits to Hellena Hempe
    def __init__(
            self,
            low_channels: int,
            inter_channels: int,
            high_channels: int,
            num_classes: int
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv3d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv3d(inter_channels, num_classes, 1)

    def forward(self, ctx):
        low = ctx["low"]
        high = ctx["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-3:], mode='trilinear', align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)



#Atrous Spatial Pyramid Pooling (Segmentation Network)
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)



class ASPPPooling_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling_3d, self).__init__(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-3:]
        x = F.adaptive_avg_pool3d(x,(1))
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='nearest')#, align_corners=False)



class ASPP_3d(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP_3d, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling_3d(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, ctx):
        return self.module(ctx) + ctx



class Backbone_3d(torch.nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, mid_stride):
        super().__init__()

        self.backbone = []
        self.backbone.append(nn.Identity())

        for i in range(len(in_channels)):
            inc = int(in_channels[i])
            midc = int(mid_channels[i])
            outc = int(out_channels[i])
            strd = int(mid_stride[i])

            layer = nn.Sequential(
                nn.Conv3d(inc,midc,1,bias=False),
                nn.BatchNorm3d(midc),nn.ReLU6(True),
                nn.Conv3d(midc,midc,3,stride=strd, padding=1, bias=False, groups=midc),
                nn.BatchNorm3d(midc),nn.ReLU6(True),
                nn.Conv3d(midc,outc,1,bias=False),nn.BatchNorm3d(outc)
            )

            if i == 0:
                layer[0] = nn.Conv3d(inc, midc, 3, padding=1, stride=2, bias=False)
            if (inc==outc)&(strd==1):
                self.backbone.append(ResBlock(layer))
            else:
                self.backbone.append(layer)

        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, ctx):
        return self.backbone(ctx)

    def __getitem__(self, idx):
        return self.backbone[idx]



#Mobile-Net 3D with depth-separable convolutions and residual connections
class MobileNet_ASPP_3D(torch.nn.Module):
    def __init__(self, in_num, num_classes, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.in_channels = torch.Tensor([in_num,16,24,24,32,32,32,64]).long()
        self.mid_channels = torch.Tensor([32,96,144,144,192,192,192,384]).long()
        self.out_channels = torch.Tensor([16,24,24,32,32,32,64,64]).long()
        self.mid_stride = torch.Tensor([1,1,1,1,1,2,1,1])

        # Complete model: MobileNet_3d + ASPP_3d + head_3d (with a single skip connection)
        self.backbone = Backbone_3d(self.in_channels,
            self.mid_channels,
            self.out_channels,
            self.mid_stride
        )

        self.aspp = ASPP_3d(64,(2,4,8,16),128)#ASPP_3d(64,(1,),128)#

        self.head = nn.Sequential(
            nn.Conv3d(128+16, 64, 1, padding=0,groups=1, bias=False),
            nn.BatchNorm3d(64),nn.ReLU(),
            nn.Conv3d(64, 64, 3, groups=1,padding=1, bias=False),
            nn.BatchNorm3d(64),nn.ReLU(),
            nn.Conv3d(64, num_classes, 1)
        )
        # Do I need to call this?
        self.apply()

    def forward(self, ctx):
        if self.use_checkpointing:
            high = checkpoint(self.backbone[:2], ctx)
            low = checkpoint(self.backbone[2:], high)
            low = checkpoint(self.aspp, low)
            # Skip-connection
            # y = torch.cat((x1, F.interpolate(low, scale_factor=2)), 1)
            y1 = checkpoint(self.head, {"low": low, "high": high})

        else:
            x1 = self.backbone[:2](ctx)
            x2 = self.backbone[2:](x1)
            y = self.aspp(x2)
            # Skip-connection
            y = torch.cat((x1, F.interpolate(y, scale_factor=2)), 1)
            y1 = self.head(y)

        out = F.interpolate(y1, scale_factor=2, mode='trilinear', align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result

    def apply(self, init_func=None):
        if init_func:
            super().apply(init_func)

        else:
            for b_mod in self.backbone.modules():
                # Weight initialization
                if isinstance(b_mod, nn.Conv3d):
                    nn.init.kaiming_normal_(b_mod.weight, mode='fan_out')
                    if b_mod.bias is not None:
                        nn.init.zeros_(b_mod.bias)

                elif isinstance(b_mod, (nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.ones_(b_mod.weight)
                    nn.init.zeros_(b_mod.bias)

                elif isinstance(b_mod, nn.Linear):
                    nn.init.normal_(b_mod.weight, 0, 0.01)
                    nn.init.zeros_(b_mod.bias)



class MobileNet_LRASPP_3D(MobileNet_ASPP_3D):
    def __init__(self, in_num, num_classes, use_checkpointing=True):
        super().__init__(in_num, num_classes, use_checkpointing)

        self.head = LRASPPHead_3d(
            high_channels=16,
            inter_channels=128,
            low_channels=128,
            num_classes=num_classes,
        )

    def forward(self, ctx):
        return super().forward(ctx)
