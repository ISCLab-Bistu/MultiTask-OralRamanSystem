import torch
import torch.nn as nn
import torchvision
from ..builder import BACKBONES
from .SpatialAttention import SpatialAttention


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(places * self.expansion)
            )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.ReLU(out)
        return out


@BACKBONES.register_module()
class ResNet50(nn.Module):
    def __init__(self, blocks=[3, 4, 6, 3], num_classes=6, expansion=4):
        super(ResNet50, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64, stride=1)

        self.layer1 = self.make_res_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_res_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_res_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_res_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # 注意力机制
        self.spatial_attention = SpatialAttention(900)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='ReLU')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_res_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 对于空间注意力机制，考虑加到卷积特征提取之前
        x = self.spatial_attention(x)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# model = ResNet50()
# print(model)
