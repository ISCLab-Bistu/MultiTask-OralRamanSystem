# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

from .SpatialAttention import SpatialAttention
from .MultiHeadSelfAttention import MultiHeadSelfAttention


@BACKBONES.register_module()
class AlexNet(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 注意力机制
        # self.spatial_attention = SpatialAttention(900)
        self.spatial_attention = MultiHeadSelfAttention(900, 10)

    def forward(self, x):
        # 对于空间注意力机制，考虑加到卷积特征提取之前
        x = self.spatial_attention(x)
        x = self.feature_extraction(x)

        return (x,)
