# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class MultiTaskLinearClsHead(ClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 labels_f=[2, 2, 2],
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(MultiTaskLinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        # 对于三个任务，需要建立三个不同的Loss函数
        # self.loss1 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 2.38, 1.43])
        # self.loss2 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 1.0, 1.0])
        # self.loss3 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 1.37, 1.14])

        self.loss1 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 4.0, 1.2])
        self.loss2 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 2, 1.0])
        self.loss3 = dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 2.5, 1.07])

        # Attention layer weights
        # self.attention_weights1 = nn.Parameter(torch.ones(3))
        # self.attention_weights2 = nn.Parameter(torch.ones(3))
        # self.attention_weights3 = nn.Parameter(torch.ones(3))

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # 任务一（对癌细胞分化程度进行分类）

        self.fc1 = self.build_linear(labels_f[0])
        # 任务二（对TNM进行分类）
        self.fc2 = self.build_linear(labels_f[1])
        # 任务三考虑使用LSTM来进行分类
        self.fc3 = self.build_linear(labels_f[2])

    def build_linear(self, index):
        if index == 2:
            linear_all = nn.Sequential(
                nn.Linear(self.in_channels, 640),
                nn.BatchNorm1d(640),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(640, 3),
            )
        elif index == 3:
            linear_all = nn.Sequential(
                nn.Linear(self.in_channels, 320),
                nn.BatchNorm1d(320),
                nn.ELU(),
                nn.Linear(320, 160),
                nn.BatchNorm1d(160),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(160, 3),
            )
        elif index == 4:
            linear_all = nn.Sequential(
                nn.Linear(self.in_channels, 640),
                nn.BatchNorm1d(640),
                nn.ELU(),
                nn.Linear(640, 320),
                nn.BatchNorm1d(320),
                nn.ELU(),
                nn.Linear(320, 160),
                nn.BatchNorm1d(160),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(160, 3),
            )
        elif index == 1:
            linear_all = nn.Sequential(
                nn.Linear(self.in_channels, 3),
            )

        return linear_all

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=False):
        x = self.pre_logits(x)
        cls_score1 = self.fc1(x)
        cls_score2 = self.fc2(x)
        cls_score3 = self.fc3(x)

        # Apply attention weights
        # attention_probs1 = torch.softmax(self.attention_weights1, dim=0)  # attention the 2
        # attention_probs2 = torch.softmax(self.attention_weights2, dim=0)  # attention the 2
        # attention_probs3 = torch.softmax(self.attention_weights3, dim=0)  # attention the 2
        #
        # cls_score1 = cls_score1 * attention_probs1
        # cls_score2 = cls_score2 * attention_probs2
        # cls_score3 = cls_score3 * attention_probs3

        if softmax:
            pred1 = (F.softmax(cls_score1, dim=1) if cls_score1 is not None else None)
            pred2 = (F.softmax(cls_score2, dim=1) if cls_score2 is not None else None)
            pred3 = (F.softmax(cls_score3, dim=1) if cls_score2 is not None else None)
        else:
            pred1 = cls_score1
            pred2 = cls_score2
            pred3 = cls_score3
        pred = torch.cat((pred1, pred2, pred3), dim=1)
        # pred = torch.cat((pred1, pred2), dim=1)
        pred = list(pred.detach().cpu())
        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, labels, **kwargs):
        x = self.pre_logits(x)
        labels1 = labels[:, 0]
        labels2 = labels[:, 1]
        labels3 = labels[:, 2]

        cls_score1 = self.fc1(x)

        cls_score2 = self.fc2(x)
        cls_score3 = self.fc3(x)

        # Apply attention weights
        # attention_probs1 = torch.softmax(self.attention_weights1, dim=0)  # attention the 2
        # attention_probs2 = torch.softmax(self.attention_weights2, dim=0)  # attention the 2
        # attention_probs3 = torch.softmax(self.attention_weights3, dim=0)  # attention the 2
        #
        # cls_score1 = cls_score1 * attention_probs1
        # cls_score2 = cls_score2 * attention_probs2
        # cls_score3 = cls_score3 * attention_probs3

        losses1 = self.loss(cls_score1, labels1.long(), self.loss1, **kwargs)
        losses2 = self.loss(cls_score2, labels2.long(), self.loss2, **kwargs)
        losses3 = self.loss(cls_score3, labels3.long(), self.loss3, **kwargs)

        return losses1, losses2, losses3
