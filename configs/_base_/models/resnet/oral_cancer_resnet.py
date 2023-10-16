# ResNet50配置
model = dict(
    type='RamanClassifier',  # 分类器类型
    backbone=dict(
        type='ResNet50',
        # type='ResNet',
        # type='Retinanet',
        # type='SEResNeXt50',
        # depth=34,
        # strides=(1, 2, 2, 2),
    ),
    neck=dict(type='GlobalAveragePooling'),
    # neck=None,
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=3,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
