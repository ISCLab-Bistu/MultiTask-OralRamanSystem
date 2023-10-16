# ResNetV2配置
model = dict(
    type='RamanClassifier',
    backbone=dict(
        # type='MultiTransformer',
        type='Ml4fTransformer',
        # type='SwinTransformer',
        input_dim=900,
        # num_classes=3
    ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskLinearClsHead',
        labels_f=[2, 2, 2],
        num_classes=3,
        in_channels=900,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    )
)
