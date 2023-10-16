# ResNetV2
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='GoogLeNet', input_dim=900, num_classes=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=3,
        in_channels=480,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    )
)
