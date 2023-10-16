model = dict(
    type='RamanClassifier',  # 分类器类型
    backbone=dict(
        type='AlexNet',
        # input_size=1038,
    ),
    neck=dict(type='GlobalAveragePooling', pool_length=6),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=3,
        in_channels=256 * 6,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    )
)
