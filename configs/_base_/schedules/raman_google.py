# 优化器配置
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
runner = dict(type='EpochMultiRunner',  # 将使用的 runner 的类别，如 IterBasedRunner 或 EpochBasedRunner。
              max_epochs=400)  # runner 总回合数， 对于 IterBasedRunner 使用 `max_iters`
