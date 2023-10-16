# dataset settings
dataset_type = 'RamanSpectral'  # 数据集名称

# 训练数据流水线
train_pipeline = [
    dict(type='LoadDataFromFile', id='ID', labels=['labels1', 'labels2', 'labels3']),
    # dict(type='AddNoise', noise_std=0.01),
    # dict(type='MoveRaman', max_shift=3),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[-29, 4090]], method='drPLS', lam=10 ** 5, p=0.05),
    # dict(type='Normalize', method='intensity'),  # 归一化
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data 转为 torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels 转为 torch.Tensor
    # dict(type='Collect', keys=['data', 'labels1', 'labels2'])  # 决定数据中哪些键应该传递给检测器的流程
]
# 测试数据流水线
test_pipeline = [
    dict(type='LoadDataFromFile', id='ID', labels=['labels1', 'labels2', 'labels3']),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[-29, 4090]], method='drPLS', lam=10 ** 5, p=0.05),
    # dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=64,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 的 线程数
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/oral_cancer/health_bod_tnm/health_30_mean.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/oral_cancer/health_bod_tnm/health_30_mean.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/oral_cancer/health_bod_tnm/health_30_mean.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)

evaluation = dict(  # 计算准确率
    interval=1,
    metric=['precision', 'f1_score'],
    metric_options={'topk': (1,)},
    save_best="auto",
    start=1
)
