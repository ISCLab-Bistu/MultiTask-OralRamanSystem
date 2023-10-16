_base_ = [
    '../_base_/models/transformer/oral_cancer_transformer.py', '../_base_/datasets/raman_oral_cancer.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'oral_cancer_transformer'
