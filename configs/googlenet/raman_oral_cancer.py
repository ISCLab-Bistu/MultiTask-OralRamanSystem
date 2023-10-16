_base_ = [
    '../_base_/models/googlenet/oral_cancer_googlenet.py', '../_base_/datasets/raman_oral_cancer.py',
    '../_base_/schedules/raman_google.py', '../_base_/default_runtime.py'
]

work_dir = 'oral_cancer_googlenet'
