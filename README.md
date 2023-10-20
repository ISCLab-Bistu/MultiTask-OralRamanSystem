# MultiTask-OralRamanSystem
## Introduction

The Multitask-Oracle ramansystem includes four multi-task oral cancer intelligent diagnosis models. 
The study has been published in Elsevier's Journal of Molecular Structure.

This study aims to improve four classical neural network models, including AlexNet, GoogleNet, ResNet50 and Transformer, to realize multi-task classification to accurately diagnose TNM staging and pathological classification of tumor patients.

The model can extract the shared Raman spectral features, and realize the multi-task diagnosis of T stage, N stage and pathological classification at the same time, so as to provide accurate diagnostic information for individualized treatment of cancer patients.

## Models
In our model, we modify the model into three components :Backbone, Neck, and Head. Backbone performs feature extraction, Neck performs global pooling, and Head implements multi-task classification.

The backbone, neck, and head components are implemented separately in our classifier module, and the four classical network model diagrams are shown below.

MTN-AlexNet network model:
![img.png](imgs/model/alexnet.png)

MTN-GoogleNet network model:
![img.png](imgs/model/googlenet.png)

MTN-ResNet50 network model:
![img.png](imgs/model/resnet50.png)

MTN-Transformer network model:

![img.png](imgs/model/transformer.png)

### Backbone
Backbone includes AlexNet,GoogleNet, ResNet50 and Transformer, and users load the corresponding models through configuration files

<details open>
<summary>Supported backbone network</summary>

- [x] [AlexNet](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/backbones/alexnet.py)
- [x] [GoogleNet](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/backbones/googlenet.py)
- [x] [ResNet50](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/backbones/resnet50.py)
- [x] [Transformer](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/backbones/transformer.py)

</details>

### Neck
Neck is mainly Global Average Pooling.

<details open>
<summary>Neck network of support</summary>

- [x] [Global Average Pooling](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/necks/gap.py)

</details>

### Head
MultiTaskLinearClsHead is mainly used to implement multitask classification in Head. Meanwhile, the corresponding Loss is integrated in the Head.

<details open>
<summary>Supported classification headers</summary>

- [x] [ClsHead](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/heads/cls_head.py)
- [x] [MultiTaskLinearClsHead](https://github.com/ISCLab-Bistu/MultiTask-OralRamanSystem/blob/master/rmsm/models/heads/multi_task_linear_head.py)

</details>

## Result
The corresponding results of the modified multi-output network model based on MTN-AlexNet,MTN-GoogleNet,MTN-ResNet50 and MTN-Transformer are shown in the following figure:

Table 1: OA, OP, OS and OF of 7-fold cross-validation and standard deviation of four multi-task network models on classification tasks involving tumor staging, lymph node staging and histologic grading.
![img.png](imgs/table1.png)

Table 2: OA, OP, OS and OF of MTN-Transformer, AlexNet, ResNet50, GoogleNet, Transformer and machine learning methods on classification tasks including tumor staging, lymph node staging and histologic grading
![img.png](imgs/table2.png)
