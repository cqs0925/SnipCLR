## SnipCLR
Code for the paper "Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localisation" (ICPR 2026).


### Requirements

Install essential packages in requirements.txt to set up python environment. Also refer to the implementation [here](https://github.com/line/Skeleton-Temporal-Action-Localization).
We use numpy == 1.23.1

### Skeleton-based Action Localization

We provide an implementation of directly applying any skeleton-based action recognition backbones, such as ST-GCN, to perform action localization, using a simple/U-shape upsampling module to predict frame-level class label. The data preprocessing, BABEL subset split, model training, and postprocessing script for action localizaiton are based on the work of [AAAI 23'](https://github.com/line/Skeleton-Temporal-Action-Localization)):

- We follow the root-shoulder-spine align (NTU normalization) for finetuning on customized datasets.
- We remove their design for weak supervision, to  achieve the best performance under full supervision.
- Action segmentation/localization are formulated slightly different in the literature, however, not fundamentally different in skeleton-based studies, please also refer to PAMI version of [USDRL](https://github.com/wengwanjiang/USDRL).

Please update the file paths in yaml configs for the train/val data, or checkpoint if needed.


To train and evaluate the model on subsets of BABEL, run following commands:

```
python train.py --config ./configs/train_BABEL_Unet.yaml
```


### Snippet-level Contrastive Learning on Skeleton Sequences

Our SnipCLR loss can be seamlessly integrated into existing video-level skeleton-based self-supervised learning methods, such as [CrosSCLR](https://github.com/LinguoLi/CrosSCLR), [AimCLR](https://github.com/Levigty/AimCLR), [RVTCLR](https://github.com/Zhuysheng/RVTCLR) and so on.

We provide an example implementation on RVTCLR, it should be straightforward to add the loss term to other baselines. 

To pretrain on of BABEL, run following commands:

```
python main.py pretrain_skeletonclr_dense --config ./config/pretext_skeletonclr_babel_dense.yaml
```

To have the same performance, please use the entire training set on babel and evaluate (linear-probe/KNN-probe) on 3 validation subsets.


