## SnipCLR
Code for the paper "Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localization" (ICPR 2026).
### Skeleton-based Action Localization

We provide an implementation of directly applying any skeleton-based action recognition backbones, such as ST-GCN, to perform action localization, using a simple/U-shape upsampling module to predict frame-level class label. The dataset preprocessing, model training, and postprocessing script for action localizaiton are based on the work of [AAAI 23'](https://github.com/line/Skeleton-Temporal-Action-Localization)):

- We follow the root-shoulder-spine align (NTU normalization) for finetuning on customized datasets.
- We remove their design for weak supervision, to  achieve the best performance under full supervision.
- Action segmentation/localization are formulated slightly different in the literature, however, not fundamentally different in skeleton-based studies, please also refer to PAMI version of [USDRL](https://github.com/wengwanjiang/USDRL).

Please follow the file paths in yaml configs to prepare the train/val data, or checkpoint if needed.
