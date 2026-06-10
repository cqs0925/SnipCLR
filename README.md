## SnipCLR
Code for the paper "Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localisation" (ICPR 2026).
### Skeleton-based Action Localization

We provide an implementation of directly applying any skeleton-based action recognition backbones, such as ST-GCN, to perform action localization, using a simple/U-shape upsampling module to predict frame-level class label. The dataset preprocessing, model training, and postprocessing script for action localizaiton are based on the work of [AAAI 23'](https://github.com/line/Skeleton-Temporal-Action-Localization)):

- We follow the root-shoulder-spine align (NTU normalization) for finetuning on customized datasets.
- We remove their design for weak supervision, to  achieve the best performance under full supervision.
- Action segmentation/localization are formulated slightly different in the literature, however, not fundamentally different in skeleton-based studies, please also refer to PAMI version of [USDRL](https://github.com/wengwanjiang/USDRL).

Please follow the file paths in yaml configs to prepare the train/val data, or checkpoint if needed.

### Snippet-level Contrastive Learning on Skeleton Sequences

Our SnipCLR loss can be seamlessly integrated into existing video-level skeleton-based self-supervised learning methods, such as [CrosSCLR](https://github.com/LinguoLi/CrosSCLR), [AimCLR](https://github.com/Levigty/AimCLR), [RVTCLR](https://github.com/Zhuysheng/RVTCLR) and so on.

We provide an example implementation on RVTCLR under the SnipCLR folder, it should straight forward to add the loss term to other baseline. 
