# RIFR
  This project provides an implementation for "Balancing Attention to Base and Novel Categories for Few-Shot Object Detection in Remote Sensing Imagery" on PyTorch. 

## Requirements
  + Python 3.7+
  + PyTorch 1.5+
  + mmcv 1.3.12+
  + mmdet 2.16.0+
  + mmcls 0.15.0+

## Get Started
  #### step1: base training
    bash ./tools/detection/dist_train.sh \
        configs/detection/rifr/dior/split1/rifr_r101_fpn_dior-split1_base-training.py 2
  #### step2: reshape the bbox head of base model
    python -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/rifr_r101_fpn_dior-split1_base-training/latest.pth \
        --method randinit \
        --save-dir work_dirs/rifr_r101_fpn_dior-split1_base-training
  #### step3: few shot fine-tuning
    bash ./tools/detection/dist_train.sh \
        configs/detection/rifr/dior/split1/rifr_r101_fpn_pt-loss_dior-split1_5shot-fine-tuning.py 2

## Acknowledgement
  This repo is based on the open-source [mmfewshot](https://github.com/open-mmlab/mmfewshot) project. We appreciate all the contributors who participated in the project.
