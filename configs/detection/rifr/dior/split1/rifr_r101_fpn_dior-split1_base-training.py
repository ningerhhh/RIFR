_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))

#base
evaluation = dict(
    interval=18000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=200, step=[24000, 32000])
runner = dict(max_iters=36000)


# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=15)))
# using regular sampler can get a better base model
use_infinite_sampler = False


work_dir = './work_dirs/rifr_r101_fpn_dior-split1_base-training/'
