_base_ = ['../_base_/models/faster_rcnn_r50_caffe_fpn_ft.py']
model = dict(
    type='RIFR',
    frozen_parameters=[
            'backbone', 'neck',
        ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            scale=20)))