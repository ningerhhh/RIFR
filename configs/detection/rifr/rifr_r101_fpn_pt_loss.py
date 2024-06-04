_base_ = ['../_base_/models/faster_rcnn_r50_caffe_fpn_ft.py']
model = dict(
    type='RIFR',
    frozen_parameters=[
        'backbone', 'neck',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        type='PTRoIHead',
        bbox_head=dict(
            type='PTBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_contrast=dict(
                type='SupervisedPTLoss',
                temperature=0.05,
                loss_weight=0.5,
                reweight_type='none'),
            scale=20,
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001)),
                dict(
                    type='Caffe2Xavier',
                    override=dict(
                        type='Caffe2Xavier', name='pt_head'))
            ])),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
            sampler=dict(num=256))))
