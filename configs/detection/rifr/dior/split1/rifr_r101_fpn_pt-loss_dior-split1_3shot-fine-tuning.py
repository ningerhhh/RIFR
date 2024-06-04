_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_dior.py',
    '../../../_base_/schedules/schedule.py',
    '../../rifr_r101_fpn_pt_loss.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDIORDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT1_3SHOT')],
        num_novel_shots=3,
        num_base_shots=3,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))

evaluation = dict(
    interval=2000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=2000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])
runner = dict(max_iters=6000)

custom_hooks = [
    dict(
        type='PTLossDecayHook',
        decay_steps=(6000, 8000),
        decay_rate=2)
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            with_weight_decay=True,
            loss_contrast=dict(loss_weight=0.5))))



load_from = ('./work_dirs/rifr_r101_fpn_dior-split1_base-training/base_model_random_init_bbox_head.pth')

work_dir = './work_dirs/rifr_r101_fpn_pt-loss_dior-split1_3shot-fine-tuning/'
