_base_ = [
    './grounding_dino_r50_scratch_8xb2_1x_coco.py'
]
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco/grounding_dino_r50_scratch_1x_coco-fe0002f2.pth'
lang_model_name = 'bert-base-uncased'
#lang_model_name = '../../work_dirs/bert-base-uncased'


data_root = 'data/EVCSDataset_VISCODA_V1.1_additional_data/'
class_name = ('electric vehicle charging station')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])


model = dict(bbox_head=dict(num_classes=num_classes))


train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='additional.json',
        data_prefix=dict(img='additional/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='additional.json',
        data_prefix=dict(img='additional/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='additional.json',
        data_prefix=dict(img='additional/')))

val_evaluator = dict(ann_file=data_root + 'additional.json')
test_evaluator = dict(ann_file=data_root + 'additional.json')

# We did not adopt the official 24e optimizer strategy
# because the results indicate that the current strategy is superior.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))
# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3], # change this to 3 from 11 on 20241107: no improvements after first a few epochs already
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
vis_backends = [dict(type='LocalVisBackend'), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')