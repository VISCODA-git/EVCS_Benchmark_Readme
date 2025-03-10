_base_ = '../../grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = 'data/EVCSDataset_VISCODA_OneClass_V1.1/'
class_name = ('electric vehicle charging station')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='training.json',
        data_prefix=dict(img='training/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='validation.json',
        data_prefix=dict(img='validation/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/')))

val_evaluator = dict(ann_file=data_root + 'validation.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

max_epoch = 24

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

vis_backends = [dict(type='LocalVisBackend'), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
auto_scale_lr = dict(enable=True, base_batch_size=32)
