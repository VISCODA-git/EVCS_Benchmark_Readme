_base_ = "../../yolox/yolox_s_8xb8-300e_coco.py"
classes = ('wallbe ZAS', 'HYC 150', 'Compleo DUO IMS', 'Compleo DUO', 'HYC 300', 'Tritium PK350', 'ABB Terra 54', 'Tesla Supercharger')
model = dict(
    bbox_head=dict(num_classes=8)
)
data_root = "data/EVCSDataset_VISCODA_V1.1/"
train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    dataset = dict(
        dataset = dict(
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file="training.json",
            data_prefix=dict(img='training/')
        )
    )
)
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='validation.json',
        data_prefix=dict(img='validation/'),
    )
)
val_evaluator = dict(
    ann_file=data_root + 'validation.json',
    classwise=True,
)
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
    )
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    classwise=True,
)
num_last_epochs = 50
max_epochs = 250
base_lr = 0.01
train_cfg=dict(val_interval=5, max_epochs=max_epochs)
# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
load_from = data_root + "../yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"

vis_backends = [dict(type='LocalVisBackend'), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
autoscale_lr=dict(enable=True)
default_hooks=dict(checkpoint=dict(type="CheckpointHook", save_best="coco/bbox_mAP", rule="greater", max_keep_ckpts=1))