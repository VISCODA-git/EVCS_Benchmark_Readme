_base_ = "../../dino/dino-5scale_swin-l_8xb2-12e_coco.py"

data_root = 'data/EVCSDataset_VISCODA_V1.1/'
annotations_train = 'training.json'
annotations_val = 'validation.json'
annotations_test = 'test.json'
classes = ('wallbe ZAS', 'HYC 150', 'Compleo DUO IMS', 'Compleo DUO', 'HYC 300', 'Tritium PK350', 'ABB Terra 54', 'Tesla Supercharger')

model = dict(
    type="DINO",
    bbox_head=dict(num_classes=8)
)

train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=annotations_train,
        data_prefix=dict(img='training'),
    ),
    batch_size=1,
)
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=annotations_val,
        data_prefix=dict(img='validation')
    )
)
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=annotations_test,
        data_prefix=dict(img='test')
    )
)
val_evaluator = dict(
    ann_file=data_root + annotations_val,
    classwise=True
)
test_evaluator = dict(
    ann_file=data_root + annotations_test,
    classwise=True
)
vis_backends = [dict(type='LocalVisBackend'), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
auto_scale_lr = dict(enable=True)
default_hooks=dict(checkpoint=dict(type="CheckpointHook", save_best="coco/bbox_mAP", rule="greater", max_keep_ckpts=1))
load_from = data_root + "../dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
