_base_ = "./yolox_s_greenautoml_evcs.py"

# model settings copied from yolox_l_8x8b-200e_coco.py
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=8))

train_dataloader = dict(
    batch_size=8
)

load_from = "data/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"