_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py'

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHeadE2E',
            iou_mask=True,
            bbox_refine=True,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=3, metric='bbox', gpu_collect=True)
