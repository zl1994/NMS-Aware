# NMS-Aware
Codes for "Bridging the gap between one-to-many and one-to-one label assignment via NMS-aware alignment module". This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection) framework.
## Prerequisites
- MMDetection version 2.11.0.
- Please follow official [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) guides.
## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'.

bash ./tools/dist_train.sh configs/NMS-aware/fcos_r50_fpn_o2m_o2o_2GCN_1x.py 2
```

## Inference

```python
bash ./tools/dist_test.sh configs/NMS-aware/fcos_r50_fpn_o2m_o2o_2GCN_1x.py work_dirs/fcos_r50_fpn_o2m_o2o_2GCN_1x/epoch_12.pth 4 --eval bbox
```

## Models

We provide the following trained models. All models are trained with 16 images in a mini-batch. It's normal to observe ~0.2AP noise in all methods.

Model | MS train | Lr schd | mAP| Config | Download
---|:---:|:---:|:---:|:---:|:---:
FCOS_R50_FPN_o2m_o2o_2GCN_1x   | N | 1x | 39.7| [config](configs/NMS-aware/fcos_r50_fpn_o2m_o2o_2GCN_1x.py) | [baidu](https://pan.baidu.com/s/1ZQWsSyWRvJfNeCZoHMt-Jw)
FCOS_R101_FPN_o2m_o2o_2GCN_1x   | N | 1x | 42.0| [config](configs/NMS-aware/fcos_r101_fpn_o2m_o2o_2GCN_1x.py) | [baidu](https://pan.baidu.com/s/1425pZx7ppaA4Kugc4koiIQ)
FCOS_R50_FPN_o2m_o2o_2GCN_3x   | Y | 3x | 42.6| [config](configs/NMS-aware/fcos_r50_fpn_o2m_o2o_2GCN_3x_ms.py) | [baidu](https://pan.baidu.com/s/1pVldhweuqQm5oWqFnCaFXA)
Faster_R50_FPN_o2m_o2o_2GCN_1x | N | 1x | 37.9| [config](configs/NMS-aware/faster_rcnn_r50_fpn_o2m_o2o_2GCN_1x.py) | [baidu](https://pan.baidu.com/s/1nTQSFRd_MsduMr0RN6dG8Q)
Faster_R101_FPN_o2m_o2o_2GCN_1x | N | 1x | 39.8| [config](configs/NMS-aware/faster_rcnn_r101_fpn_o2m_o2o_2GCN_1x.py) | [baidu](https://pan.baidu.com/s/1tUbqBBOB89-EF82UsUAkgg)
