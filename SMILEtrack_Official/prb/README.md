# YOLOv7 with PRBNet

## Performance 
### MS COCO
#### P5 Model

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>s</sub><sup>test</sup> | FPS |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv7-x** | 640 | **53.1%** | **71.2%** | **57.8%** | **33.8%** | **114** | 
|  |  |  |  |  |  |  |  
| [**PRB-FPN-CSP**](http://140.113.110.150:5000/sharing/AZc2PgpWN) | 640 | **51.8%** | **70.0%** | **56.7%** | **32.6%** |  **113** | 
| [**PRB-FPN-MSP**](http://140.113.110.150:5000/sharing/Es4M7Vprv) | 640 | **53.3%** | **71.1%** | **58.3%** | **34.1%** | **94** | 
| [**PRB-FPN-ELAN**](http://140.113.110.150:5000/sharing/orXcxcGSw) | 640 | **52.5%** | **70.4%** | **57.2%** | **33.4%** |  **70** | 
|  |  |  |  |  |  |  | 

#### P6 Model
| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup>  | FPS |
| :-- | :-: | :-: | :-: | :-:  |  :-: | 
| **YOLOv7-E6E** | 1280 | **56.8%** | **74.4%** | **62.1%** | **36**| 
|  |  |  |  |  |  |  |  
| [**PRB-FPN6**](http://140.113.110.150:5000/sharing/xvwUNT3zZ) | 1280 | **56.9%** | **74.1%** | **62.3%** | **31**| 
| [**PRB-FPN6-MSP**](http://140.113.110.150:5000/sharing/wavtpaPzu) | 1280 | **57.2%** | **74.5%** | **62.5%** | **27**| 
|  |  |  |  |  |  |  |    

## Installation & Getting started

Please refer to the [yolov7 README](./yolov7_README.md) to get started.

## Testing

Tested with: [`PyTorch Release 23.02`](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-02.html)

P5 Weights: [`prb-fpn-elan.pt`](http://140.113.110.150:5000/sharing/orXcxcGSw), [`prb-fpn-csp.pt`](http://140.113.110.150:5000/sharing/AZc2PgpWN) 

P6 Weights [`PRB-FPN6.pt`](http://140.113.110.150:5000/sharing/xvwUNT3zZ),
[`PRB-FPN6-MSP.pt`](http://140.113.110.150:5000/sharing/wavtpaPzu)



``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights prb-fpn-elan.pt --name prb-fpn-elan_640_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70304
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.36666
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.56971
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38975
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65053
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.70243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.54643
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.74958
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84504
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training for P5 model

``` shell
# train prb-fpn-csp models
python train.py --workers 8 --device 0 --batch-size 36 --data data/coco.yaml --epochs 330 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-CSP.yaml --weights '' --name PRB-FPN-CSP --hyp data/hyp.scratch.p5.yaml

# train prb-fpn-msp models
python train.py --workers 8 --device 0 --batch-size 25 --data data/coco.yaml --epochs 330 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-ELAN.yaml --weights '' --name PRB-FPN-ELAN --hyp data/hyp.scratch.p5.yaml
```

Single GPU training for P6 model

``` shell
# train PRB-FPN6-MSP models
python train_aux.py --workers 8 --device 0 --batch-size 14 --data data/coco.yaml --epochs 330 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-MSP.yaml --weights '' --name PRB-FPN6-MSP --hyp data/hyp.scratch.p6.yaml

# train PRB-FPN6-3PY models
python train_aux.py --workers 8 --device 0 --batch-size 28 --data data/coco.yaml --epochs 330 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-3PY.yaml --weights '' --name PRB-FPN6-3PY --hyp data/hyp.scratch.p6.yaml

```

Multiple GPU training for P5/6 model

``` shell
# train prb-fpn-elan p5 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 200 --data data/coco.yaml --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-ELAN.yaml --weights '' --name PRB-FPN-ELAN-8GPU --hyp data/hyp.scratch.p5.yaml

# train prb-fpn6-msp p6 models
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py --workers 8 --device 0,1,2 --sync-bn --batch-size 28 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-MSP.yaml --weights '' --name PRB-FPN6-MSP-2GPU --hyp data/hyp.scratch.p6.yaml

```


## Transfer learning

[`prb-fpn-csp.pt`](http://140.113.110.150:5000/sharing/AZc2PgpWN) [`prb-fpn-elan_training.pt`](http://140.113.110.150:5000/sharing/NE9UvOqPo) 
 [`prb-fpn6-msp-training.pt`](http://140.113.110.150:5000/sharing/MY2wRu4CI)


Single GPU finetuning for custom dataset

``` shell
# finetune p5 models (prb-fpn-csp,prb-fpn)
python train.py --workers 8 --device 0 --batch-size 36 --data data/custom.yaml --epochs 300 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-CSP.yaml --weights 'prb-fpn-csp.pt' --name prb-fpn-csp-custom --hyp data/hyp.scratch.custom.yaml

# finetune p6 models (prb-fpn6-3py)
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --epochs 300 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-3PY.yaml --weights 'prb-fpn6-3py_training.pt' --name prb-fpn6-3py-custom --hyp data/hyp.scratch.custom.yaml
```

## Re-parameterization

See [reparameterization-prb.ipynb](reparameterization-prb.ipynb)

## Inference

On video:
``` shell
python detect.py --weights prb-fpn.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights prb-fpn.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="65%"/>
    </a>
</div>

## What's New

- [2023.05.09] Support MSPNet, which was first presented at ICIP21, and its extended paper was published in T-ITS.
- [2023.03.30] Release P6 models.


## Citation

```
@ARTICLE{9603994,
  author={Chen, Ping-Yang and Chang, Ming-Ching and Hsieh, Jun-Wei and Chen, Yong-Sheng},
  journal={IEEE Transactions on Image Processing}, 
  title={Parallel Residual Bi-Fusion Feature Pyramid Network for Accurate Single-Shot Object Detection}, 
  year={2021},
  volume={30},
  number={},
  pages={9099-9111},
  doi={10.1109/TIP.2021.3118953}}
```

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
```
@INPROCEEDINGS{9506212,
  author={Ping-Yang, Chen and Hsieh, Jun-Wei and Gochoo, Munkhjargal and Chen, Yong-Sheng},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Light-Weight Mixed Stage Partial Network for Surveillance Object Detection with Background Data Augmentation}, 
  year={2021},
  volume={},
  number={},
  pages={3333-3337},
  doi={10.1109/ICIP42928.2021.9506212}}
```
```
@ARTICLE{9920960,
  author={Chen, Ping-Yang and Hsieh, Jun-Wei and Gochoo, Munkhjargal and Chen, Yong-Sheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Mixed Stage Partial Network and Background Data Augmentation for Surveillance Object Detection}, 
  year={2022},
  volume={23},
  number={12},
  pages={23533-23547},
  doi={10.1109/TITS.2022.3206709}}
```




## Acknowledgements


* https://github.com/WongKinYiu/yolov7



