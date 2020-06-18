# yolov3-pytorch-custom
yolov3 for rubbish detection

## 1. 比赛总结

### 1.0. Baseline

**Network Architecture**: yolo-v3 = (darknet53 + FPN + yolo-head)

**Training strategy**: warm-up, cosine learning rate decay, Multi-scale training， SGD optimizer

**Loss**: GIOU, iou_net (obj损失以GIOU值为标记) 

**Data Argumentation**: mosica, flip, hsv_augmentation, random_affine

**NMS**： Merge-NMS

### 1.1. 在该数据上 work 的调参技巧

a. spp (spatial pyramid pooling)

b. kmeans algorithm for clustering anchors.

c. stitcher augmentation

d. multi scale testing.


### 1.2. 该 work 却不 work 的技巧 ---没尽力调

a. 常用正则化方法，通过分析网络有很严重的过拟合, 但常用的解决过拟合方法并没有效果如：dropblock, L1&L2 regulization, label smoothing.

b. 小物体检测mAP的值要明显低于其他尺度的mAP. 在计算损失时，通过损失加权，增大小物体的损失 (2 - w * h) 没效果。

c. 某些类的mAP值要普遍比其他类低很多，我们通过mAP的值对样本采样进行重加权, 没效果。通过增加每个类的数据，提升较少。

d. focal loss, 没提升，网络很难收敛


### 1.3. 目前模型存在的问题

a. 在训练集上过拟合

b. 某些类 mAP 比较差

c. 小物体 mAP 普遍比较差

d. 数据集中存在遮挡的问题

### 1.4. 未来想尝试的技巧

a. 将损失使用传统 YOLO-v3 的形式实现。在其中添加 IOU-Net 分支 

b. TSD (SenseTime 1st place) 在网络中将框预测和类别预测进行解耦

c. 改进 NMS 和 Testing

## 2. 代码说明

### 2.1. Training

**command:**
默认模型使用 spatial pyramid pooling 结构和 stitcher 数据增广。训练集验证集分为9:1的比例。模型以及训练参数会保存在文件夹 baseline-stitcher (save) 参数下。

```
python3 train.py --cfg cfg/yolov3-spp-44.cfg --data data/rubbish.data --weights weights/yolov3-sppu.pt --batch-size 16 --epochs 120 --save baseline-stitcher
```

其他训练参数可以参考train.py文件中的参数设置部分:

```
parser.add_argument('--reg-ratio', type=float, default=0.0, help='reg_ratio for L1&L2 regulization to weights')
parser.add_argument('--ssd-aug', action='store_true', help='use ssd augmentation or not')
parser.add_argument('--image-weights', action='store_true', help='use image_weights or not')
parser.add_argument('--smooth-ratio', type=float, default=0.0, help='label smooth ratio for cls bceloss')
parser.add_argument('--lbox-weight', action='store_true', help='weight box loss by size of gt-box or not')
```

### 2.2. Testing

**command:**
```
python3 test.py --cfg trained_models/baseline-stitcher/yolov3-spp-44.cfg --data data/rubbish.data --weights trained_models/baseline-stitcher/best.pt --batch-size 8
```

### 2.3. Submit

**command:**
```
python gen_submit_dir.py -m trained_models/baseline-stitcher -s submit/baseline-stitcher
```

然后将submit/baseline-stitcher下生成的model文件夹上传到 华为云的obs云储存中
```
cd submit/baseline-stitcher
obsutil cp -r -f model/ obs://hellopytorch/baseline-stithcer
```

最后部署到modelart中，提交即可。


<center><font size='5'> <b>Reference</b> </font> </center>

[Crowdhuman人体检测比赛第一名经验总结](https://zhuanlan.zhihu.com/p/68677880)

[CVPR 2020丨​商汤TSD目标检测算法获得Open Images冠军](https://zhuanlan.zhihu.com/p/131576433)

[目标检测比赛中的tricks（已更新更多代码解析）](https://zhuanlan.zhihu.com/p/102817180)

[Stithcer Augmentation](https://www.zhihu.com/question/390191723/answer/1185984775)

[Imbalance Problems in Object Detection: A Review](https://zhuanlan.zhihu.com/p/82371629)
