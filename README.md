# yolov3-pytorch-custom
yolov3 for rubbish detection


### 1. 基础知识 (object detection)
#### 1.1. 目标检测之 Anchor
[Anchor详解](https://zhuanlan.zhihu.com/p/55824651)

#### 1.2. 目标检测指标 mAP
[mAP指标详解](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge)

[github讲解mAP指标](https://github.com/rafaelpadilla/Object-Detection-Metrics)

Steps:

1. First, compute the Average Precision (AP) for each class, 

2. and then compute the mean across all classes. 

##### 如何计算AP

**The key here is to compute the AP for each class, in general for computing Precision (P) and Recall (R) you must define what are: True Positives (TP), False Positives (FP), True Negative (TN) and False Negative (FN).**

1. TP: are the Bounding Boxes (BB) that the intersection over union (IoU) with the ground truth (GT) is above 0.5

2. FP: two cases 
(a) the BB that the IoU with GT is below 0.5 
(b) the BB that have IoU with a GT that has already been detected.

3. TN: there are not true negative, the image are expected to contain at least one object

4. FN: those images were the method failed to produce a BB

Now each predicted BB have a confidence value for the given class. So the scoring method sort the predictions for decreasing order of confidence and compute the P = TP / (TP + FP) and R = TP / (TP + FN) for each possible rank k = 1 up to the number of predictions. So now you have a (P, R) for each rank those P and R are the "raw" Precision-Recall curve. To compute the interpolated P-R curve foreach value of R you select the maximum P that has a corresponding R' >= R.

There are two different ways to sample P-R curve points according to voc devkit doc. For VOC Challenge before 2010, we select the maximum P obtained for any R' >= R, which R belongs to 0, 0.1, ..., 1 (eleven points). The AP is then the average precision at each of the Recall thresholds. For VOC Challenge 2010 and after, we still select the maximum P for any R' >= R, while R belongs to all unique recall values (include 0 and 1). The AP is then the area size under P-R curve. Notice that in the case that you don't have a value of P with Recall above some of the thresholds the Precision value is 0.


From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. Currently, the interpolation performed by PASCAL VOC challenge uses all data points, rather than interpolating only 11 equally spaced points as stated in their paper.


##### 如何计算mAP

mAP: mAP 是 AP = [.50:.05:1.)，也就是IOU_T设置为0.5,0.55,0.60,0.65……0.95，算十个APx，然后再求平均，得到的就是mAP

AP50: 就是IOU_T设置为0.5

AP75: 就是IOU_T设置为0.75



### 2. 代码实现 (code)
#### 2.1 数据处理

步骤1. 通过src/dataset中的数据结构，将数据的元信息读入到代码中，储存变量为roidb，数据格式为dict.

数据读取的类结构：

src/dataset/imdb.py
src/dataset/rbdata.py

```
roidb数据 字典格式为 
    'boxes': boxes,
    'gt_classes': gt_classes,
    'gt_ishard': ishards,
    'gt_overlaps': overlaps,
    'flipped': False,
    'seg_areas': seg_areas
```

步骤2. 通过src/roi_data_layer/xx.py中的函数，处理roidb，例如(roidb.py)：

1. (roidb.py/prepare_roidb 函数) 丰富roidb的数据条目， 

2. (roidb.py/rank_roidb_ratio 函数) 计算roidb数据中的长宽比，为后面处理batch中的数据做准备（同一个batch中数据的长款得相同） 

3. (roibatchLoader.py Line 166 ~ Line 191) 类似与pytorch中的dataset数据集，返回一个数据，数据格式为 (data, im_info: hws, gt_boxes, num_boxes) 而且会根据数据的长宽比，填充数据，使得同一个batch中数据的长宽相同。

4. (train_net.py) 实现了sampler类，保证batch中的数据在ratio_list中是连续的，方便之后的操作。

Sampler测试结果:

```
class sampler():
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)
    def __len__(self):
        return self.num_data

s = sampler(20, 4)

for a in s:
    print(a, end=' ')
##output:
##8 9 10 11  12 13 14 15  4 5 6 7  16 17 18 19  0 1 2 3 
```

步骤3. 封装成DataLoader，用于模型训练。

步骤4. 分出训练和验证集，用于选择模型。


**2020/05/12. 展示一下数据集**

code/src/example.jpg

#### 2.2. 配置文件设置及保存
[YAML 教程](https://www.runoob.com/w3cnote/yaml-intro.html)

YACS

保存网络训练的配置文件:

```
def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            #f.write(json.dumps(cfg, cls=NumpyEncoder))
            f.write("cfgs:")
            json.dump(cfg, f, cls=NumpyEncoder, indent=2)
            f.write("\n args:")
            json.dump(args.__dict__, f, cls=NumpyEncoder, indent=2)
        #logger.info("Full config saved to {}".format(path))
        print("Full config saved to {}".format(path))
```

<font color='red'>近期需完成的任务: 需要将网络构建的代码也转成配置文件的形式！！</font>
#### 2.3. Logger类定义记录中间结果


### Time Log:

2020/05/07: 熟悉pytorch Detectron2框架，

2020/05/08: Baseline. 已上传至华为云训练和测试/已本地在tensorflow1.14的环境下运行训练。接下来将baseline改为pytorch版本。

2020/05/09: 基本弄清楚目标检测中faster-rcnn的网络结构设计细节。

2020/05/10: 处理完成数据集相关代码 Load Data，生成DataLoader可以用来训练

2020/05/11: 需开始写相关代码，并本地运行。

**Problem 1:**

```
File "/home/ibrain/git/object_detection/code/src/model/rpn/proposal_target_layer_cascade.py", line 190, in _sample_rois_pytorch
    raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
ValueError: bg_num_rois = 0 and fg_num_rois = 0, this should not happen!
```

初始解决方案: 应该是实验参数设置的问题，需要进一步弄懂所有的代码，然后解决该训练的问题。

Bug原因: 读取gt_box时，所有坐标-1，代码位置src/datasets/rbdata.py Line 128

```
# Make pixel indexes 0-based
x1 = float(bbox.find('xmin').text) - 1
y1 = float(bbox.find('ymin').text) - 1
x2 = float(bbox.find('xmax').text) - 1
y2 = float(bbox.find('ymax').text) - 1
```

解决方案: 

1. 去掉该文件中的-1操作

2. src/datasets/imdb.py Line 117&118 去掉　-1　操作


2020/05/12: 完成 Detectron2 框架的数据加载部分，即将在lrh-dev分支上运行相关代码。

2020/05/13: 完成 Detectron2 框架的 Train 和 Validation 部分，准备开始重构验证代码。

**比赛数据集问题: 在使用 Detectron2 框架中的 API 读取数据时，我们发现在数据集中存在误标记，1. hw 标反，20190816_095611 和 20190816_095633。 2. 标记错误 img_1877 和 img_1882。**


2020/05/14: 数据集按照train和val 9:1 的比例已经分好。

之后需要完善 Evaluator 的过程，以及代码 **(已完成Evaluator相关代码，并测试现有模型的结果)**。

在 Detectron2 框架添加 log 记录，完善 Log 类记录loss, 
    并且画图（学会使用tensorboardX用来可视化）

Detectron2 训练过程中的记录是依靠 python logging 模块 (detectron2.utils.logger.py) 和自定义 EventWriter 类(detectron2.utils.event.py)。 EventWriter类是通过pytorch中的Hook机制调用，在训练过程中保存希望保存的变量 (detectron2.engine.DefaultTrainer: L387-392)

以及如何将网络也改写成 参数文件 配置的形式。

2020/05/15: 已在华为云上部署 Pytorch 分类模型； 可提交测试。 尝试部署 Detectron2 包，由于版本依赖问题，失败！决定采用替代方案。

1. Trainer 类规范化 (不需要)

2. Pytorch 模型提交 (已提交Pytorch分类模型)

2020/05/16: 首先自定义模型架构，提交目标检测模型到华为云上。

2020/05/17: 

1. 把pytorch faster-rcnn 框架部署到云上。[失败，未部署成功]

2. 跑pytorch YoLo-v3的代码。[ing]

Detectron2 中自定义模型架构[因某些层依赖fvcore,故无法将detectron2的网络定义独立出来。]

2020/05/18 部署pytorch faster-rcnn到华为云服务器失败，准备直接copy github上面的部署代码。先把yolov3-pytorch部署到华为云上。

2020/05/19 解决pytorch yolov3的部署问题

2020/05/21 解决pytorch-yolov3的训练问题

2020/05/22 初始模型测试评分 mAP: 0.4953. 提交 自定义anchor训练的模型。正在训练yolov3-spp模型。

接下来需要完成的任务:

1. 统计模型在数据集上的表现，找出目前模型所存在的问题

2. 弄清楚yolov3和yolov3-spp的原理

2.1. 弄清数据预处理部分

3. 完善yolov3包的日志等功能 (滞后完成).


2020/05/26:

#### RTX2080Ti 测试阶段显存占用不稳定.

Evoluation Algorithm:

    1. 不能在代码中设置 generation 代数，与 apex包 一起用会出现显存泄露。

    2. 不能在shell脚本中循环运行python代码，会有多个python运行，导致运行显存不够。

    3. set generation, without apex.
    
2020/06/17: 

#### 0. Baseline

**Network Architecture**: yolo-v3 = (darknet53 + yolo-head)

**Training strategy**: warm-up, cosine learning rate decay, Multi-scale training， SGD optimizer

**Loss**: GIOU, iou_net (obj损失以GIOU值为标记) 

**Data Argumentation**: mosica, flip, hsv_augmentation, random_affine

**NMS**： Merge-NMS

#### 1. 在该数据上 work 的调参技巧

a. spp (spatial pyramid pooling)

b. kmeans algorithm for clustering anchors.

c. stitcher augmentation

d. multi scale testing.


#### 2. 该 work 却不 work 的技巧 ---没尽力调

a. 常用正则化方法，通过分析网络有很严重的过拟合, 但常用的解决过拟合方法并没有效果如：dropblock, L1&L2 regulization, label smoothing.

b. 小物体检测mAP的值要明显低于其他尺度的mAP. 在计算损失时，通过损失加权，增大小物体的损失 (2 - w * h) 没效果。

c. 某些类的mAP值要普遍比其他类低很多，我们通过mAP的值对样本采样进行重加权, 没效果。通过增加每个类的数据，提升较少。

d. focal loss, 没提升，网络很难收敛


#### 3. 目前模型存在的问题

a. 在训练集上过拟合

b. 某些类 mAP 比较差

c. 小物体 mAP 普遍比较差

d. 数据集中存在遮挡的问题

#### 4. 未来想尝试的技巧

a. 将损失使用传统 YOLO-v3 的形式实现。在其中添加 IOU-Net 分支 

b. TSD (SenseTime 1st place) 在网络中将框预测和类别预测进行解耦

c. 改进 NMS 和 Testing

<center><font size='5'> <b>Reference</b> </font> </center>

[Crowdhuman人体检测比赛第一名经验总结](https://zhuanlan.zhihu.com/p/68677880)

[CVPR 2020丨​商汤TSD目标检测算法获得Open Images冠军](https://zhuanlan.zhihu.com/p/131576433)

[目标检测比赛中的tricks（已更新更多代码解析）](https://zhuanlan.zhihu.com/p/102817180)

[Stithcer Augmentation](https://www.zhihu.com/question/390191723/answer/1185984775)

[Imbalance Problems in Object Detection: A Review](https://zhuanlan.zhihu.com/p/82371629)
