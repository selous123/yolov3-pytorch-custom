
该文件用来记录 ultralytics库 中 对于yolo模型在coco数据集上的表现(tricks)的一些讨论:

1. [如何进行 Testing/Inference Augmentation？](https://github.com/ultralytics/yolov3/issues/931)

2. [DIOU-NMS？](https://github.com/ultralytics/yolov3/pull/795)

3. [在YOLO-v4的改进提升中，其中50%来自架构，50%来自训练方式](https://github.com/ultralytics/yolov5/issues/6#issuecomment-643644347)

glenn-jocher: "Perhaps it means training methods and loss functions are becoming more important these days than architecture, since after all yolov3 used to be near 33 AP, and we've pulled it up to 45.6 now with no changes at all to the architecture."