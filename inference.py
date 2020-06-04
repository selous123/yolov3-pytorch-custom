from my_utils.datasets import *
import cv2
import numpy as np
from models import Darknet
import torch
from torchvision import transforms as transforms
import torchvision

img_size = 512
file_name = '/home/lrh/program/git/object_detection/code/yolov3/asserts/img_17247.jpg'
img0 = cv2.imread(file_name)
img = letterbox(img0, new_shape=img_size)[0]
#img = img / 255.0
print(img.shape)
img = img[:, :, ::-1]  # BGR to RGB, to 3x416x416
#img = np.ascontiguousarray(img)
img = transforms.ToTensor()(np.array(img))
img = img.unsqueeze(0)

#img = img.cuda()

#model = Darknet('submit/yolov4/model/csdarknet53s-panet-spp.cfg', img_size=img_size)
#model.load_state_dict(torch.load('submit/yolov4/model/best.pkl', map_location='cpu'))
model = Darknet('trained_models/yolov3-spp-default/yolov3-spp-44.cfg', img_size=img_size)
model.load_state_dict(torch.load('trained_models/yolov3-spp-default/best.pt', map_location='cpu')['model'])
#model = model.cuda()
model.eval()

torchvision.utils.save_image(img, 'input.jpg')


pred = model(img)[0]

print(pred[0,:4,:4])
