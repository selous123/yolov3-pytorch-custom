from PIL import Image, ImageDraw
import torch
import numpy as np
def tensor2PIL(t, m = 255):
    if m == 1:
        t = t * 255
    t = t.permute(1,2,0).numpy()
    img = Image.fromarray(t.astype(np.uint8))
    return img

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def load_classes(path = "/store/dataset/rubbish_yolo/classes.names"):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().strip().split("\n")
    return names

"""
Param:
    image: PIL.Image
    box: array, xyxy
"""
def show_box_on_image(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(list(box), outline='red', width=3)
    del draw


def draw_box_on_tensor(tensor, boxes, m = 1, box_format='xyxy'):
    img = tensor2PIL(tensor, m)
    whwh = torch.tensor([img.size[0], img.size[1], img.size[0], img.size[1]])
    if box_format == 'xywh':
        boxes = xywh2xyxy(boxes)
    boxes = boxes * whwh
    show_box_on_image(img, boxes)
    return img
