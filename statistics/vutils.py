from PIL import Image, ImageDraw
from my_utils.utils import *
def tensor2PIL(t, m = 255):
    if m == 1:
        t = t * 255
    t = t.permute(1,2,0).numpy()
    img = Image.fromarray(t.astype(np.uint8))
    return img


"""
Param:
    image: PIL.Image
    box: array
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
