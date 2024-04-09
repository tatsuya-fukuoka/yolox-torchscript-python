import time

import numpy as np
import cv2
import torch
import torchvision

from yolox.coco_classes import COCO_CLASSES
from yolox.yolox_torchscritpt import Yoloxtorchscript

model_path = "yolox-tiny.torchscript.pt"
device = 'cpu'
input_size = 416
cls_names=COCO_CLASSES
confthre = 0.3
nmsthre = 0.25

predictor = Yoloxtorchscript(
    model_path,
    input_size,
    device,
    cls_names,
    confthre,
    nmsthre)

image = cv2.imread('000000.jpg')

for i in range(5):
    start = time.time()
    predict = predictor.detect(image)
    print('Proc Time: ', time.time()-start)

result_image = predictor.visual(predict[0])
cv2.imwrite('result_torchscript.jpg', result_image)
