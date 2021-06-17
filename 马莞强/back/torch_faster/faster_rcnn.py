from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import cv2
import sys
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
# os.chdir(sys.path[0])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained model's head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cpu")
model.load_state_dict(torch.load("torch_faster/fasterrcnn_resnet50_fpn_9.pth", map_location=torch.device(device)))
model.eval()
model.to(device)

def check_box(final_boxes,scores):
    overlap_idx = []
    for i in range(len(final_boxes)):
        box1 = final_boxes[i]
        for j in range(i + 1, len(final_boxes)):
            box2 = final_boxes[j]
            if box1[0] > box2[2] or box1[1] > box2[3] or box1[2] < box2[0] or box1[3] < box2[1]:
                continue
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            overlap_area = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            total_area = area1 + area2 - overlap_area
            if 2 * overlap_area > total_area or overlap_area > 0.9 * area2 or overlap_area > 0.9 * area1:
                if area1 > area2:
                    overlap_idx.append((i, j))
                else:
                    overlap_idx.append((j, i))

    temp_overlaps = overlap_idx
    overlaps = []
    for left_laps, right_laps in temp_overlaps:
        if scores[left_laps] > 0.5 and scores[right_laps] <= 0.5:
            overlaps.append(right_laps)
        elif scores[left_laps] <= 0.5 and scores[right_laps] > 0.5:
            overlaps.append(left_laps)
        else:
            overlaps.append(right_laps)

    for i in range(0, len(final_boxes)):
        if scores[i] < 0.5:
            overlaps.append(i)

    overlaps = np.unique(np.array(overlaps)).tolist()

    final_result = [final_boxes[i] for i in range(0, len(final_boxes)) if i not in overlaps]
    return [[int(i) for i in r.tolist()] for r in final_result]


def detect(cv2_image):
    transformer = A.Compose([ToTensorV2(p=1.0)])
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 255.0
    sample = {
        'image': img,
    }
    sample = transformer(**sample)
    img = sample["image"]
    output = model([img])[0]
    bboxes = output["boxes"]
    scores = output["scores"]
    return check_box(bboxes, scores)