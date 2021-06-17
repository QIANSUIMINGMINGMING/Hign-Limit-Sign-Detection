import os
import sys
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


m_state_dict = torch.load('yolo/model.pt')
new_model= models.resnet18(pretrained=False, progress=True)
num_fc_in = new_model.fc.in_features

new_model.fc = nn.Linear(num_fc_in, 2)

new_model.load_state_dict(m_state_dict)

new_model.eval()


test_transform = transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(), # 将PIL图像转为Tensor，并且进行归一化
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
                ])

def classify(img):
    img_=test_transform(img).unsqueeze(0)
    outputs = new_model(img_)
    return outputs[0][1] > outputs[0][0]