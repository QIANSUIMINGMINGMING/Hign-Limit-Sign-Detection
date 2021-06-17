#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from PIL import Image
import random
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TORCH_HOME']='/openbayes/input/input1/torch_m'

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# In[2]:


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# In[3]:


def read_data(dir_path):
    names=[]
    for name in os.listdir(dir_path):
        name=name.split(".")[0]
        names.append(name)
    return np.unique(names)

test_names=read_data("/openbayes/input/input0/test")
train_names=read_data("/openbayes/input/input0/train")

class SignData(Dataset):
    def __init__(self,root_path,names,transforms):
        self.root=root_path
        self.names=names
        self.transforms = transforms

    def __getitem__(self, index):
        name=self.names[index]
        annotation_file=name+".txt"
        img_name=name+".jpg"
        regions=[]
        with open(self.root+"/"+annotation_file) as file:
            for line in file.readlines():                          #依次读取每行  
                line = line.strip()                             #去掉每行头尾空白  
                if not len(line) or line.startswith('#'):       #判断是否是空行或注释行  
                    continue
                line_datas=line.split(" ")[1:]
                line_datas = [float(i) for i in line_datas]
                
                regions.append(line_datas)
        
        target={}
        
        bboxes = np.zeros((len(regions), 4))
        areas=np.zeros(len(regions),)
        labels=np.zeros(len(regions),)
        
        image=cv2.imread(self.root+"/"+img_name,cv2.IMREAD_COLOR)
        image=cv2.resize(image,(2048,2048))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image=image/255.0
        
        width=image.shape[0]
        height=image.shape[1]
#         print(width,height)
        
        i=0
        for region in regions:
            bbox=np.zeros(4)
            
            bbox[0]=(2*region[0]-region[2])*width/2
            bbox[1]=(2*region[1]-region[3])*height/2
            bbox[2]=(2*region[0]+region[2])*width/2
            bbox[3]=(2*region[1]+region[3])*height/2
#             for bb in bbox:
#                 if bb<0 or bb>2047:
#                     print(index,bb)
            bbox[bbox<0]=float(0)
            bbox[bbox>2047]=float(2047)
            
            bboxes[i]=bbox
            
            labels[i]=1
            b_height, b_width= region[3], region[2]
            areas[i]=height*width*b_height*b_width
            i=i+1
        
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas,dtype=torch.float32)
        iscrowd = torch.zeros((len(regions),), dtype=torch.int64)
        target["boxes"] = bboxes
        target["image_id"]=torch.tensor([index])
        target["labels"]=labels
        target["area"]=areas
        target["iscrowd"] = iscrowd
        
        if self.transforms:
            sample = {
            'image': image,
            'bboxes': target['boxes'],
            'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        return image, target
    
    
    def __len__(self):
        return len(self.names)
    
    @staticmethod
    def get_train_transform():
        return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    @staticmethod
    def get_valid_transform():
        return A.Compose([
        ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# In[4]:


train_dataset=SignData("/openbayes/input/input0/train",train_names,SignData.get_train_transform())
valid_dataset=SignData("/openbayes/input/input0/test",test_names,SignData.get_valid_transform())
# img,target=train_dataset[2145]
# cv2.imwrite("da.jpg",img)
# print(target)
# datas = [train_dataset[i] for i in range(2)]
# imgs = [d[0].permute(1, 2, 0).numpy() for d in datas]
# # print(len(train_dataset))
# i=0
# j=0
# for data,target in train_dataset:
# #     data_t=torch.isnan(data)
# #     if True in data_t:
# #         i=i+1
#     for t in target.values():
#         tar_t=torch.isnan(t)
#         if True in tar_t:
#             j=j+1
        
# print(i,j)


# In[5]:


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


# In[6]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print(model.roi_heads.box_predictor)


# In[36]:


num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained model's head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
print(model.roi_heads.box_predictor)


# In[7]:


from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

train_data_loader = DataLoader(
  train_dataset,
  batch_size=8,
  shuffle=True,
  num_workers=4,
  collate_fn=collate_fn
)

valid_data_loader = DataLoader(
  valid_dataset,
  batch_size=8,
  shuffle=False,
  num_workers=4,
  collate_fn=collate_fn
)


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# create an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(params,lr=1e-1)
# criterion = F.nll_loss
# optimizer =torch.optim.Adam(params,lr=0.001,weight_decay=0.0005)
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
# logs,losses = find_lr()
# plt.plot(logs[10:-5],losses[10:-5])
# create a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

# train it for 10 epochs
num_epochs = 10


# In[40]:

import seaborn as sns

import time
from tqdm import tqdm
#from tqdm.notebook import tqdm as tqdm

itr = 1

total_train_loss = []
total_valid_loss = []

losses_value = 0

for epoch in range(num_epochs):

    start_time = time.time()

    # train ------------------------------

    model.train()
    train_loss = []

    pbar = tqdm(train_data_loader, desc='let\'s train')
    for images, targets in pbar:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        for loss in loss_dict.values():
            assert torch.isnan(loss).sum() == 0, print(loss)

        losses = sum(loss for loss in loss_dict.values())
        losses_value = losses.item()
        train_loss.append(losses_value)

        optimizer.zero_grad()
        losses.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 5, norm_type=2)
        optimizer.step()

        pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {losses_value}")
        itr += 1

    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    # valid ------------------------------

    with torch.no_grad():
        valid_loss = []

        for images, targets in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            valid_loss.append(loss_value)
    
    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_'+str(epoch+1)+'.pth')

    epoch_valid_loss = np.mean(valid_loss)
    total_valid_loss.append(epoch_valid_loss)

    if epoch>1:

        plt.figure(figsize=(8, 5))
        sns.set_style(style="whitegrid")
        sns.lineplot(x=range(1, len(total_train_loss)+1), y=total_train_loss, label="Train Loss")
        sns.lineplot(x=range(1, len(total_train_loss)+1), y=total_valid_loss, label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig("loss.png")

    # print ------------------------------

    print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
        f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")


# In[ ]:





# In[ ]:





# In[ ]:




