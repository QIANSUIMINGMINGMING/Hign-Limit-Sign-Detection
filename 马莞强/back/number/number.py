#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# In[2]:


# img = cv2.imread("slices/1.jpg")
# img=cv2.resize(img,(100,100))
# size=100
# gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


# In[3]:


# gray=cv2.GaussianBlur(gray,(11,11),1)
# gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)

def make_pic(img,pic_size=100):
    img=cv2.resize(img,(pic_size,pic_size))
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(11,11),1)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)
    return gray,pic_size


# In[4]:



def horizon_split(src_img,size):
    lines=[]
    for line in src_img:
        l = sum(line)
        lines.append(l)

    lines=np.array(lines)
#     plt.plot(lines)

    i=0
    left_highs=[]
    right_highs=[]
    line_max=np.mean(lines)
    step=int(size/20)
    while i<size:
        left=0
        right=len(lines)
        if i>step:
            left=i-step
        if i<=right-step:
            right=i+step
        if max(lines[left:right])==lines[i] and lines[i]>line_max:
            if i<=int(size/2):
                left_highs.append(i)
            else:
                right_highs.append(i)
            i=right
        else:
            i=i+1

    i=0
    left_lows=[]
    right_lows=[]
    step=step*2
    line_max=0.6*np.mean(lines[left_highs[0]:right_highs[-1]])
    # line_max=np.mean(lines[for i in lines: i>0.01*line_max])

    while i<size:
        left=0
        right=len(lines)
        if i>step:
            left=i-step
        if i<=right-step:
            right=i+step
        if min(lines[left:right])==lines[i] and lines[i]<line_max:
            if i<=int(size/2):
                left_lows.append(i)
            else:
                right_lows.append(i)
            i=right
        else:
            i=i+1
    
    return left_lows[-1],right_lows[0]

# i=0
# left_highs=[]
# right_highs=[]
# line_max=0.7*np.mean(lines[left_highs[0].key:right])
# while i<100:
#     left=0
#     right=len(lines)
#     if i>5:
#         left=i-5
#     if i<=right-5:
#         right=i+5
#     if max(lines[left:right])==lines[i] and lines[i]>line_max:
#         if i<=50:
#             left_highs.append({i:lines[i]})
#         else:
#             right_highs.append({i:lines[i]})
#         i=right
#     else:
#         i=i+1

# print(left_lows)
# print(right_lows)
# print(left_highs)
# print(right_highs)


# In[5]:


def vertical_split(vert, size):
    lines=[]
    for line in vert:
        l = sum(line)
        lines.append(l)

    lines=np.array(lines)

    i=0
    highs=[]
    line_max=np.mean(lines)
    step=int(size/20)
    while i<size:
        left=0
        right=len(lines)
        if i>step:
            left=i-step
        if i<=right-step:
            right=i+step
        if max(lines[left:right])==lines[i] and lines[i]>line_max:
            highs.append(i)
            i=right
        else:
            i=i+1

    i=0
    lows=[]

    line_mean=0.01*np.mean(lines[highs[0]:highs[-1]])
    while i<size:
        left=0
        right=len(lines)
        if i>step:
            left=i-step
        if i<=right-step:
            right=i+step
        if min(lines[left:right])==lines[i] and lines[i]<line_mean:
            if len(lows)>0:
                mid=int((i+lows[-1])/2+1)
                if lines[mid]<line_mean:
                    lows.pop(-1)
                    lows.append(mid)
                    i=right
                    continue
            lows.append(i)
            i=right
        else:
            i=i+1

    r_lows=[]
    for low in lows:
        if low<highs[-1] and low>highs[0]:
            r_lows.append(low)

    crop=-1
    pos=-1
    if len(r_lows)>3:
        for i in range(len(r_lows)-1):
            if r_lows[i+1]-r_lows[i]>28:
                if lines[r_lows[i+1]]>=lines[r_lows[i]]:
                    crop=r_lows[i]+6
                    pos=i+1
                else:
                    crop=r_lows[i+1]-4
                    pos=i+2
                break
    if crop>0:
        r_lows.insert(pos,crop)
                
    return r_lows


# In[6]:


# image,size=make_pic("OIP (1).jfif")
# l,r=horizon_split(image,size)
# # plt.imshow(image[l:r])
           
# vert1=image[l:r]
# vert1=np.rot90(vert1)
# # plt.imshow(vert1)
# points=vertical_split(vert1, size)


# In[7]:


# print(points)


# In[8]:


# plt.imshow(vert1)


# In[32]:


def crop_empty(sub_pic):
    i=0
    for line in sub_pic:
        if sum(line)<5:
            i=i+1
        else:
            break
    return i

def get_targets(pic,low_points):
    targets=[]
    num=len(low_points)-1
    # if num<2:
    #     num=num+1
    for i in range(num-1):
        left=low_points[num-i-1]
        right=low_points[num-i]
        if right-left<10:
            continue
        target=pic[left:right]
        for k in range(4):
            crop=crop_empty(target)
            target=target[crop:]
            target=np.rot90(target)
        for line in target:
            if sum(line)<=4:
                start
#         for j in range(3):
#             target=np.rot90(target)
        target=cv2.resize(target,(28,28))
        for s in range(3):
            target=np.rot90(target)
            target=target.copy()
        targets.append(target)
    return targets

# ts=get_targets(vert1,points)


# In[33]:


# t=ts[1]
# # t=t[7:]
# t=np.rot90(t)
# t=np.rot90(t)
# t=np.rot90(t)
# t=cv2.resize(t,(28,28))
# plt.imshow(t)


# In[43]:





# In[44]:


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

DEVICE = "cpu"
model = ConvNet().to(DEVICE)
model=torch.load("number/helloworld.pth",map_location=torch.device(DEVICE))
model.to(DEVICE)
model.eval()
transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

# data=transform(t).unsqueeze(0)
# output=model(data).squeeze(0)
# print(output)
# result=torch.argmax(output)
# print(int(result))

def give_result(cv2_img):
    image,size=make_pic(cv2_img)
    l,r=horizon_split(image,size)
    vert=image[l:r]
    vert=np.rot90(vert)
    points=vertical_split(vert, size)
    ts=get_targets(vert,points)
    # print("AAAAAAAAAAAA")
#     t=ts[0]
#     t=cv2.resize(t,(28,28))
#     plt.imshow(t)
    i=0
#     print(ts[0])
    result=0
    for a in ts:
        data=transform(a).unsqueeze(0)
        output=model(data).squeeze(0)
        # print(int(torch.argmax(output)))
        result=result+int(torch.argmax(output))*pow(0.1,i)
        i=i+1
    return result


# In[45]:


# give_result("OIP (1).jfif")


# In[ ]:




