import numpy as np
import cv2
import sys
import os
os.chdir(sys.path[0])

import number
import torch
from number import ConvNet
import cv2
import os
import sys
os.chdir(sys.path[0])
DEVICE = "cpu"
model = ConvNet().to(DEVICE)
model=torch.load("helloworld.pth",map_location=torch.device(DEVICE))
model.to(DEVICE)
model.eval()

import classification
from PIL import Image

from torch_faster import faster_rcnn


cap = cv2.VideoCapture("2.mp4")



def getBBoxes(cv2_image):
    ans = []
    bboxes = faster_rcnn.detect(cv2_image)
    for bbox in bboxes:
        cropped = cv2_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if classification.classify(Image.fromarray(cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB))):
            rbbox = []
            rbbox.append([bbox[0], bbox[1]])
            rbbox.append([bbox[2], bbox[1]])
            rbbox.append([bbox[2], bbox[3]])
            rbbox.append([bbox[0], bbox[3]])
            rbbox.append(number.give_result(model, cropped))
            ans.append(rbbox)
    return ans


def initiate_features(video_cap):
    feature_params = dict(maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7)
    the_lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ran_color = np.random.randint(0, 255, (100, 3))
    ############################################################## 运行检测模型，前两个值替换成检测到的坐标
    pt1 = np.array([50, 200, 1], dtype=np.float).transpose()
    pt2 = np.array([150, 210, 1], dtype=np.float).transpose()
    ##############################################################
    signal, old_frame = video_cap.read()
    bboxes = getBBoxes(old_frame)
    gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0Features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)  # 选取好的特征点，返回特征点列表
    frame_mask = np.zeros_like(old_frame)
    return bboxes, gray, the_lk_params, ran_color, p0Features, frame_mask


bboxes, old_gray, lk_params, color, p0, mask = initiate_features(cap)

count = 0

while 1:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 计算新的一副图像中相应的特征点额位置
    try:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) < 10:
            print("!")
            raise TypeError
    except TypeError:
        bboxes, old_gray, lk_params, color, p0, mask = initiate_features(cap)
        ret, frame = cap.read()
        if frame is None:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 计算新的一副图像中相应的特征点额位置
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    finally:
        H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3)
        for bbox in bboxes:
            nb1 = np.matmul(H, np.array([bbox[0][0], bbox[0][1], 1], dtype=np.float).transpose())
            nb2 = np.matmul(H, np.array([bbox[1][0], bbox[1][1], 1], dtype=np.float).transpose())
            nb3 = np.matmul(H, np.array([bbox[2][0], bbox[2][1], 1], dtype=np.float).transpose())
            nb4 = np.matmul(H, np.array([bbox[3][0], bbox[3][1], 1], dtype=np.float).transpose())
            bbox[0] = [nb1[0], nb1[1]]
            bbox[1] = [nb2[0], nb2[1]]
            bbox[2] = [nb3[0], nb3[1]]
            bbox[3] = [nb4[0], nb4[1]]
        # point1 = np.matmul(H, point1)
        # point2 = np.matmul(H, point2)
        img = frame

        count += 1
        if(count >= 20):
            bboxes = getBBoxes(img)
            count = 0
        print(img.shape)
        for bbox in bboxes:
            cv2.line(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (0, 255, 0), 1)
            cv2.line(img, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 1)
            cv2.line(img, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), (0, 255, 0), 1)
            cv2.line(img, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), (0, 255, 0), 1)
        cv2.putText(img, "{} [{:.2f}]".format("height limit", float(bbox[4])),
                        (int(bbox[0][0]), int(bbox[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        # print(H.shape)
        # print(point1.shape)
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     if i == 100:
        #         break
        #     a, b = new.ravel()  # ravel()函数用于降维并且不产生副本
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        # img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
