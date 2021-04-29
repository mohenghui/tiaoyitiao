# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import subprocess
import random
import imutils
import os
press_coefficient = 1.68/2*pow(3,0.5)
swipe_x1 = 0; swipe_y1 = 0; swipe_x2 = 0; swipe_y2 = 0
def plt_show0(img):
    # print(img.shape)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
def self_detect(img):
    # shape[1]是宽shape[0]是高
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.7)
    # 原本是1080
    # 1080*0.3到1080*0.7的范围
    region=img[region_upper:region_lower,:]
    color_lower=np.int32([105,25,45])
    color_upper=np.int32([135,125,130])
    img_hsv=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)
    print(img_hsv.shape)
    # channels=cv2.split(img_hsv)
    # 打印通道情况
    # print(len(channels))
    # for i in range(len(channels)):
    #     print(channels[i])
    # cv2.imshow("hsv",img_hsv)
    # inRange相当于threshold()函数将感兴趣的区域变成255(白色),
    # 将不感兴趣去的区域设置为0黑色
    color_mask=cv2.inRange(img_hsv,color_lower,color_upper)
    # 打印二值图
    # cv2.imshow("color_mask",color_mask)
    contours = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]

    # 打印轮廓信息
    # print("轮廓信息,坐标%s,轮廓数量%d"%(contours,len(contours)))
    # 寻找轮廓
    if len(contours)>0:
        # 将最大的那组轮廓拿出
        # 最小的话就定位到头部
        # max_contour=min(contours,key=cv2.contourArea)
        max_contour=max(contours,key=cv2.contourArea)

        max_contour=cv2.convexHull(max_contour)
        # 轮廓的凸包
        # 把点连接起来
        # print(max_contour)
        # cv2.imshow("max_contour",max_contour)
        # 获取矩形的位置信息
        rect=cv2.boundingRect(max_contour)
        x,y,w,h=rect
        center_coord=(x+int(w/2),y+h+region_upper-20)
        cv2.circle(img,center_coord,5,(0,255,0),-1)
        # return center_coord
        return center_coord
    # 返回点的位置信息
    # return color_mask
    else:
        return False
def goal_detect(img,body_position):
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.6)
    # 判断小人在左边还是右边
    # 小人在左,方块在右
    if body_position[0]<(img.shape[1]/2.0):
        region_left=body_position[0]+30
        region_right=img.shape[1]-30
    else:
        region_left=30
        region_right=body_position[0]-30
    region = img[region_upper:region_lower,region_left:region_right]
    # cv2.imshow("test1",region)

    edge_list = [0, 0, 0, 0]
    for i in range(3):
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, i]
        # 获取三个通道

        # region_gray=cv2.equalizeHist(region_gray)
        # 全都进行轮廓检测,存入数组里
        edge_list[i] = cv2.Canny(region_gray, 100, 160)
        # cv2.imshow(str(i),edge_list[i])
    # 原格子图转换为灰度图
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # region_gray = cv2.equalizeHist(region_gray)
    # egion_gray = cv2.GaussianBlur(region_gray, (5, 5), 0)
    # cv2.imshow("gray",region_gray)
    # 设置算子的大小5*5
    edge_list[3] = cv2.Canny(region_gray, 100, 160, apertureSize=5)
    # 大于max阈值则为边界，通过梯度进行判断
    edge_list[1] = np.bitwise_or(edge_list[0], edge_list[1])
    edge_list[2] = np.bitwise_or(edge_list[2], edge_list[1])
    edge_final = np.bitwise_or(edge_list[3], edge_list[2])

    # cv2.imshow('edge', edge_final)

    contours = cv2.findContours(edge_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    # 寻找轮廓
    # max_contour = max(contours, key=cv2.contourArea)
    # max_contour = cv2.convexHull(max_contour)

    y = 99999
    for contour in contours:
        # 将轮廓进行排序
        # lambda函数
        sorted_top = sorted(contour, key=lambda contour: contour[0][1])
        # print(sorted_top)
        if sorted_top[0][0][1] < y:
            raw_x = x = sorted_top[0][0][0]
            raw_y = y = sorted_top[0][0][1]

    # 确定第一个白点的位置
    print((int(x + region_left), int(y + region_upper)))
    # 在方块定义一个全零的矩阵
    # 如果是对于完整图像都要使用，则掩码层大小为原图行数 + 2，列数 + 2.
    # 是一个二维的0矩阵，边缘一圈会在使用算法是置为1。而只有对于掩码层上对应为0的位置才能泛洪，所以掩码层初始化为0矩阵
    # seed：为泛洪算法的种子点，也是根据该点的像素判断决定和其相近颜色的像素点，是否被泛洪处理
    mask = np.zeros((region_lower - region_upper + 2, region_right - region_left + 2), np.uint8)
    mask = cv2.floodFill(region, mask, (raw_x, raw_y + 16), [0, 100, 0])[2]
    # 感兴趣部分填1
    # print(mask)
    cv2.circle(img, (int(x + region_left), int(y + region_upper)), 5, (255, 0, 5), -1)
    # cv2.imshow("region",region)
    # 画第一个白点
    #
    M = cv2.moments(mask)
    # 计算中心距离
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    print("yuan",[x,y])
    if y < raw_y or abs(x - raw_x) > 40:
        x = raw_x
        y = raw_y
        y += region_upper
        x += region_left
        y = (-abs(x - body_position[0]) / pow(3, 0.5) + body_position[1])
        print("TESTTEST")

    # cv2.imshow("test",mask)
    # cv2.imshow("edge", edge_final)
    else:
        y += region_upper
        x += region_left

    # y = (-abs(x-body_position[0])/pow(3,0.5)+body_position[1])
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    # cv2.imshow('dealed', img)
    plt_show0(img)
    return [x, y]

def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))    # 随机防 ban
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top



def getdistance(playerpos,cubepos):
    distance = (playerpos[0] - cubepos[0]) ** 2 + (playerpos[1] - cubepos[1]) ** 2
    distance = distance ** 0.5
    return distance

def jump(distance):
    global press_coefficient
    set_button_position([900,1600])
    # press_time = distance * press_coefficient
    press_time = distance * 2.26
    press_time = max(press_time, 200)   # 设置 200ms 是最小的按压时间
    press_time = int(press_time)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    print(cmd)
    os.system(cmd)

def get_screenshot(id):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(id))
    os.system('adb pull /sdcard/%s.png .' % str(id))

if __name__=='__main__':
    while(1):
        get_screenshot(0)
        img=cv2.imread('%s.png' % 0)
        # img=cv2.imread('1.jpg')
        # 调整图片大小
        img=cv2.resize(img,(720,1080))
        print(img.shape)
        # cv2.imshow('test',img)
        PlayPos=self_detect(img)
        print("棋子的点",PlayPos)
        CubePos=goal_detect(img,PlayPos)
        print("方块的点",CubePos)
        distance=getdistance(PlayPos,CubePos)
        print(distance)
        jump(distance)
        # print(img.shape[1],img.shape[0])
        # cv2.waitKey(0)
        time.sleep(0.1)
        # 给与截图判断的时间