import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def location_label(bw1, bw2, Img, XYn, Loc1, show=False):
    """
    位置标记
    :param bw1: 第一个二值图像
    :param bw2: 第二个二值图像
    :param Img: 原始图像
    :param XYn: 线段坐标列表
    :param Loc1: 右边界位置信息
    :param show: 是否显示处理结果
    :return: 区域属性列表1，区域属性列表2，线段列表
    """
    # 标记连通区域
    L1, num1 = ndimage.label(bw1)
    stats1 = []
    for i in range(1, num1 + 1):
        mask = (L1 == i)
        props = {
            'Centroid': ndimage.center_of_mass(mask),
            'Area': np.sum(mask)
        }
        stats1.append(props)
    
    L2, num2 = ndimage.label(bw2)
    stats2 = []
    for i in range(1, num2 + 1):
        mask = (L2 == i)
        props = {
            'Centroid': ndimage.center_of_mass(mask),
            'Area': np.sum(mask)
        }
        stats2.append(props)
    
    # 处理线段
    Line1 = XYn[0]
    Line2 = XYn[1]
    
    # 根据平均y坐标调整线段顺序
    if np.mean(Line2[:, 1]) < np.mean(Line1[:, 1]):
        Line1 = XYn[1]
        Line2 = XYn[0]
    
    # 找到最小列坐标
    r1, c1 = np.where(bw1)
    r2, c2 = np.where(bw2)
    Loc2 = min([min(c1), min(c2)]) - 5
    
    # 创建水平线段
    Line1 = np.array([[1, np.mean(Line1[:, 1])], [Img.shape[1], np.mean(Line1[:, 1])]])
    Line2 = np.array([[1, np.mean(Line2[:, 1])], [Img.shape[1], np.mean(Line2[:, 1])]])
    
    # 创建垂直线段
    Line3 = np.array([[Loc2, 1], [Loc2, Img.shape[0]]])
    Line4 = np.array([[Loc1, 1], [Loc1, Img.shape[0]]])
    
    # 存储所有线段
    Line = [Line1, Line2, Line3, Line4]
    
    if show:
        plt.figure(figsize=(15, 15))
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        plt.title('标记图像', fontweight='bold')
        
        # 绘制质心
        for i in range(num1):
            temp = stats1[i]['Centroid']
            plt.plot(temp[1], temp[0], 'r.')
        
        plt.show(block=False)
    
    return stats1, stats2, Line 