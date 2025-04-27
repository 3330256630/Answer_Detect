import cv2
import numpy as np
import matplotlib.pyplot as plt
from font_setting import set_chinese_font

def image_smooth(I1, hsize=(3, 3), sigma=0.5, show=False):
    """
    图像平滑处理
    :param I1: 输入图像
    :param hsize: 高斯核大小，默认为(3, 3)
    :param sigma: 高斯核标准差，默认为0.5
    :param show: 是否显示处理结果
    :return: 平滑后的图像
    """
    # 使用高斯滤波进行平滑
    I2 = cv2.GaussianBlur(I1, hsize, sigma)
    
    if show:
        # 设置中文字体
        set_chinese_font()
        
        plt.figure(figsize=(10, 10))
        
        # 显示平滑前的图像
        plt.subplot(2, 1, 1)
        if len(I1.shape) == 3:
            plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I1, cmap='gray')
        plt.title('平滑前图像', fontweight='bold')
        
        # 显示平滑后的图像
        plt.subplot(2, 1, 2)
        if len(I2.shape) == 3:
            plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I2, cmap='gray')
        plt.title('平滑后图像', fontweight='bold')
        
        plt.show(block=False)
    
    return I2 