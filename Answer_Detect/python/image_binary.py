import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from font_setting import set_chinese_font

def image_binary(I, show=False, thresh_scale=0.8, use_equalize=False):
    """
    图像二值化处理（仿MATLAB im2bw+graythresh+反色），可调节阈值
    :param I: 输入图像
    :param show: 是否显示处理结果
    :param thresh_scale: 阈值缩放因子，默认0.9
    :param use_equalize: 是否先做直方图均衡化
    :return: 二值化图像
    """
    # 转为灰度，归一化到0~1
    if len(I.shape) == 3:
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        gray = I
    gray = gray.astype(np.float32) / 255.0

    # 可选：直方图均衡化
    if use_equalize:
        gray = cv2.equalizeHist((gray * 255).astype(np.uint8)) / 255.0

    # Otsu阈值并手动降低
    thresh = threshold_otsu(gray) * thresh_scale
    bw1 = (gray > thresh).astype(np.uint8)
    bw2 = 1 - bw1

    if show:
        set_chinese_font()
        plt.figure(figsize=(18, 6), facecolor='#dddddd')
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray', vmin=0, vmax=1)
        plt.title('待处理图像', fontweight='bold')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(bw1 * 255, cmap='gray', vmin=0, vmax=255)
        plt.title('二值化图像', fontweight='bold')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(bw2 * 255, cmap='gray', vmin=0, vmax=255)
        plt.title('二值化反色图像', fontweight='bold')
        plt.axis('off')
        plt.show(block=False)

    return bw2 