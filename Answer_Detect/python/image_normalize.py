import cv2
import numpy as np
import matplotlib.pyplot as plt
from font_setting import set_chinese_font

def image_normalize(I, show=False):
    """
    图像归一化处理
    :param I: 输入图像
    :param show: 是否显示处理结果
    :return: 归一化后的图像
    """
    # 检查图像大小
    if I.shape[0] > 2000:
        # 使用双线性插值进行缩放
        I = cv2.resize(I, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    
    # 确保输入图像是8位无符号整数类型
    if I.dtype != np.uint8:
        I = I.astype(np.uint8)
    
    # 使用OpenCV的normalize函数进行归一化
    I1 = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX)
    
    if show:
        # 设置中文字体
        set_chinese_font()
        
        plt.figure(figsize=(10, 10))
        
        # 显示原始图像
        plt.subplot(2, 1, 1)
        if len(I.shape) == 3:
            plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I, cmap='gray')
        plt.title('原始图像', fontweight='bold')
        
        # 显示归一化图像
        plt.subplot(2, 1, 2)
        if len(I1.shape) == 3:
            plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I1, cmap='gray')
        plt.title('归一化图像', fontweight='bold')
        
        plt.show(block=False)
    
    return I1 