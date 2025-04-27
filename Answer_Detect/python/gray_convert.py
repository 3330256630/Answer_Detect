import cv2
import numpy as np
import matplotlib.pyplot as plt
from font_setting import set_chinese_font

def gray_convert(I, show=False):
    """
    将图像转换为灰度图像
    :param I: 输入图像
    :param show: 是否显示处理结果
    :return: 灰度图像
    """
    # 检查图像维度
    if len(I.shape) == 3:
        # 转换为灰度图像
        I1 = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        I1 = I
    
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
        plt.title('RGB图像', fontweight='bold')
        
        # 显示灰度图像
        plt.subplot(2, 1, 2)
        plt.imshow(I1, cmap='gray')
        plt.title('灰度图像', fontweight='bold')
        
        plt.show(block=False)
    
    return I1 