import cv2
import numpy as np
import matplotlib.pyplot as plt
# 假设这里有set_chinese_font函数的实现，这里简单注释掉
# from font_setting import set_chinese_font

def image_rotate(I, bw, angle, show=False):
    """
    图像旋转处理
    :param I: 原始图像
    :param bw: 二值图像
    :param angle: 旋转角度
    :param show: 是否显示处理结果
    :return: 旋转后的原始图像和二值图像
    """
    # 计算旋转矩阵
    height, width = I.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -90-angle, 1.0)

    # 计算旋转后图像的新尺寸
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵以适应新的尺寸
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 旋转图像
    I1 = cv2.warpAffine(I, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
    bw1 = cv2.warpAffine(bw.astype(np.uint8), rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    if show:
        # 设置中文字体
        # set_chinese_font()

        plt.figure(figsize=(15, 15))

        # 显示原始图像
        plt.subplot(2, 2, 1)
        if len(I.shape) == 3:
            plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I, cmap='gray')
        plt.title('原始图像', fontweight='bold')

        # 显示原始二值图像
        plt.subplot(2, 2, 3)
        plt.imshow(bw, cmap='gray')
        plt.title('原始二值图像', fontweight='bold')

        # 显示旋转后的图像
        plt.subplot(2, 2, 2)
        if len(I1.shape) == 3:
            plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(I1, cmap='gray')
        plt.title('校正图像', fontweight='bold')

        # 显示旋转后的二值图像
        plt.subplot(2, 2, 4)
        plt.imshow(bw1, cmap='gray')
        plt.title('校正二值图像', fontweight='bold')

        plt.show(block=False)

    return I1, bw1
