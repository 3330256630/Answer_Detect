import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def morph_process(bw1, show=False, min_area_ratio=0.00005):
    """
    形态学处理+去除小连通域
    :param bw1: 输入二值图像
    :param show: 是否显示处理结果
    :param min_area_ratio: 小区域面积比例（默认0.0005）
    :return: 处理后的二值图像和位置信息
    """
    selem = np.ones((2,2))
    # 形态学操作
    bw2 = ndimage.binary_opening(bw1, structure=selem)
    bw2 = ndimage.binary_fill_holes(bw2)

    # 去除小连通域
    label_im, num = ndimage.label(bw2)
    sizes = ndimage.sum(bw2, label_im, range(num + 1))
    min_area = int(bw2.size * min_area_ratio)
    mask_size = sizes < min_area
    remove_pixel = mask_size[label_im]
    bw2[remove_pixel] = 0

    # 列和阈值
    bws = np.sum(bw2, axis=0)
    threshold = round(np.sum(bw2) * 0.007)
    inds = np.where(bws > threshold)[0]

    Loc = None
    if len(inds) > 0:
        Loc = inds[0] - 5
        bw2[:, Loc:] = 0

    # 闭运算和填充
    bw2 = ndimage.binary_closing(bw2, structure=selem)
    bw2 = ndimage.binary_fill_holes(bw2)

    # 再次去除小连通域
    label_im, num = ndimage.label(bw2)
    sizes = ndimage.sum(bw2, label_im, range(num + 1))
    mask_size = sizes < min_area
    remove_pixel = mask_size[label_im]
    bw2[remove_pixel] = 0

    if show:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(bw1, cmap='gray')
        plt.title('输入图像', fontweight='bold')
        plt.subplot(1, 2, 2)
        plt.imshow(bw2, cmap='gray')
        plt.title('滤波+去小区域', fontweight='bold')
        plt.show(block=False)

    return bw2, Loc