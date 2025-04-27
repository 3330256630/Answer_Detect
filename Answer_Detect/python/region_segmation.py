import cv2
import numpy as np
import matplotlib.pyplot as plt
def region_segmation(XY, bw, Img, show=False):
    y1 = np.mean(XY[0][:, 1])
    y2 = np.mean(XY[1][:, 1])
    minr = int(min(y1, y2))
    maxr = int(max(y1, y2))

    bw1 = bw.copy()
    bw2 = bw.copy()
    bw1[:minr + 5, :] = 0
    bw1[maxr - 5:, :] = 0
    bw2[minr - 5:, :] = 0
    bw2[:round(minr * 0.5), :] = 0

    if show:
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        plt.title('原图像', fontweight='bold')
        plt.subplot(2, 2, 2)
        plt.imshow(bw, cmap='gray')
        plt.title('原二值图像', fontweight='bold')
        for i in range(2):
            xy = XY[i]
            plt.plot(xy[:, 0], xy[:, 1], 'r-', linewidth=2)
        plt.subplot(2, 2, 3)
        plt.imshow(bw1, cmap='gray')
        plt.title('下半部分图像', fontweight='bold')
        plt.subplot(2, 2, 4)
        plt.imshow(bw2, cmap='gray')
        plt.title('上半部分图像', fontweight='bold')
        plt.show(block=False)

    return bw1, bw2