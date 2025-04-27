import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_process(bw, Img, show=False):
    """
    Hough变换处理，尽量仿MATLAB
    :param bw: 二值图像（0/1或0/255）
    :param Img: 原始图像
    :param show: 是否显示处理结果
    :return: 线段长度列表，线段坐标列表，最长线段坐标
    """
    # 确保二值图像为uint8类型且0/255
    if bw.max() <= 1:
        bw = (bw * 255).astype(np.uint8)
    else:
        bw = bw.astype(np.uint8)

    # OpenCV的HoughLinesP参数
    lines = cv2.HoughLinesP(
        bw, 1, np.pi/180, threshold=80, minLineLength=7, maxLineGap=50
    )

    Len = []
    XY = []
    max_len = 0
    xy_long = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xy = np.array([[x1, y1], [x2, y2]])
            length = np.linalg.norm(xy[0] - xy[1])
            Len.append(length)
            XY.append(xy)
            if length > max_len:
                max_len = length
                xy_long = xy

        # 按长度排序，取前4条最长的线
        indices = np.argsort(Len)[::-1][:10]  # 先多取几条，便于去重
        Len = np.array(Len)[indices]
        XYn_raw = [XY[i] for i in indices]

        # 去重逻辑：端点都很接近的线段只保留一条
        def is_similar(line1, line2, thresh=10):
            # 判断两条线段的两个端点是否都很接近
            return (np.linalg.norm(line1[0] - line2[0]) < thresh and np.linalg.norm(line1[1] - line2[1]) < thresh) or \
                   (np.linalg.norm(line1[0] - line2[1]) < thresh and np.linalg.norm(line1[1] - line2[0]) < thresh)
        XYn = []
        for xy in XYn_raw:
            if not any(is_similar(xy, f) for f in XYn):
                XYn.append(xy)
            if len(XYn) >= 4:
                break
        if len(XYn) == 0:
            XYn = XYn_raw[:1]
        xy_long = XYn[0]
    else:
        Len = []
        XYn = []
        xy_long = None

    # 调整最长线段（仿MATLAB）
    if xy_long is not None:
        x = xy_long[:, 0]
        y = xy_long[:, 1]
        if abs(x[1] - x[0]) < abs(y[1] - y[0]):
            x = np.array([np.mean(x), np.mean(x)])
        else:
            y = np.array([0.7*y[0] + 0.3*y[1], 0.3*y[0] + 0.7*y[1]])
        xy_long = np.column_stack((x, y))

    if show:
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(bw, cmap='gray')
        plt.title('二值图像', fontweight='bold')
        plt.subplot(2, 2, 2)
        edges = cv2.Canny(bw, 50, 150, apertureSize=3)
        plt.imshow(edges, cmap='gray')
        plt.title('边缘检测', fontweight='bold')
        plt.subplot(2, 2, 3)
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        plt.title('原始图像', fontweight='bold')
        plt.subplot(2, 2, 4)
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        plt.title('直线检测图像', fontweight='bold')
        # 绘制检测到的前4条直线
        if lines is not None:
            for xy in XYn:
                plt.plot(xy[:, 0], xy[:, 1], 'g-', linewidth=2)
        # 绘制最长线段
        if xy_long is not None:
            plt.plot(xy_long[:, 0], xy_long[:, 1], 'r-', linewidth=3)
        plt.show(block=False)

    return Len, XYn, xy_long 