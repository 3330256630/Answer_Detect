import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_normalize import image_normalize
from image_smooth import image_smooth
from gray_convert import gray_convert
from image_binary import image_binary
from hough_process import hough_process
from compute_angle import compute_angle
from image_rotate import image_rotate
from morph_process import morph_process
from region_segmation import region_segmation
from location_label import location_label
from analysis import analysis
from write_results import write_results

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    # 读取图像
    I = cv2.imread(r'D:\360MoveData\Users\PC\Desktop\test1\images\1.jpg')

    # 图像归一化

    # 2. 归一化
    I1 = image_normalize(I, show=False)

    # 3. 平滑
    I2 = image_smooth(I1, hsize=(3, 3), sigma=0.5, show=False)

    # 4. 灰度
    I3 = gray_convert(I2, show=False)

    # 5. 二值化
    bw2 = image_binary(I3, show=False)

    # 6. Hough检测
    _, _, xy_long = hough_process(bw2, I, show=False)

    # 7. 角度
    angle = compute_angle(xy_long)

    # 8. 旋转
    I4, bw3 = image_rotate(I1, bw2, angle * 1.8, show=False)

    # 9. 形态学处理
    bw4, Loc1 = morph_process(bw3, show=False)

    # 10. 再次Hough检测
    Len, XYn, xy_long = hough_process(bw4, I, show=False)

    # 11. 区域分割
    bw5, bw6 = region_segmation(XYn, bw4, I4, show=False)

    # 12. 连通域分析
    stats1, stats2, Line = location_label(bw5, bw6, I4, XYn, Loc1, show=False)

    # 13. 分析
    Dom, Aom, Answer, Bn = analysis(stats1, stats2, Line, I4, show=True)

    # 14. 保存结果
    write_results()

if __name__ == "__main__":
    main()