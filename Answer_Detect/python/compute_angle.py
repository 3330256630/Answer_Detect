import numpy as np

def compute_angle(xy_long):
    """
    计算线段的角度
    :param xy_long: 线段坐标，格式为[[x1, y1], [x2, y2]]
    :return: 角度（度）
    """
    x1 = xy_long[:, 0]
    y1 = xy_long[:, 1]
    
    # 计算斜率
    K1 = -(y1[1] - y1[0]) / (x1[1] - x1[0])
    
    # 计算角度（弧度转角度）
    angle = np.arctan(K1) * 180 / np.pi
    
    return angle 