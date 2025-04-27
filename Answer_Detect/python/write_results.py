import os
import matplotlib.pyplot as plt

def write_results():
    """
    保存所有打开的图形窗口到results目录
    """
    # 创建results目录（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 获取所有打开的图形窗口
    figures = [plt.figure(i) for i in plt.get_fignums()]
    
    # 保存每个图形窗口
    for i, fig in enumerate(figures, 1):
        # 保存图像
        fig.savefig(fr'D:\360MoveData\Users\PC\Desktop\test1\results\result{i}.png')
