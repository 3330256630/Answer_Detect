import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter import Tk, messagebox
import pandas as pd

def save_to_excel(Answer, Bn, filename='analysis_results.xlsx'):
    # Create a DataFrame for the answers
    answer_data = []
    for ans in Answer:
        if ans['aw']:  # Only include answers that have been marked
            answer_data.append({
                '题号': ans['no'],
                '选项': ', '.join(ans['aw'])
            })
    
    # Create a DataFrame for the special areas (Bn)
    special_data = []
    if Bn[0]['result']:  # 准考证
        special_data.append({'区域': '试卷类型', '结果': Bn[0]['result'][0]})
    if Bn[1]['result']:  # 准考证号
        special_data.append({'区域': '准考证号', '结果': ''.join(map(str, Bn[1]['result']))})
    if Bn[2]['result']:  # 科目
        special_data.append({'区域': '科目', '结果': Bn[2]['result'][0]})
    
    # Create Excel writer
    with pd.ExcelWriter(filename) as writer:
        # Save answers to first sheet
        pd.DataFrame(answer_data).to_excel(writer, sheet_name='答题结果', index=False)
        # Save special areas to second sheet
        pd.DataFrame(special_data).to_excel(writer, sheet_name='特殊区域', index=False)
    
    return filename

def analysis(stats1, stats2, Line, Img, show=True):
    # 提取四条分割线
    Line1 = Line[0]
    Line2 = Line[1]
    Line3 = Line[2]
    Line4 = Line[3]

    # 计算主水平分割线（用于分割答题区的三大块）
    yn1 = round(Line1[0, 1] + 0.18 * (Line2[0, 1] - Line1[0, 1]))
    yn2 = round(Line1[0, 1] + 0.34 * (Line2[0, 1] - Line1[0, 1]))
    yn3 = round(Line1[0, 1] + 0.50 * (Line2[0, 1] - Line1[0, 1]))
    Linen1_1 = np.array([[Line1[0, 0], yn1], [Line1[1, 0], yn1]])
    Linen2_1 = np.array([[Line1[0, 0], yn2], [Line1[1, 0], yn2]])
    Linen3_1 = np.array([[Line1[0, 0], yn3], [Line1[1, 0], yn3]])

    # 计算主竖直分割线（用于分割答题区的四大列）
    xn1 = round(Line3[0, 0] + 0.22 * (Line4[0, 0] - Line3[0, 0]))
    xn2 = round(Line3[0, 0] + 0.26 * (Line4[0, 0] - Line3[0, 0]))
    xn3 = round(Line3[0, 0] + 0.48 * (Line4[0, 0] - Line3[0, 0]))
    xn4 = round(Line3[0, 0] + 0.52 * (Line4[0, 0] - Line3[0, 0]))
    xn5 = round(Line3[0, 0] + 0.73 * (Line4[0, 0] - Line3[0, 0]))
    xn6 = round(Line3[0, 0] + 0.77 * (Line4[0, 0] - Line3[0, 0]))
    xn7 = round(Line3[0, 0] + 0.98 * (Line4[0, 0] - Line3[0, 0]))

    Linen1_2 = np.array([[xn1, Line3[0, 1]], [xn1, Line3[1, 1]]])
    Linen2_2 = np.array([[xn2, Line3[0, 1]], [xn2, Line3[1, 1]]])
    Linen3_2 = np.array([[xn3, Line3[0, 1]], [xn3, Line3[1, 1]]])
    Linen4_2 = np.array([[xn4, Line3[0, 1]], [xn4, Line3[1, 1]]])
    Linen5_2 = np.array([[xn5, Line3[0, 1]], [xn5, Line3[1, 1]]])
    Linen6_2 = np.array([[xn6, Line3[0, 1]], [xn6, Line3[1, 1]]])
    Linen7_2 = np.array([[xn7, Line3[0, 1]], [xn7, Line3[1, 1]]])

    # 计算每个大块内的细分横线（用于分割每行题目）
    ym1_1 = round(Line1[0, 1] + 0.32 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym2_1 = round(Line1[0, 1] + 0.5 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym3_1 = round(Line1[0, 1] + 0.65 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym4_1 = round(Line1[0, 1] + 0.80 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym5_1 = round(Line1[0, 1] + 0.95 * (Linen1_1[0, 1] - Line1[0, 1]))
    Linem1_1 = np.array([[Line1[0, 0], ym1_1], [Line1[1, 0], ym1_1]])
    Linem2_1 = np.array([[Line1[0, 0], ym2_1], [Line1[1, 0], ym2_1]])
    Linem3_1 = np.array([[Line1[0, 0], ym3_1], [Line1[1, 0], ym3_1]])
    Linem4_1 = np.array([[Line1[0, 0], ym4_1], [Line1[1, 0], ym4_1]])
    Linem5_1 = np.array([[Line1[0, 0], ym5_1], [Line1[1, 0], ym5_1]])

    ym1_2 = round(Linen1_1[0, 1] + 0.25 * (Linen2_1[0, 1] - Linen1_1[0, 1]))
    ym2_2 = round(Linen1_1[0, 1] + 0.40 * (Linen2_1[0, 1] - Linen1_1[0, 1]))
    ym3_2 = round(Linen1_1[0, 1] + 0.60 * (Linen2_1[0, 1] - Linen1_1[0, 1]))
    ym4_2 = round(Linen1_1[0, 1] + 0.75 * (Linen2_1[0, 1] - Linen1_1[0, 1]))
    ym5_2 = round(Linen1_1[0, 1] + 0.90 * (Linen2_1[0, 1] - Linen1_1[0, 1]))
    Linem1_2 = np.array([[Line1[0, 0], ym1_2], [Line1[1, 0], ym1_2]])
    Linem2_2 = np.array([[Line1[0, 0], ym2_2], [Line1[1, 0], ym2_2]])
    Linem3_2 = np.array([[Line1[0, 0], ym3_2], [Line1[1, 0], ym3_2]])
    Linem4_2 = np.array([[Line1[0, 0], ym4_2], [Line1[1, 0], ym4_2]])
    Linem5_2 = np.array([[Line1[0, 0], ym5_2], [Line1[1, 0], ym5_2]])

    ym1_3 = round(Linen2_1[0, 1] + 0.25 * (Linen3_1[0, 1] - Linen2_1[0, 1]))
    ym2_3 = round(Linen2_1[0, 1] + 0.40 * (Linen3_1[0, 1] - Linen2_1[0, 1]))
    ym3_3 = round(Linen2_1[0, 1] + 0.60 * (Linen3_1[0, 1] - Linen2_1[0, 1]))
    ym4_3 = round(Linen2_1[0, 1] + 0.75 * (Linen3_1[0, 1] - Linen2_1[0, 1]))
    ym5_3 = round(Linen2_1[0, 1] + 0.90 * (Linen3_1[0, 1] - Linen2_1[0, 1]))
    Linem1_3 = np.array([[Line1[0, 0], ym1_3], [Line1[1, 0], ym1_3]])
    Linem2_3 = np.array([[Line1[0, 0], ym2_3], [Line1[1, 0], ym2_3]])
    Linem3_3 = np.array([[Line1[0, 0], ym3_3], [Line1[1, 0], ym3_3]])
    Linem4_3 = np.array([[Line1[0, 0], ym4_3], [Line1[1, 0], ym4_3]])
    Linem5_3 = np.array([[Line1[0, 0], ym5_3], [Line1[1, 0], ym5_3]])

    # Fine vertical lines
    xm1_1 = round(Line3[0, 0] + 0.07 * (Linen1_2[0, 0] - Line3[0, 0]))
    xm1_2 = round(Line3[0, 0] + 0.25 * (Linen1_2[0, 0] - Line3[0, 0]))
    xm1_3 = round(Line3[0, 0] + 0.43 * (Linen1_2[0, 0] - Line3[0, 0]))
    xm1_4 = round(Line3[0, 0] + 0.63 * (Linen1_2[0, 0] - Line3[0, 0]))
    xm1_5 = round(Line3[0, 0] + 0.83 * (Linen1_2[0, 0] - Line3[0, 0]))
    xm1_6 = round(Line3[0, 0] + 1.02 * (Linen1_2[0, 0] - Line3[0, 0]))

    Linem1_1_2 = np.array([[xm1_1, Line3[0, 1]], [xm1_1, Line3[1, 1]]])
    Linem1_2_2 = np.array([[xm1_2, Line3[0, 1]], [xm1_2, Line3[1, 1]]])
    Linem1_3_2 = np.array([[xm1_3, Line3[0, 1]], [xm1_3, Line3[1, 1]]])
    Linem1_4_2 = np.array([[xm1_4, Line3[0, 1]], [xm1_4, Line3[1, 1]]])
    Linem1_5_2 = np.array([[xm1_5, Line3[0, 1]], [xm1_5, Line3[1, 1]]])
    Linem1_6_2 = np.array([[xm1_6, Line3[0, 1]], [xm1_6, Line3[1, 1]]])

    xm2_1 = round(Linen2_2[0, 0] + 0.05 * (Linen3_2[0, 0] - Linen2_2[0, 0]))
    xm2_2 = round(Linen2_2[0, 0] + 0.22 * (Linen3_2[0, 0] - Linen2_2[0, 0]))
    xm2_3 = round(Linen2_2[0, 0] + 0.41 * (Linen3_2[0, 0] - Linen2_2[0, 0]))
    xm2_4 = round(Linen2_2[0, 0] + 0.58 * (Linen3_2[0, 0] - Linen2_2[0, 0]))
    xm2_5 = round(Linen2_2[0, 0] + 0.78 * (Linen3_2[0, 0] - Linen2_2[0, 0]))
    xm2_6 = round(Linen2_2[0, 0] + 0.98 * (Linen3_2[0, 0] - Linen2_2[0, 0]))

    Linem2_1_2 = np.array([[xm2_1, Line3[0, 1]], [xm2_1, Line3[1, 1]]])
    Linem2_2_2 = np.array([[xm2_2, Line3[0, 1]], [xm2_2, Line3[1, 1]]])
    Linem2_3_2 = np.array([[xm2_3, Line3[0, 1]], [xm2_3, Line3[1, 1]]])
    Linem2_4_2 = np.array([[xm2_4, Line3[0, 1]], [xm2_4, Line3[1, 1]]])
    Linem2_5_2 = np.array([[xm2_5, Line3[0, 1]], [xm2_5, Line3[1, 1]]])
    Linem2_6_2 = np.array([[xm2_6, Line3[0, 1]], [xm2_6, Line3[1, 1]]])

    xm3_1 = round(Linen4_2[0, 0] + 0.03 * (Linen5_2[0, 0] - Linen4_2[0, 0]))
    xm3_2 = round(Linen4_2[0, 0] + 0.22 * (Linen5_2[0, 0] - Linen4_2[0, 0]))
    xm3_3 = round(Linen4_2[0, 0] + 0.41 * (Linen5_2[0, 0] - Linen4_2[0, 0]))
    xm3_4 = round(Linen4_2[0, 0] + 0.58 * (Linen5_2[0, 0] - Linen4_2[0, 0]))
    xm3_5 = round(Linen4_2[0, 0] + 0.78 * (Linen5_2[0, 0] - Linen4_2[0, 0]))
    xm3_6 = round(Linen4_2[0, 0] + 0.98 * (Linen5_2[0, 0] - Linen4_2[0, 0]))

    Linem3_1_2 = np.array([[xm3_1, Line3[0, 1]], [xm3_1, Line3[1, 1]]])
    Linem3_2_2 = np.array([[xm3_2, Line3[0, 1]], [xm3_2, Line3[1, 1]]])
    Linem3_3_2 = np.array([[xm3_3, Line3[0, 1]], [xm3_3, Line3[1, 1]]])
    Linem3_4_2 = np.array([[xm3_4, Line3[0, 1]], [xm3_4, Line3[1, 1]]])
    Linem3_5_2 = np.array([[xm3_5, Line3[0, 1]], [xm3_5, Line3[1, 1]]])
    Linem3_6_2 = np.array([[xm3_6, Line3[0, 1]], [xm3_6, Line3[1, 1]]])

    xm4_1 = round(Linen6_2[0, 0] + 0.03 * (Linen7_2[0, 0] - Linen6_2[0, 0]))
    xm4_2 = round(Linen6_2[0, 0] + 0.22 * (Linen7_2[0, 0] - Linen6_2[0, 0]))
    xm4_3 = round(Linen6_2[0, 0] + 0.41 * (Linen7_2[0, 0] - Linen6_2[0, 0]))
    xm4_4 = round(Linen6_2[0, 0] + 0.58 * (Linen7_2[0, 0] - Linen6_2[0, 0]))
    xm4_5 = round(Linen6_2[0, 0] + 0.78 * (Linen7_2[0, 0] - Linen6_2[0, 0]))
    xm4_6 = round(Linen6_2[0, 0] + 0.98 * (Linen7_2[0, 0] - Linen6_2[0, 0]))

    Linem4_1_2 = np.array([[xm4_1, Line3[0, 1]], [xm4_1, Line3[1, 1]]])
    Linem4_2_2 = np.array([[xm4_2, Line3[0, 1]], [xm4_2, Line3[1, 1]]])
    Linem4_3_2 = np.array([[xm4_3, Line3[0, 1]], [xm4_3, Line3[1, 1]]])
    Linem4_4_2 = np.array([[xm4_4, Line3[0, 1]], [xm4_4, Line3[1, 1]]])
    Linem4_5_2 = np.array([[xm4_5, Line3[0, 1]], [xm4_5, Line3[1, 1]]])
    Linem4_6_2 = np.array([[xm4_6, Line3[0, 1]], [xm4_6, Line3[1, 1]]])

    # Additional horizontal lines above
    ym1_4 = round(Line1[0, 1] - 0.18 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym2_4 = round(Line1[0, 1] - 0.35 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym3_4 = round(Line1[0, 1] - 0.50 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym4_4 = round(Line1[0, 1] - 0.65 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym5_4 = round(Line1[0, 1] - 0.80 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym6_4 = round(Line1[0, 1] - 0.95 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym7_4 = round(Line1[0, 1] - 1.10 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym8_4 = round(Line1[0, 1] - 1.22 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym9_4 = round(Line1[0, 1] - 1.35 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym10_4 = round(Line1[0, 1] - 1.50 * (Linen1_1[0, 1] - Line1[0, 1]))
    ym11_4 = round(Line1[0, 1] - 1.65 * (Linen1_1[0, 1] - Line1[0, 1]))

    Linem1_4 = np.array([[Line1[0, 0], ym1_4], [Line1[1, 0], ym1_4]])
    Linem2_4 = np.array([[Line1[0, 0], ym2_4], [Line1[1, 0], ym2_4]])
    Linem3_4 = np.array([[Line1[0, 0], ym3_4], [Line1[1, 0], ym3_4]])
    Linem4_4 = np.array([[Line1[0, 0], ym4_4], [Line1[1, 0], ym4_4]])
    Linem5_4 = np.array([[Line1[0, 0], ym5_4], [Line1[1, 0], ym5_4]])
    Linem6_4 = np.array([[Line1[0, 0], ym6_4], [Line1[1, 0], ym6_4]])
    Linem7_4 = np.array([[Line1[0, 0], ym7_4], [Line1[1, 0], ym7_4]])
    Linem8_4 = np.array([[Line1[0, 0], ym8_4], [Line1[1, 0], ym8_4]])
    Linem9_4 = np.array([[Line1[0, 0], ym9_4], [Line1[1, 0], ym9_4]])
    Linem10_4 = np.array([[Line1[0, 0], ym10_4], [Line1[1, 0], ym10_4]])
    Linem11_4 = np.array([[Line1[0, 0], ym11_4], [Line1[1, 0], ym11_4]])

    # 构建Dom结构，描述每个答题区块的分割线
    xt = [
        [xm1_1, xm1_2, xm1_3, xm1_4, xm1_5, xm1_6],
        [xm2_1, xm2_2, xm2_3, xm2_4, xm2_5, xm2_6],
        [xm3_1, xm3_2, xm3_3, xm3_4, xm3_5, xm3_6],
        [xm4_1, xm4_2, xm4_3, xm4_4, xm4_5, xm4_6]
    ]
    Dom = [
        {'Loc': [Line1[0, 1], Linen1_1[0, 1]], 'y': [ym1_1, ym2_1, ym3_1, ym4_1, ym5_1], 'x': xt},
        {'Loc': [Linen1_1[0, 1], Linen2_1[0, 1]], 'y': [ym1_2, ym2_2, ym3_2, ym4_2, ym5_2], 'x': xt},
        {'Loc': [Linen2_1[0, 1], Linen3_1[0, 1]], 'y': [ym1_3, ym2_3, ym3_3, ym4_3, ym5_3], 'x': xt}
    ]

    # 构建Aom结构，描述准考证、科目等特殊区域的分割线
    Aom = [
        {'Loc': [ym7_4, ym6_4], 'y': [ym7_4, ym6_4], 'x': [xm1_5, xm1_6]},
        {'Loc': [ym11_4, ym1_4], 'y': [ym11_4, ym10_4, ym9_4, ym8_4, ym7_4, ym6_4, ym5_4, ym4_4, ym3_4, ym2_4, ym1_4], 'x': [xm2_5, xm2_6, xm3_1, xm3_2, xm3_3, xm3_4, xm3_5, xm3_6, xm4_1, xm4_2]},
        {'Loc': [ym11_4, ym1_4], 'y': [ym11_4, ym10_4, ym9_4, ym8_4, ym7_4, ym6_4, ym5_4, ym4_4, ym3_4, ym2_4, ym1_4], 'x': [xm4_5, xm4_6]}
    ]

    # 初始化Answer结构，存储每道题的坐标、编号、选项
    aw = ['A', 'B', 'C', 'D']
    Answer = [{'Loc': [], 'no': i+1, 'aw': []} for i in range(60)]  # 假设60题

    # 遍历所有连通域质心，分配到对应题号和选项
    for stat in stats1:
        temp = stat['Centroid']
        for i1, dom in enumerate(Dom):
            Loc = dom['Loc']
            if Loc[0] <= temp[0] <= Loc[1]:
                x = dom['x']
                y = dom['y']
                i_y = i1 * 20  # 每块20题
                for i2, xt_ in enumerate(x):
                    for i3 in range(len(xt_) - 1):
                        if xt_[i3] <= temp[1] <= xt_[i3 + 1]:
                            i_x = i2 * 5 + i3
                i_n = i_y + i_x
                for i4 in range(len(y) - 1):
                    if y[i4] <= temp[0]<= y[i4 + 1]:
                        i_a = aw[i4]
                        break
                break
        Answer[i_n]['Loc'].append(temp)
        Answer[i_n]['no'] = i_n
        Answer[i_n]['aw'].append(i_a)


    # 构建Bn结构，存储准考证、科目等特殊区域的识别结果
    strs = ['政治', '语文', '数学', '物理', '化学', '外语', '历史', '地理', '生物']
    Bn = [{'result': [], 'Loc': []} for _ in range(3)]

    for stat in stats2:
        temp = stat['Centroid']
        # 判断是否在准考证区域
        if Aom[0]['x'][0] <= temp[1] <= Aom[0]['x'][1] and Aom[0]['y'][0] <= temp[0] <= Aom[0]['y'][1]:
            Bn[0]['Loc'] = [temp]
            Bn[0]['result'] = [1]
        # 判断是否在准考证号区域
        if Aom[1]['Loc'][0] <= temp[0] <= Aom[1]['Loc'][1]:
            for i1 in range(len(Aom[1]['x']) - 1):
                if Aom[1]['x'][i1] <= temp[1] <= Aom[1]['x'][i1 + 1]:
                    for i2 in range(len(Aom[1]['y']) - 1):
                        if Aom[1]['y'][i2] <= temp[1] <= Aom[1]['y'][i2 + 1]:
                            Bn[1]['Loc'].append(temp)
                            Bn[1]['result'].append(i2)
        # 判断是否在科目区域
        if Aom[2]['Loc'][0] <= temp[0] <= Aom[2]['Loc'][1] and Aom[2]['x'][0] <= temp[1] <= Aom[2]['x'][1]:
            for i1 in range(len(Aom[2]['y']) - 1):
                if Aom[2]['y'][i1] <= temp[0] <= Aom[2]['y'][i1 + 1]:
                    Bn[2]['Loc'].append(temp)
                    Bn[2]['result'].append(strs[i1])

    # 可视化部分
    if show:
        # 第一张图：画所有分割线
        plt.figure(figsize=(12, 12))
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        red_lines = [Linem1_1, Linem2_1, Linem3_1, Linem4_1, Linem5_1,
                     Linem1_2, Linem2_2, Linem3_2, Linem4_2, Linem5_2,
                     Linem1_3, Linem2_3, Linem3_3, Linem4_3, Linem5_3,
                     Linem1_1_2, Linem1_2_2, Linem1_3_2, Linem1_4_2, Linem1_5_2, Linem1_6_2,
                     Linem2_1_2, Linem2_2_2, Linem2_3_2, Linem2_4_2, Linem2_5_2, Linem2_6_2,
                     Linem3_1_2, Linem3_2_2, Linem3_3_2, Linem3_4_2, Linem3_5_2, Linem3_6_2,
                     Linem4_1_2, Linem4_2_2, Linem4_3_2, Linem4_4_2, Linem4_5_2, Linem4_6_2]
        blue_lines = [Linem1_4, Linem2_4, Linem3_4, Linem4_4, Linem5_4, Linem6_4, Linem7_4, Linem8_4, Linem9_4, Linem10_4, Linem11_4]
        for line in red_lines:
            plt.plot(line[:, 0], line[:, 1], 'r-', linewidth=1)
        for line in blue_lines:
            plt.plot(line[:, 0], line[:, 1], 'b-', linewidth=1)
        plt.title('网格线生成', fontweight='bold')
        plt.show()

        # 第二张图：画答题结果和特殊区域标记
        plt.figure(figsize=(12, 12))
        if len(Img.shape) == 3:
            plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(Img, cmap='gray')
        # 标记所有答题点的选项
        for ans in Answer:
            if ans['Loc']:
                for loc, aw in zip(ans['Loc'], ans['aw']):
                    plt.text(loc[1], loc[0], aw, color='b')
        Err = [0, 0, 0]
        # 标记准考证、科目等特殊区域
        if Bn[0]['Loc']:
            for loc, res in zip(Bn[0]['Loc'], Bn[0]['result']):
                plt.text(loc[1], loc[0], str(res), color='b')
        else:
            Err[0] = 1
        if Bn[1]['Loc']:
            for loc, res in zip(Bn[1]['Loc'], Bn[1]['result']):
                plt.text(loc[1], loc[0], str(res), color='b')
            if len(Bn[1]['Loc']) != 9:
                Err[1] = 1
        else:
            Err[1] = 1
        if Bn[2]['Loc']:
            for loc, res in zip(Bn[2]['Loc'], Bn[2]['result']):
                plt.text(loc[1], loc[0], res, color='b')
            if len(Bn[2]['Loc']) != 1:
                Err[2] = 1
        else:
            Err[2] = 1
        plt.title('结果分析标记', fontweight='bold')
        plt.show()

        # 错误提示弹窗
        root = Tk()
        root.withdraw()
        if Err[0]:
            messagebox.showinfo('提示信息', '试卷类型报警！')
        if Err[1]:
            messagebox.showinfo('提示信息', '准考证报警，请检查是否涂抹正确！')
        if Err[2]:
            messagebox.showinfo('提示信息', '考试科目报警！')
        root.destroy()

        # 保存结果到Excel
        excel_file = save_to_excel(Answer, Bn)
        messagebox.showinfo("保存成功", f"分析结果已保存到: {excel_file}")

    return Dom, Aom, Answer, Bn
