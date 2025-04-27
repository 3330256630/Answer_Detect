# Answer Sheet Analysis System / 答题卡分析系统

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This project is a computer vision-based system for analyzing answer sheets. It can detect and process multiple-choice answers, exam numbers, and subject information from scanned answer sheets.

### Features

- Automatic detection of answer sheet grid lines
- Processing of multiple-choice answers (A, B, C, D)
- Recognition of exam numbers and subject information
- Visualization of detected answers and grid lines
- Export of results to Excel format
- Error detection and alerts for missing or incorrect information

### Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Pandas

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/answer-sheet-analysis.git
cd answer-sheet-analysis
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

The main functionality is provided by the `analysis` function in `analysis.py`. Here's a basic example of how to use it:

```python
import cv2
from analysis import analysis

# Load your answer sheet image
image = cv2.imread('answer_sheet.jpg')

# Process the image
# Note: You'll need to provide the appropriate parameters for stats1, stats2, and Line
Dom, Aom, Answer, Bn = analysis(stats1, stats2, Line, image, show=True)
```

The function will:
1. Process the image and detect grid lines
2. Analyze the answers and special areas
3. Display visualizations of the results
4. Save the results to an Excel file
5. Show alerts for any missing or incorrect information

### Output

The analysis results are saved in an Excel file with two sheets:
- "答题结果" (Answer Results): Contains the question numbers and selected answers
- "特殊区域" (Special Areas): Contains information about exam type, exam number, and subject

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<a name="chinese"></a>
## 中文

本项目是一个基于计算机视觉的答题卡分析系统。它能够从扫描的答题卡中检测和处理选择题答案、准考证号和科目信息。

### 功能特点

- 自动检测答题卡网格线
- 处理选择题答案（A、B、C、D）
- 识别准考证号和科目信息
- 可视化检测到的答案和网格线
- 将结果导出为Excel格式
- 错误检测和缺失/错误信息提醒

### 系统要求

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Pandas

### 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/answer-sheet-analysis.git
cd answer-sheet-analysis
```

2. 安装所需包：
```bash
pip install -r requirements.txt
```

### 使用方法

主要功能由 `analysis.py` 中的 `analysis` 函数提供。以下是基本使用示例：

```python
import cv2
from analysis import analysis

# 加载答题卡图像
image = cv2.imread('answer_sheet.jpg')

# 处理图像
# 注意：需要提供适当的 stats1、stats2 和 Line 参数
Dom, Aom, Answer, Bn = analysis(stats1, stats2, Line, image, show=True)
```

该函数将：
1. 处理图像并检测网格线
2. 分析答案和特殊区域
3. 显示结果可视化
4. 将结果保存到Excel文件
5. 显示缺失或错误信息的提醒

### 输出结果

分析结果保存在Excel文件中，包含两个工作表：
- "答题结果"：包含题号和所选答案
- "特殊区域"：包含试卷类型、准考证号和科目信息
