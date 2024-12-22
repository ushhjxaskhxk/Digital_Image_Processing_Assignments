import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detection(image_path):
    image = cv2.imread(image_path, 0)
    # 使用Canny算子进行边缘检测
    canny_edges = cv2.Canny(image, 100, 200)
    plt.imshow(canny_edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.show()

    # 与Prewitt算子检测结果比较
    # 这里可以从边缘的连续性、细腻程度、对噪声的抗干扰性等方面进行比较
    # 例如，直观上看Canny算子检测的边缘相对更细、连续性可能更好，而且对一些弱边缘的响应可能更合理等

    return canny_edges

image_path = r'C:\PICTURE\IMG2.jpg'  # 替换为实际的图像路径
canny_result = canny_edge_detection(image_path)
# 可以进一步通过可视化等方式更详细对比prewitt_result（前面Prewitt检测得到的二值化边缘图像）和canny_result的差异

