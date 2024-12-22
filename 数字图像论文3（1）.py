import cv2
import numpy as np
import matplotlib.pyplot as plt


def prewitt_edge_detection(image_path, threshold=100):
    # 读取灰度图像
    image = cv2.imread(image_path, 0)

    # 使用中值滤波进行预处理去除噪声
    image = cv2.medianBlur(image, 3)

    # 使用Prewitt算子进行卷积计算水平和垂直方向梯度
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
    gradient_x = cv2.filter2D(image, -1, prewitt_x)
    gradient_y = cv2.filter2D(image, -1, prewitt_y)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2).astype(np.uint8)

    # 分析图片效果
    # 这里可以简单查看一下原始梯度幅值图像的情况
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Prewitt Gradient Magnitude")
    plt.show()

    # 采用更大的高斯核进行模糊
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gradient_x_blurred = cv2.filter2D(blurred_image, -1, prewitt_x)
    gradient_y_blurred = cv2.filter2D(blurred_image, -1, prewitt_y)
    gradient_magnitude_blurred = np.sqrt(gradient_x_blurred ** 2 + gradient_y_blurred ** 2).astype(np.uint8)
    plt.imshow(gradient_magnitude_blurred, cmap='gray')
    plt.title("Prewitt Gradient Magnitude after Gaussian Blur")
    plt.show()

    # 采用对角线的Prewitt梯度算子进行处理（定义对角线算子）
    prewitt_diagonal_1 = np.array([[0, 1, 1],
                                   [-1, 0, 1],
                                   [-1, -1, 0]])
    prewitt_diagonal_2 = np.array([[1, 1, 0],
                                   [1, 0, -1],
                                   [0, -1, -1]])
    gradient_diagonal_1 = cv2.filter2D(image, -1, prewitt_diagonal_1)
    gradient_diagonal_2 = cv2.filter2D(image, -1, prewitt_diagonal_2)
    gradient_magnitude_diagonal = np.sqrt(gradient_diagonal_1 ** 2 + gradient_diagonal_2 ** 2).astype(np.uint8)
    plt.imshow(gradient_magnitude_diagonal, cmap='gray')
    plt.title("Diagonal Prewitt Gradient Magnitude")
    plt.show()

    # 确保combined_gradient_magnitude的数据类型为CV_8UC1
    combined_gradient_magnitude = (gradient_magnitude + gradient_magnitude_blurred + gradient_magnitude_diagonal) / 3
    combined_gradient_magnitude = combined_gradient_magnitude.astype(np.uint8)

    # 使用自适应阈值处理
    binary_image = cv2.adaptiveThreshold(combined_gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Prewitt Edge Detection after Thresholding")
    plt.show()

    return binary_image


image_path = r'C:\PICTURE\IMG2.jpg'  # 替换为实际的图像路径
prewitt_result = prewitt_edge_detection(image_path)
