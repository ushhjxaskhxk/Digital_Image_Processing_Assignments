import cv2
import numpy as np
import os


# 读取灰度图像
def read_gray_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the image.")
    return image


# Prewitt 算子边缘检测
def prewitt_edge_detection(image):
    # 定义 Prewitt 算子的水平和垂直模板
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # 使用 cv2.filter2D 进行卷积操作
    gradient_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)
    return gradient_magnitude


# 高斯模糊后的 Prewitt 边缘检测
def prewitt_edge_detection_with_gaussian(image, kernel_size=(5, 5), sigma=1.0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return prewitt_edge_detection(blurred_image)


# 对角线 Prewitt 梯度幅值图
def diagonal_prewitt_edge_detection(image):
    prewitt_diagonal1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    prewitt_diagonal2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    gradient_diagonal1 = cv2.filter2D(image, cv2.CV_64F, prewitt_diagonal1)
    gradient_diagonal2 = cv2.filter2D(image, cv2.CV_64F, prewitt_diagonal2)
    gradient_magnitude = np.sqrt(gradient_diagonal1 ** 2 + gradient_diagonal2 ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)
    return gradient_magnitude


# 阈值化后的 Prewitt 边缘检测图
def thresholded_prewitt_edge_detection(image, threshold):
    gradient_magnitude = prewitt_edge_detection(image)
    _, thresholded_image = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    return thresholded_image


# 保存图像函数
def save_image(image, save_path):
    cv2.imwrite(save_path, image)


# 示例使用
if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG3.png'  # 请替换为您的图像路径
    output_dir = 'C:\\PICTURE\\3\\1'  # 输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录，如果不存在

    try:
        image = read_gray_image(image_path)

        # 原始 Prewitt 边缘检测
        prewitt_edges = prewitt_edge_detection(image)
        save_path = os.path.join(output_dir, 'Prewitt Edges.png')
        save_image(prewitt_edges, save_path)

        # 高斯模糊后的 Prewitt 边缘检测
        prewitt_edges_gaussian = prewitt_edge_detection_with_gaussian(image)
        save_path = os.path.join(output_dir, 'Prewitt Edges with Gaussian.png')
        save_image(prewitt_edges_gaussian, save_path)

        # 对角线 Prewitt 梯度幅值图
        diagonal_prewitt_edges = diagonal_prewitt_edge_detection(image)
        save_path = os.path.join(output_dir, 'Diagonal Prewitt Edges.png')
        save_image(diagonal_prewitt_edges, save_path)

        # 阈值化后的 Prewitt 边缘检测图
        threshold = 100
        thresholded_prewitt_edges = thresholded_prewitt_edge_detection(image, threshold)
        save_path = os.path.join(output_dir, f'Thresholded Prewitt Edges (th={threshold}).png')
        save_image(thresholded_prewitt_edges, save_path)
    except ValueError as e:
        print(f"Error: {e}")
