import cv2
import numpy as np
import os


def harris_corner_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the image.")
    dst = cv2.cornerHarris(image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = 255
    cv2.imshow('Harris Corner Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def save_image(image, save_path):
    cv2.imwrite(save_path, image)


# 示例使用
if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG3.png'  # 请替换为您的图像路径
    output_dir = 'C:\\PICTURE\\3\\6'  # 输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录，如果不存在
    try:
        result_image = harris_corner_detection(image_path)
        save_path = os.path.join(output_dir, 'Harris Corner Detection.png')
        save_image(result_image, save_path)
    except ValueError as e:
        print(f"Error: {e}")