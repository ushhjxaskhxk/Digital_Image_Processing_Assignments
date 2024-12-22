import cv2
import numpy as np
import os


def hough_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Hough Transform', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def save_image(image, save_path):
    cv2.imwrite(save_path, image)


# 示例使用
if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG3.png'  # 请替换为您的图像路径
    output_dir = 'C:\\PICTURE\\3\\4'  # 输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录，如果不存在
    try:
        result_image = hough_transform(image_path)
        save_path = os.path.join(output_dir, 'Hough Transform.png')
        save_image(result_image, save_path)
    except ValueError as e:
        print(f"Error: {e}")
