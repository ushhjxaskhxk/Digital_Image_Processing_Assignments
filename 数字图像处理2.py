import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 导入 Pillow 库

# 读取图像
image_path = r"C:\PICTURE\IMG2.jpg" # 这里替换为你的图片路径

# 尝试使用 OpenCV 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    # 如果 OpenCV 无法读取，尝试使用 Pillow 读取
    try:
        image = Image.open(image_path).convert('L')  # 使用 Pillow 读取并转换为灰度图像
        image = np.array(image)  # 将 PIL Image 对象转换为 NumPy 数组
    except:
        print(f"无法读取图像，请检查图片路径和格式：{image_path}")
        exit()

# 平滑处理
# 1. 均值滤波器
mean_filtered_image = cv2.blur(image, (5, 5))

# 2. 方框滤波器
box_filtered_image = cv2.boxFilter(image, -1, (5, 5), normalize=True)

# 3. 高斯滤波器
gaussian_filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(mean_filtered_image, cmap='gray')
plt.title('Mean Filtered Image')

plt.subplot(1, 4, 3)
plt.imshow(box_filtered_image, cmap='gray')
plt.title('Box Filtered Image')

plt.subplot(1, 4, 4)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filtered Image')

plt.tight_layout()
plt.show()

