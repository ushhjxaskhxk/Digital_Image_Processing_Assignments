import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载噪声污染的图片
filename_gaussian = r"C:\PICTURE\1\2-1\gaussian_noise_resized.jpg"  # 替换为你的高斯噪声图片路径
filename_uniform = r"C:\PICTURE\1\2-1\uniform_noise_resized.jpg"  # 替换为你的均匀噪声图片路径
filename_salt_pepper = r"C:\PICTURE\1\2-1\salt_pepper_noise_resized.jpg"  # 替换为你的椒盐噪声图片路径

img_gaussian = cv2.imread(filename_gaussian, cv2.IMREAD_GRAYSCALE)
img_uniform = cv2.imread(filename_uniform, cv2.IMREAD_GRAYSCALE)
img_salt_pepper = cv2.imread(filename_salt_pepper, cv2.IMREAD_GRAYSCALE)

# 选择合适的滤波器
# 高斯噪声：使用高斯滤波
img_gaussian_filter = cv2.GaussianBlur(img_gaussian, (5, 5), 0)  # 调整内核大小 (5, 5) 和标准差 (0)

# 均匀噪声：使用中值滤波
img_uniform_median = cv2.medianBlur(img_uniform, 5)  # 调整内核大小 (5) 控制滤波范围

# 椒盐噪声：使用中值滤波
img_salt_pepper_median = cv2.medianBlur(img_salt_pepper, 5)  # 调整内核大小 (5) 控制滤波范围

# 调整图像尺寸，保持纵横比
height, width = img_gaussian.shape[:2]  # 使用高斯噪声图片的尺寸
max_size = 640  # 设置最大尺寸
if height > width:
    new_height = max_size
    new_width = int(width * max_size / height)
else:
    new_width = max_size
    new_height = int(height * max_size / width)
img_gaussian_resized = cv2.resize(img_gaussian, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_uniform_resized = cv2.resize(img_uniform, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_salt_pepper_resized = cv2.resize(img_salt_pepper, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_gaussian_filter_resized = cv2.resize(img_gaussian_filter, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_uniform_median_resized = cv2.resize(img_uniform_median, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_salt_pepper_median_resized = cv2.resize(img_salt_pepper_median, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 使用 matplotlib 展示对比图
plt.figure(figsize=(12, 4))  # 设置图形大小

plt.subplot(1, 3, 1)  # 创建子图 1
plt.imshow(img_gaussian_resized, cmap='gray')  # 显示高斯噪声图片
plt.title("Original Gaussian Noise")

plt.subplot(1, 3, 2)  # 创建子图 2
plt.imshow(img_gaussian_filter_resized, cmap='gray')  # 显示高斯滤波处理后的图片
plt.title("Gaussian Filter")

plt.subplot(1, 3, 3)  # 创建子图 3
plt.imshow(img_uniform_resized, cmap='gray')  # 显示均匀噪声图片
plt.title("Original Uniform Noise")

plt.figure(figsize=(12, 4))  # 新建图形
plt.subplot(1, 2, 1)  # 创建子图 1
plt.imshow(img_uniform_median_resized, cmap='gray')  # 显示中值滤波处理后的图片
plt.title("Median Filter (Uniform Noise)")

plt.subplot(1, 2, 2)  # 创建子图 2
plt.imshow(img_salt_pepper_resized, cmap='gray')  # 显示椒盐噪声图片
plt.title("Original Salt & Pepper Noise")

plt.figure(figsize=(12, 4))  # 新建图形
plt.subplot(1, 1, 1)  # 创建子图 1
plt.imshow(img_salt_pepper_median_resized, cmap='gray')  # 显示中值滤波处理后的图片
plt.title("Median Filter (Salt & Pepper Noise)")

plt.tight_layout()  # 调整子图间距
plt.show()
