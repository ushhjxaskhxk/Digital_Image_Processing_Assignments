import cv2
import numpy as np
import matplotlib.pyplot as plt  # 添加 matplotlib 库用于绘制直方图

# 加载图片
filename = r"C:\PICTURE\IMG2.jpg"  # 替换为你的图片路径
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声
noise_gaussian = np.random.normal(0, 20, img.shape)  # 调整标准差值 (20) 控制噪声强度
img_gaussian = img + noise_gaussian
img_gaussian = np.clip(img_gaussian, 0, 255).astype(np.uint8)

# 添加均匀噪声
noise_uniform = np.random.uniform(0, 50, img.shape)  # 调整噪声范围 (0, 50) 控制强度
img_uniform = img + noise_uniform
img_uniform = np.clip(img_uniform, 0, 255).astype(np.uint8)

# 添加椒盐噪声
prob = 0.05  # 调整概率控制噪声密度
img_salt_pepper = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.random.rand() < prob:
            img_salt_pepper[i, j] = 0  # 黑点
        elif np.random.rand() < prob:
            img_salt_pepper[i, j] = 255  # 白点

# 调整图像尺寸，保持纵横比
height, width = img.shape[:2]
max_size = 640  # 设置最大尺寸
if height > width:
    new_height = max_size
    new_width = int(width * max_size / height)
else:
    new_width = max_size
    new_height = int(height * max_size / width)
img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_gaussian_resized = cv2.resize(img_gaussian, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_uniform_resized = cv2.resize(img_uniform, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_salt_pepper_resized = cv2.resize(img_salt_pepper, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 显示图片
cv2.imshow("Original Image", img_resized)
cv2.imshow("Gaussian Noise", img_gaussian_resized)
cv2.imshow("Uniform Noise", img_uniform_resized)
cv2.imshow("Salt & Pepper Noise", img_salt_pepper_resized)

# 设置窗口可调整大小
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Gaussian Noise", cv2.WINDOW_NORMAL)
cv2.namedWindow("Uniform Noise", cv2.WINDOW_NORMAL)
cv2.namedWindow("Salt & Pepper Noise", cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow("Original Image", 640, 480)
cv2.resizeWindow("Gaussian Noise", 640, 480)
cv2.resizeWindow("Uniform Noise", 640, 480)
cv2.resizeWindow("Salt & Pepper Noise", 640, 480)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite(r"C:\PICTURE\1\original_resized.jpg", img_resized)
cv2.imwrite(r"C:\PICTURE\1\gaussian_noise_resized.jpg", img_gaussian_resized)
cv2.imwrite(r"C:\PICTURE\1\uniform_noise_resized.jpg", img_uniform_resized)
cv2.imwrite(r"C:\PICTURE\1\salt_pepper_noise_resized.jpg", img_salt_pepper_resized)

# 绘制直方图
plt.figure(figsize=(12, 4))  # 设置图形大小

plt.subplot(1, 4, 1)  # 创建子图 1
plt.hist(img_resized.ravel(), 256, [0, 256])  # 绘制原图直方图
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1, 4, 2)  # 创建子图 2
plt.hist(img_gaussian_resized.ravel(), 256, [0, 256])  # 绘制高斯噪声直方图
plt.title("Gaussian Noise Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1, 4, 3)  # 创建子图 3
plt.hist(img_uniform_resized.ravel(), 256, [0, 256])  # 绘制均匀噪声直方图
plt.title("Uniform Noise Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1, 4, 4)  # 创建子图 4
plt.hist(img_salt_pepper_resized.ravel(), 256, [0, 256])  # 绘制椒盐噪声直方图
plt.title("Salt & Pepper Noise Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()  # 调整子图间距
plt.show()
