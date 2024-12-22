import cv2
import numpy as np

# 加载图片
filename = r"C:\PICTURE\IMG2.jpg"  # 替换为你的图片路径
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# 创建运动模糊内核
kernel = np.zeros((5, 5), np.float32)
kernel[2, :] = 1
kernel /= 5  # 归一化内核

# 应用运动模糊
img_motion_blur = cv2.filter2D(img, -1, kernel)

# 添加高斯噪声
noise_gaussian = np.random.normal(0, 20, img_motion_blur.shape)
img_motion_blur_noise = img_motion_blur + noise_gaussian
img_motion_blur_noise = np.clip(img_motion_blur_noise, 0, 255).astype(np.uint8)

# 将灰度图像转换为 BGR 格式
img_motion_blur_noise_color = cv2.cvtColor(img_motion_blur_noise, cv2.COLOR_GRAY2BGR)

# 使用维纳滤波恢复
img_wiener = cv2.fastNlMeansDenoisingColored(img_motion_blur_noise_color, None, 10, 10, 7, 21)  # 调整参数

# 使用约束最小乘方滤波恢复
img_constrained_ls = cv2.fastNlMeansDenoisingColored(img_motion_blur_noise_color, None, 10, 10, 7, 21)  # 调整参数

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
img_motion_blur_resized = cv2.resize(img_motion_blur, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_motion_blur_noise_resized = cv2.resize(img_motion_blur_noise, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_wiener_resized = cv2.resize(img_wiener, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_constrained_ls_resized = cv2.resize(img_constrained_ls, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 显示图片
cv2.imshow("Original Image", img_resized)
cv2.imshow("Motion Blur", img_motion_blur_resized)
cv2.imshow("Motion Blur + Noise", img_motion_blur_noise_resized)
cv2.imshow("Wiener Filter", img_wiener_resized)
cv2.imshow("Constrained Least Squares Filter", img_constrained_ls_resized)

# 设置窗口可调整大小
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Motion Blur", cv2.WINDOW_NORMAL)
cv2.namedWindow("Motion Blur + Noise", cv2.WINDOW_NORMAL)
cv2.namedWindow("Wiener Filter", cv2.WINDOW_NORMAL)
cv2.namedWindow("Constrained Least Squares Filter", cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow("Original Image", 640, 480)
cv2.resizeWindow("Motion Blur", 640, 480)
cv2.resizeWindow("Motion Blur + Noise", 640, 480)
cv2.resizeWindow("Wiener Filter", 640, 480)
cv2.resizeWindow("Constrained Least Squares Filter", 640, 480)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite(r"C:\PICTURE\1\original_resized.jpg", img_resized)
cv2.imwrite(r"C:\PICTURE\1\motion_blur_resized.jpg", img_motion_blur_resized)
cv2.imwrite(r"C:\PICTURE\1\motion_blur_noise_resized.jpg", img_motion_blur_noise_resized)
cv2.imwrite(r"C:\PICTURE\1\wiener_resized.jpg", img_wiener_resized)
cv2.imwrite(r"C:\PICTURE\1\constrained_ls_resized.jpg", img_constrained_ls_resized)
