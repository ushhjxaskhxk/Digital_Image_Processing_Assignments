import cv2
import numpy as np

# 读取灰度图像
img = cv2.imread(r'C:\PICTURE\IMG2-modified.jpg', cv2.IMREAD_GRAYSCALE)

# 定义灰度分层阈值
threshold1 = 50  # 调整 threshold1 和 threshold2 的值
threshold2 = 150

# 创建伪彩色查找表
lut = np.zeros((256, 1), dtype='uint8')
lut[0:threshold1] = 0  # 0-threshold1 为蓝色
lut[threshold1+1:threshold2] = 1  # threshold1-threshold2 为绿色
lut[threshold2+1:256] = 2  # threshold2-255 为红色

# 将灰度图像转换为伪彩色图像
indexedImage = lut[img]
# 使用 squeeze 函数移除最后一个维度
indexedImage = np.squeeze(indexedImage)

# 定义颜色映射
color_map = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype='uint8')  # 使用更鲜明的颜色

# 使用 color_map(indexedImage, 1) 直接获取颜色索引矩阵
pseudo_color_img = color_map[indexedImage]

# 缩放到宽度为 400 像素
pseudo_color_img_resized = cv2.resize(pseudo_color_img, (400, int(pseudo_color_img.shape[0] * 400 / pseudo_color_img.shape[1])))

# 检查 indexedImage 的值范围
print(f"indexedImage 的最小值: {np.min(indexedImage)}")
print(f"indexedImage 的最大值: {np.max(indexedImage)}")

# 显示图像
cv2.imshow('伪彩色图像', pseudo_color_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
