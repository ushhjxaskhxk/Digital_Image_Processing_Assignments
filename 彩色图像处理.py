import cv2
import numpy as np
import os

# 使用 uigetfile 函数选择图像文件  (这部分在实际运行中可能需要根据你的环境修改)
filename = r"C:\PICTURE\IMG2.jpg"  # Replace with your image file path
img = cv2.imread(filename)

# RGB 空间处理
# 平滑
smooth_img_rgb = cv2.GaussianBlur(img, (5, 5), 3)  # 高斯平滑，sigma = 3
# 锐化
sharp_img_rgb = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

# HIS 空间处理
# 转换为 HIS 空间
img_his = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 平滑
smooth_img_his = cv2.GaussianBlur(img_his, (5, 5), 3)
# 锐化
sharp_img_his = cv2.filter2D(img_his, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
# 转换回 RGB 空间
sharp_img_his = cv2.cvtColor(sharp_img_his, cv2.COLOR_HSV2BGR)

# 定义保存路径
save_path = r"C:\PICTURE\0"

# 创建文件夹如果不存在
os.makedirs(save_path, exist_ok=True)

# 保存图像
cv2.imwrite(os.path.join(save_path, "original.jpg"), img)
cv2.imwrite(os.path.join(save_path, "smooth_rgb.jpg"), smooth_img_rgb)
cv2.imwrite(os.path.join(save_path, "sharp_rgb.jpg"), sharp_img_rgb)
cv2.imwrite(os.path.join(save_path, "his.jpg"), img_his)
cv2.imwrite(os.path.join(save_path, "smooth_his.jpg"), smooth_img_his)
cv2.imwrite(os.path.join(save_path, "sharp_his.jpg"), sharp_img_his)


# 显示图像 (可选，如果不需要显示，可以注释掉)
cv2.imshow('原图', img)
cv2.imshow('RGB 平滑', smooth_img_rgb)
cv2.imshow('RGB 锐化', sharp_img_rgb)
cv2.imshow('HIS 空间', img_his)
cv2.imshow('HIS 平滑', smooth_img_his)
cv2.imshow('HIS 锐化', sharp_img_his)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Images saved to:", save_path)
