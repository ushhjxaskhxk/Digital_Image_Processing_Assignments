import cv2
import numpy as np

# 读取图片
filename = r"C:\PICTURE\IMG2.jpg"
img = cv2.imread(filename, 0)

# Kirsch算子的8个卷积核
kirsch_kernels = [
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
]

# 计算边缘强度
max_intensity = np.zeros_like(img, dtype=np.float64)
for kernel in kirsch_kernels:
    conv = cv2.filter2D(img, cv2.CV_32F, kernel)
    np.maximum(max_intensity, conv, max_intensity)

# 转换为8位无符号整数
edge_img = np.uint8(np.absolute(max_intensity))

# 显示结果
cv2.imshow("Kirsch Edge Detection", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图像到指定路径
output_path = r"C:\PICTURE\4\edge_img.jpg"
cv2.imwrite(output_path, edge_img)