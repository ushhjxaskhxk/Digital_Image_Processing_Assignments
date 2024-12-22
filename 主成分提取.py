import cv2
import numpy as np
from sklearn.decomposition import PCA

# 读取图像
filename = r"C:\PICTURE\IMG2.jpg"
img = cv2.imread(filename, 0)  # 以灰度模式读取图像

# 将图像数据重塑为二维数组
img_reshaped = img.reshape(-1, 1)

# 创建 PCA 对象并指定要保留的主成分数量
pca = PCA(n_components=1)

# 拟合数据并转换
img_pca = pca.fit_transform(img_reshaped)

# 转换回原始形状（如果需要可视化）
img_pca_reshaped = img_pca.reshape(img.shape)

# 显示结果（这里可能需要进一步处理以可视化，因为 PCA 结果可能不是图像格式）
cv2.imshow("PCA Result", img_pca_reshaped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图像到指定路径
output_path = r"C:\PICTURE\4\pca_result.jpg"
cv2.imwrite(output_path, img_pca_reshaped)