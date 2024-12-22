import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 导入 Pillow 库

# 1. 灰度级切片
def gray_level_slicing(image, threshold1, threshold2):
    """
    灰度级切片函数
    Args:
        image: 输入图像
        threshold1: 第一个阈值
        threshold2: 第二个阈值
    Returns:
        切片后的图像
    """
    # 将像素值小于 threshold1 或大于 threshold2 的像素值设置为 0
    # 将介于 threshold1 和 threshold2 之间的像素值设置为 255
    sliced_image = np.where((image >= threshold1) & (image <= threshold2), 255, 0).astype(np.uint8)
    return sliced_image

# 2. 位平面切片
def bit_plane_slicing(image):
    """
    位平面切片函数
    Args:
        image: 输入图像
    Returns:
        8 个位平面图像
    """
    bit_planes = []
    for i in range(8):
        # 获取第 i 位平面
        bit_plane = (image >> i) & 1
        # 将位平面转换为 8 位图像
        bit_plane = bit_plane * 255
        bit_planes.append(bit_plane)
    return bit_planes

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

# 1. 灰度级切片
threshold1 = 100
threshold2 = 200
sliced_image = gray_level_slicing(image, threshold1, threshold2)

# 2. 位平面切片
bit_planes = bit_plane_slicing(image)

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(sliced_image, cmap='gray')
plt.title('Sliced Image')

plt.subplot(1, 3, 3)
for i, bit_plane in enumerate(bit_planes):
    plt.subplot(2, 4, i + 1)
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {i + 1}')

plt.tight_layout()
plt.show()
