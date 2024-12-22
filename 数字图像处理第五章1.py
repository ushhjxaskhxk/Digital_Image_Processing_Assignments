import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像
filename = r"C:\PICTURE\IMG2.jpg"
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


# Part A: 离散傅立叶变换和逆变换
# 计算傅立叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)


# 显示频谱图
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# 逆傅立叶变换
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


# 显示重建图像
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.show()


# Part B: 低通滤波
# 创建高斯低通滤波器
def gaussian_lowpass(shape, cutoff):
    center = np.array(shape) // 2
    x, y = np.indices(shape)
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    filter = np.exp(-(distance_from_center ** 2) / (2 * (cutoff ** 2)))
    return filter


# 创建理想低通滤波器
def ideal_lowpass(shape, cutoff):
    center = np.array(shape) // 2
    x, y = np.indices(shape)
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    filter = np.where(distance_from_center <= cutoff, 1, 0)
    return filter


# 创建巴特沃斯低通滤波器
def butterworth_lowpass(shape, cutoff, order):
    center = np.array(shape) // 2
    x, y = np.indices(shape)
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    filter = 1 / (1 + (distance_from_center / cutoff) ** (2 * order))
    return filter


# 滤波参数
cutoff = 50
order = 2


# 创建滤波器
gaussian_filter = gaussian_lowpass(img.shape, cutoff)
ideal_filter = ideal_lowpass(img.shape, cutoff)
butterworth_filter = butterworth_lowpass(img.shape, cutoff, order)


# 应用滤波器
filtered_gaussian = fshift * gaussian_filter
filtered_ideal = fshift * ideal_filter
filtered_butterworth = fshift * butterworth_filter


# 逆傅立叶变换
img_back_gaussian = np.fft.ifft2(np.fft.ifftshift(filtered_gaussian))
img_back_ideal = np.fft.ifft2(np.fft.ifftshift(filtered_ideal))
img_back_butterworth = np.fft.ifft2(np.fft.ifftshift(filtered_butterworth))


# 显示结果
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(np.abs(img_back_gaussian), cmap='gray')
plt.title('Gaussian Lowpass'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(np.abs(img_back_ideal), cmap='gray')
plt.title('Ideal Lowpass'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(np.abs(img_back_butterworth), cmap='gray')
plt.title('Butterworth Lowpass'), plt.xticks([]), plt.yticks([])
plt.show()


# Part C: 频率域拉普拉斯算子
# 拉普拉斯算子核
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


# 扩展拉普拉斯算子核
kernel_fft = np.fft.fft2(laplacian_kernel, s=img.shape)


# 频率域拉普拉斯算子
f_laplacian = np.fft.fft2(img)
fshift_laplacian = np.fft.fftshift(f_laplacian)
filtered_laplacian = fshift_laplacian * kernel_fft
img_back_laplacian = np.fft.ifft2(np.fft.ifftshift(filtered_laplacian))
img_back_laplacian = np.abs(img_back_laplacian)


# 空间域拉普拉斯算子
img_laplacian = cv2.Laplacian(img, cv2.CV_64F)


# 显示结果
plt.subplot(1, 2, 1), plt.imshow(img_back_laplacian, cmap='gray')
plt.title('Frequency Domain Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(img_laplacian, cmap='gray')
plt.title('Spatial Domain Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

