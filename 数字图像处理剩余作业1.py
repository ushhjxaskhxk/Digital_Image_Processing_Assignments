import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置 matplotlib 的字体，以支持中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# 读取灰度图像
def read_gray_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("图像读取失败，请检查路径是否正确")
    return image


# 显示图像
def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# 离散傅里叶变换及逆变换
def dft_and_idft(image_path):
    # 读取灰度图像
    image = read_gray_image(image_path)
    # 离散傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # 显示频率图
    show_image(magnitude_spectrum, "离散傅里叶变换频率图")
    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # 显示重建图像
    show_image(img_back, "逆变换后的重建图像")


# 理想低通滤波
def ideal_lowpass_filter(image_path, cutoff):
    image = read_gray_image(image_path)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # 生成理想低通滤波器
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    return img_filtered


# 巴特沃斯低通滤波
def butterworth_lowpass_filter(image_path, cutoff, order):
    image = read_gray_image(image_path)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # 生成巴特沃斯低通滤波器
    mask = np.zeros((rows, cols), np.float32)
    for i in np.arange(rows):
        for j in np.arange(cols):
            dist = math.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 / (1 + (dist / cutoff) ** (2 * order))
    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    return img_filtered


# 高斯低通滤波
def gaussian_lowpass_filter(image_path, cutoff):
    image = read_gray_image(image_path)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # 生成高斯低通滤波器
    mask = np.zeros((rows, cols), np.float32)
    for i in np.arange(rows):
        for j in np.arange(cols):
            dist = math.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = np.exp(-(dist ** 2) / (2 * (cutoff ** 2)))
    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    return img_filtered


# 空间平滑滤波（以均值滤波为例）
def spatial_smoothing_filter(image_path, kernel_size):
    image = read_gray_image(image_path)
    smoothed_image = cv2.blur(image, (kernel_size, kernel_size))
    return smoothed_image


# 对比结果
def compare_filters(image_path, cutoff_ideal, order_butterworth, cutoff_gaussian, kernel_size):
    image = read_gray_image(image_path)
    show_image(image, "原始图像")
    # 理想低通滤波
    img_ideal = ideal_lowpass_filter(image_path, cutoff_ideal)
    show_image(img_ideal, "理想低通滤波结果")
    # 巴特沃斯低通滤波
    img_butterworth = butterworth_lowpass_filter(image_path, cutoff_butterworth, order_butterworth)
    show_image(img_butterworth, "巴特沃斯低通滤波结果")
    # 高斯低通滤波
    img_gaussian = gaussian_lowpass_filter(image_path, cutoff_gaussian)
    show_image(img_gaussian, "高斯低通滤波结果")
    # 空间平滑滤波
    img_spatial = spatial_smoothing_filter(image_path, kernel_size)
    show_image(img_spatial, "空间平滑滤波结果")


if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG2-modified.jpg'
    cutoff_ideal = 30
    cutoff_butterworth = 2
    cutoff_gaussian = 30
    kernel_size = 5
    # 离散傅里叶变换及逆变换
    dft_and_idft(image_path)
    # 对比不同低通滤波结果
    compare_filters(image_path, cutoff_ideal, cutoff_butterworth, cutoff_gaussian, kernel_size)