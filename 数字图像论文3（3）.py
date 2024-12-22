import cv2
import numpy as np
import matplotlib.pyplot as plt

def hog_feature_extraction(image_path, cell_size=(8, 8), block_size=(2, 2), block_stride=(1, 1), num_bins=9):
    """
    对图片进行HOG特征提取，并绘制归一化后的直方图。

    Args:
        image_path: 图片路径
        cell_size: cell大小
        block_size: block大小
        block_stride: block步长
        num_bins: 直方图bin的数量

    Returns:
        归一化后的HOG特征向量
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate ideal dimensions divisible by block and cell sizes
    ideal_width = (gray_image.shape[1] // cell_size[0]) * cell_size[0]
    ideal_height = (gray_image.shape[0] // cell_size[1]) * cell_size[1]

    # Resize the image to the ideal dimensions
    gray_image = cv2.resize(gray_image, (ideal_width, ideal_height))

    #Calculate winSize such that it is compatible with blockSize and blockStride.
    win_size_width = gray_image.shape[1]
    win_size_height = gray_image.shape[0]


    hog = cv2.HOGDescriptor(_winSize=(win_size_width, win_size_height),
                            _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
                            _blockStride=(block_stride[0] * cell_size[0], block_stride[1] * cell_size[1]),
                            _cellSize=(cell_size[0], cell_size[1]),
                            _nbins=num_bins)

    hog_features = hog.compute(gray_image).ravel()
    normalized_hog_features = hog_features / np.linalg.norm(hog_features)

    plt.hist(normalized_hog_features, bins=num_bins)
    plt.title("HOG Normalized Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()

    return normalized_hog_features

image_path = r'C:\PICTURE\IMG2.jpg'  # 替换为实际的图像路径
try:
    hog_features = hog_feature_extraction(image_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")




