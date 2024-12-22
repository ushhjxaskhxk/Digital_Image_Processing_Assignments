import cv2


def hog_feature_extraction(image_path, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the image.")
    hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1], image.shape[0] // cell_size[0] * cell_size[0]),
                         _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                         _blockStride=(cell_size[1], cell_size[0]),
                         _cellSize=(cell_size[1], cell_size[0]),
                         _nbins=nbins)
    features = hog.compute(image)
    print(f'HOG features shape: {features.shape}')


# 示例使用
if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG3.png'  # 请替换为您的图像路径
    try:
        hog_feature_extraction(image_path)
    except ValueError as e:
        print(f"Error: {e}")
