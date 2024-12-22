import cv2
import os


def canny_edge_detection(image_path, low_threshold, high_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the image.")
    edges = cv2.Canny(image, low_threshold, high_threshold)
    cv2.imshow(f'Canny Edges (th={low_threshold}, {high_threshold})', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges


def save_image(image, save_path):
    cv2.imwrite(save_path, image)


# 示例使用
if __name__ == "__main__":
    image_path = 'C:\\PICTURE\\IMG3.png'  # 请替换为您的图像路径
    low_threshold = 50
    high_threshold = 150
    output_dir = 'C:\\PICTURE\\3\\2'  # 输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录，如果不存在
    try:
        edges = canny_edge_detection(image_path, low_threshold, high_threshold)
        save_path = os.path.join(output_dir, f'Canny Edges (th={low_threshold}, {high_threshold}).png')
        save_image(edges, save_path)
    except ValueError as e:
        print(f"Error: {e}")