import cv2
import numpy as np
import os

# 1. 灰度图像伪彩色处理 - 灰度分层
def grayscale_to_pseudocolor(image_path):
    """将灰度图像转换为伪彩色图像。

    Args:
        image_path: 灰度图像路径。

    Returns:
        伪彩色图像。
    """

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    # 缩放图像
    img = cv2.resize(img, (400, 400))  # 缩放到 400x400 像素

    # 创建颜色映射表
    color_map = np.array([
        [0, 0, 0],  # 黑色
        [0, 255, 0],  # 绿色
        [255, 0, 0],  # 红色
        [255, 255, 0],  # 黄色
        [0, 0, 255],  # 蓝色
    ])
    # 将灰度值映射到颜色
    pseudocolor_img = color_map[np.floor_divide(img, 51)]
    return pseudocolor_img

# 2. 彩色图像处理
def color_image_processing(image_path):
    """对彩色图像进行平滑、锐化等处理。

    Args:
        image_path: 彩色图像路径。

    Returns:
        处理后的彩色图像。
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    # 平滑处理 (均值滤波)
    smooth_img = cv2.blur(img, (5, 5))
    # 锐化处理 (拉普拉斯算子)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return smooth_img, sharpened_img

# 3. HIS 直方图均衡化
def his_histogram_equalization(image_path):
    """对彩色图像进行 HIS 直方图均衡化。

    Args:
        image_path: 彩色图像路径。

    Returns:
        均衡化后的图像。
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    # 将图像转换为 HIS 空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 对亮度分量进行直方图均衡化
    hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])
    # 将图像转换回 RGB 空间
    equalized_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return equalized_img

# 4. 一阶锐化处理
def first_order_sharpening(image_path, operator_type):
    """对图像进行一阶锐化处理。

    Args:
        image_path: 图像路径。
        operator_type: 锐化算子类型，支持 'roberts', 'sobel', 'prewitt', 'kirsch'。

    Returns:
        锐化后的图像。
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    if operator_type == 'roberts':
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
    elif operator_type == 'sobel':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == 'prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif operator_type == 'kirsch':
        kernels = [
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, -3, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [5, 0, -3], [-3, -3, -3]]),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [-3, 5, 5]]),
            np.array([[-3, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        ]
        sharpened_img = np.zeros_like(img)
        for kernel in kernels:
            temp = cv2.filter2D(img, -1, kernel)
            sharpened_img = np.maximum(sharpened_img, temp)
        return sharpened_img
    else:
        print("Invalid operator type. Supported types: roberts, sobel, prewitt, kirsch")
        return None

    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad

# 5. 二阶锐化处理 (拉普拉斯算子)
def second_order_sharpening(image_path):
    """对图像进行二阶锐化处理 (拉普拉斯算子)。

    Args:
        image_path: 图像路径。

    Returns:
        锐化后的图像。
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    # 拉普拉斯算子
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

# 显示结果
def display_results(original_img, sharpened_img, operator_type):
    """显示原图、算子图片以及叠加后的锐化图片。

    Args:
        original_img: 原图。
        sharpened_img: 锐化后的图像。
        operator_type: 锐化算子类型。
    """

    # 将 sharpened_img 转换为 uint8 类型
    sharpened_img = sharpened_img.astype(np.uint8)

    # 缩小图像尺寸，确保 original_img 和 sharpened_img 尺寸一致
    original_img = cv2.resize(original_img, (600, 800))  # 调整尺寸为 600x800
    sharpened_img = cv2.resize(sharpened_img, (600, 800))

    cv2.imshow(f"Original Image", original_img)
    cv2.imshow(f"{operator_type.upper()} Operator", sharpened_img)

    # 叠加显示
    alpha = 0.5
    beta = 1 - alpha
    merged_img = cv2.addWeighted(original_img, alpha, sharpened_img, beta, 0.0)
    cv2.imshow(f"Sharpened Image ({operator_type.upper()})", merged_img)

    # 等待一段时间后关闭窗口
    cv2.waitKey(0)  # 等待用户按下任意键
    cv2.destroyAllWindows()

# 示例调用
if __name__ == "__main__":
    # 读取图像
    image_path = "C:\PICTURE\IMG2-modified.jpg"  # 替换成您的图像路径
    print(f"Image Path: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
    else:
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is not None:
            print(f"Successfully read image: {image_path}")  # 打印确认信息

            # 一阶锐化处理，只显示最后一种算子的结果
            operator_types = ['roberts', 'sobel', 'prewitt', 'kirsch']
            for operator_type in operator_types:
                sharpened_img = first_order_sharpening(image_path, operator_type)
                if sharpened_img is not None:
                    pass  # 不显示中间结果
                else:
                    print(f"Error: Unable to process image with {operator_type} operator.")

            # 显示最后一种算子的结果
            sharpened_img = first_order_sharpening(image_path, 'kirsch')
            if sharpened_img is not None:
                display_results(original_img, sharpened_img, 'kirsch')
            else:
                print(f"Error: Unable to process image with kirsch operator.")

            # 二阶锐化处理 (拉普拉斯算子)
            sharpened_img = second_order_sharpening(image_path)
            if sharpened_img is not None:
                display_results(original_img, sharpened_img, "Laplacian")
            else:
                print(f"Error: Unable to process image with Laplacian operator.")

        else:
            print(f"Error: Unable to read image: {image_path}")