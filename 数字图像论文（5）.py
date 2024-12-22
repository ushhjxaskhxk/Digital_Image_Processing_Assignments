import cv2
import os


def viola_jones_face_detection(image_path, scale_factor=0.5):
    """
    使用 Viola-Jones 算法进行人脸检测。

    参数：
        image_path: 图片的路径 (字符串)。
        scale_factor: 图像缩放比例 (浮点数，默认为 0.5)。
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件：{image_path}")

        # 级联分类器的相对路径 (确保 'data' 文件夹存在且包含分类器文件)
        cascade_path = 'data/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise ValueError(f"无法加载级联分类器：{cascade_path}")


        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 绘制人脸矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 获取原始图像的尺寸
        height, width = image.shape[:2]
        # 按比例缩放图像
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # 显示结果
        cv2.imshow('Viola-Jones 人脸检测', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return resized_image
    except ValueError as e:
        print(f"值错误：{e}")
        return None
    except cv2.error as e:
        print(f"OpenCV 错误：{e}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None


def save_image(image, save_path):
    try:
        # 检查图像是否包含 alpha 通道
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(save_path, image)
    except cv2.error as e:
        print(f"保存图像时出错：{e}")


if __name__ == "__main__":
    # 图片路径 --- 请替换为您的图片路径 ---
    image_path = 'C:\\PICTURE\\IMG3.png'

    # 打印当前工作目录，方便调试
    print(f"当前工作目录：{os.getcwd()}")

    # 创建 'data' 文件夹，如果它不存在
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 将 haarcascade_frontalface_default.xml 复制到 'data' 文件夹 (仅在文件不在该文件夹时执行)
    cascade_file = 'data/haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_file):
        source_cascade = 'C:\\Users\\木木波\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
        try:
            os.rename(source_cascade, cascade_file)  # 尝试重命名，如果目标已存在则会失败
        except FileExistsError:
            print(f"警告：'haarcascade_frontalface_default.xml' 已存在于 'data' 文件夹。")
        except FileNotFoundError:
            print(f"错误：找不到源文件：'{source_cascade}'")
        except Exception as e:
            print(f"复制文件时出现未知错误：{e}")


    try:
        # 调用函数并指定缩放比例，例如 0.7 表示缩小为原来的 70%
        result_image = viola_jones_face_detection(image_path, scale_factor=0.24)
        if result_image is not None:
            output_dir = 'C:\\PICTURE\\3\\5'  # 输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # 创建输出目录，如果不存在
            save_path = os.path.join(output_dir, 'Viola-Jones 人脸检测结果.png')
            print(f"保存路径为: {save_path}")  # 打印保存路径，方便检查
            save_image(result_image, save_path)
    except Exception as e:
        print(f"程序执行出错：{e}")