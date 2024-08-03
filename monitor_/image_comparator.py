import cv2
from minio import Minio
from minio.error import S3Error
from PIL import Image
import numpy as np
import io


class ImageComparator:
    def __init__(self):
        pass

    def load_jpeg_image(self, jpeg_bytes):
        pil_image = Image.open(io.BytesIO(jpeg_bytes)).convert('L')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2BGR)

    # def compare_images(self, jpeg_bytes1, jpeg_bytes2, threshold_percent):
    #     """
    #     Compare two JPEG images loaded from bytes to determine if they differ by more than the specified threshold.
    #     :param jpeg_bytes1: First JPEG image data as bytes.
    #     :param jpeg_bytes2: Second JPEG image data as bytes.
    #     :param threshold_percent: The percentage of differing pixels allowed.
    #     :return: True if the images are similar enough, False otherwise.
    #     """
    #     # Load images from bytes and convert them to grayscale
    #     image1 = self.load_jpeg_image(jpeg_bytes1)
    #     image2 = self.load_jpeg_image(jpeg_bytes2)
    #
    #     # Apply histogram equalization and Gaussian blur
    #     image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #     image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #     image1_processed = cv2.equalizeHist(image1_gray)
    #     image2_processed = cv2.equalizeHist(image2_gray)
    #     image1_processed = cv2.GaussianBlur(image1_processed, (5, 5), 0)
    #     image2_processed = cv2.GaussianBlur(image2_processed, (5, 5), 0)
    #
    #     # Compute difference
    #     diff = cv2.absdiff(image1_processed,   image2_processed)
    #     differing_pixels = cv2.countNonZero(diff)
    #     total_pixels = image1_processed.shape[0] * image1_processed.shape[1]
    #
    #     if total_pixels == 0:
    #         raise ValueError("Both images must have at least one pixel.")
    #
    #     percent_differing = (differing_pixels / total_pixels) * 100
    #     return percent_differing <= threshold_percent


    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    def compare_images(self, image1, image2, area_threshold_percent=5):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        # 使用GMM背景减除器处理图像
        fg_mask1 = bg_subtractor.apply(image1)
        fg_mask2 = bg_subtractor.apply(image2)

        # 计算差异图像
        diff = cv2.absdiff(fg_mask1, fg_mask2)

        # 进行二值化
        _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算图像总面积
        total_area = image1.shape[0] * image1.shape[1]
        area_threshold = total_area * (area_threshold_percent / 100)

        # 筛选面积大于阈值的轮廓
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

        # 绘制轮廓
        result = image1.copy()
        cv2.drawContours(result, large_contours, -1, (0, 255, 0), 2)

        return result

if __name__ == '__main__':
   comparator = ImageComparator()
   with open('example_sunny.jpg', 'rb') as file:
       image1 = file.read()
   with open('example_time_1.jpg', 'rb') as file:
       image2 = file.read()

   print(comparator.compare_images(image1, image2, 90))