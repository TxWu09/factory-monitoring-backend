import time

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

    def detect_different_regions(self,src_img, dst_img):
        if src_img.shape != dst_img.shape:
            # 调整图像尺寸，这里假设我们使用src_img的尺寸作为参考
            dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]))
        # 对原始图像和目标图像进行高斯模糊，以减少噪声影响
        src_img = cv2.GaussianBlur(src_img, [7, 7], 1)
        dst_img = cv2.GaussianBlur(dst_img, [7, 7], 1)

        # 计算两张图像的差异
        diff = cv2.absdiff(src_img, dst_img)

        # 转换为灰度图
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 应用阈值化，得到二值图像
        _, result = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # 对二值图像进行膨胀，突出差异区域
        result = cv2.dilate(result, np.ones([5, 5]))

        # 寻找差异区域的轮廓
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []

        # 计算轮廓面积
        for c in contours:
            area = cv2.contourArea(c)
            areas.append(area)
        areas = np.array(areas)

        # 获取面积最大的5个轮廓
        index = np.argsort(areas)[-5:]
        top5_contours = []
        rect_pos = []

        # 提取前5个轮廓，并获取其边界矩形的坐标
        for i in range(5):
            top5_contours.append(contours[index[i]])
        for c in top5_contours:
            # x y w h
            rect_pos.append(cv2.boundingRect(c))

        return rect_pos

    def draw_boxes(self, img, rect_pos):
        img_copy = img.copy()
        for pos in rect_pos:
            x, y, w, h = pos
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img_copy






if __name__ == '__main__':
   comparator = ImageComparator()
   image1 = cv2.imread('example_time_1.jpg')
   image2 = cv2.imread('example_time_2.jpg')


   # print(comparator.compare_images(image1, image2, 1))
   results = comparator.detect_different_regions(image1, image2)
   comparator.draw_boxes(image1, results)
   comparator.draw_boxes(image2, results)

   cv2.imshow('result1', comparator.draw_boxes(image1, results))
   cv2.imshow('result2', comparator.draw_boxes(image2, results))
   cv2.waitKey(0)

