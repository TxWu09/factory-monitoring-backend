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
        """
        Load a JPEG image from bytes and convert it to grayscale.
        :param jpeg_bytes: JPEG image data as bytes.
        :return: Grayscale image as a NumPy array.
        """
        pil_image = Image.open(io.BytesIO(jpeg_bytes)).convert('L')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2BGR)

    def compare_images(self, jpeg_bytes1, jpeg_bytes2, threshold_percent):
        """
        Compare two JPEG images loaded from bytes to determine if they differ by more than the specified threshold.
        :param jpeg_bytes1: First JPEG image data as bytes.
        :param jpeg_bytes2: Second JPEG image data as bytes.
        :param threshold_percent: The percentage of differing pixels allowed.
        :return: True if the images are similar enough, False otherwise.
        """
        # Load images from bytes and convert them to grayscale
        image1 = self.load_jpeg_image(jpeg_bytes1)
        image2 = self.load_jpeg_image(jpeg_bytes2)

        # Apply histogram equalization and Gaussian blur
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image1_processed = cv2.equalizeHist(image1_gray)
        image2_processed = cv2.equalizeHist(image2_gray)
        image1_processed = cv2.GaussianBlur(image1_processed, (5, 5), 0)
        image2_processed = cv2.GaussianBlur(image2_processed, (5, 5), 0)

        # Compute difference
        diff = cv2.absdiff(image1_processed, image2_processed)
        differing_pixels = cv2.countNonZero(diff)
        total_pixels = image1_processed.shape[0] * image1_processed.shape[1]

        if total_pixels == 0:
            raise ValueError("Both images must have at least one pixel.")

        percent_differing = (differing_pixels / total_pixels) * 100
        return percent_differing <= threshold_percent



#
# if __name__ == '__main__':
#    minio = MinioImageManager('127.0.0.1:9000','minio','miniosecret',False)
#    print(minio.check_image_exists('videotype1', 'example'))
#    minio.put_image('videotype1', 'example', b'123', '2023-07-01 12:00:00')