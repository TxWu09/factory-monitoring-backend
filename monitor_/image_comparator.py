import cv2
import os


class ImageComparator:
    def __init__(self, path1, path2):
        """
        Initialize the comparator and process the images.

        :param path1: Path of the first image.
        :param path2: Path of the second image.
        """
        if not self.is_valid_path(path1) or not self.is_valid_path(path2):
            raise ValueError("Invalid image path(s) provided.")

        try:
            # 灰度处理
            self.image1 = self.process_image(cv2.imread(path1, cv2.IMREAD_GRAYSCALE), "image1")
            self.image2 = self.process_image(cv2.imread(path2, cv2.IMREAD_GRAYSCALE), "image2")
        except cv2.error as e:
            raise IOError(f"Failed to read the image(s): {e}")

    def process_image(self, image, image_name):
        """
        Process a single image (Histogram Equalization and Gaussian Blur).

        :param image: The image to process.
        :param image_name: The name of the image for error reporting.
        :return: The processed image.
        """
        if image is None:
            raise IOError(f"Failed to read the {image_name} image.")

        # Histogram Equalization
        image = cv2.equalizeHist(image)

        # Gaussian Blur for noise reduction
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def compare_images(self, threshold_percent):
        """
        Compare two images to determine if they differ by more than the specified threshold.

        :param threshold_percent: The percentage of differing pixels allowed.
        :return: True if the images are similar enough, False otherwise.
        """
        diff = cv2.absdiff(self.image1, self.image2)
        differing_pixels = cv2.countNonZero(diff)
        total_pixels = self.image1.shape[0] * self.image1.shape[1]

        if total_pixels == 0:
            raise ValueError("Both images must have at least one pixel.")

        percent_differing = (differing_pixels / total_pixels) * 100
        return percent_differing <= threshold_percent

    @staticmethod
    def is_valid_path(path):
        """
        Validate the path to ensure it's safe and legal. This is a placeholder method.

        :param path: The path to validate.
        :return: True if the path is valid, False otherwise.
        """
        return os.path.isabs(path)

