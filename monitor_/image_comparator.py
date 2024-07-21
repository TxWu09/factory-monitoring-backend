from PIL import Image
import numpy as np
from PIL import ImageEnhance, ImageFilter, ImageOps

class ImageComparator:
    def __init__(self, path1, path2):
        self.image1 = Image.open(path1)
        self.image2 = Image.open(path2)

        # grayscale
        self.gray_image1 = self.image1.convert('L')
        self.gray_image2 = self.image2.convert('L')

        # Histogram Equalization
        self.image1 = ImageOps.equalize(self.image1)
        self.image2 = ImageOps.equalize(self.image2)

        # Gaussian Blur for noise reduction
        self.image1 = self.image1.filter(ImageFilter.GaussianBlur(radius=1))
        self.image2 = self.image2.filter(ImageFilter.GaussianBlur(radius=1))


    def compare_images(self, threshold_percent):
        img1_data = np.array(self.gray_image1)
        img2_data = np.array(self.gray_image2)

        # absolute difference
        diff = np.abs(img1_data - img2_data)
        total_pixels = img1_data.size
        differing_pixels = np.count_nonzero(diff > 0)

        # percentage difference
        percent_differing = (differing_pixels / total_pixels) * 100
        return percent_differing <= threshold_percent