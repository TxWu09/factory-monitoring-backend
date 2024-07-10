import unittest
from PIL import Image
from unittest.mock import MagicMock, patch

# Import the class to be tested
from object_detector import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    @patch('your_module.LoadImages')
    @patch('your_module.attempt_load')
    @patch('your_module.non_max_suppression')
    @patch('your_module.scale_coords')
    def test_detect_objects(self, mock_scale_coords, mock_non_max_suppression, mock_attempt_load, mock_load_images):
        # Set up mock objects
        mock_load_images.return_value = [(None, None, None, None)]
        mock_attempt_load.return_value = MagicMock()
        mock_non_max_suppression.return_value = [MagicMock()]
        mock_scale_coords.return_value = MagicMock()

        # Create an instance of ImageProcessor
        image_processor = ImageProcessor(img_path='test.jpg')

        # Call the detect_objects method
        result = image_processor.detect_objects()

        # Assert that the LoadImages function was called with the correct parameters
        mock_load_images.assert_called_with('test.jpg', img_size=image_processor.imgsz)

        # Assert that the attempt_load function was called with the correct parameters
        mock_attempt_load.assert_called_with('yolov5l.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Assert that the non_max_suppression function was called with the correct parameters
        mock_non_max_suppression.assert_called_with(mock_attempt_load.return_value, conf_thres=0.25, iou_thres=0.45,
                                                    classes=None, agnostic=False)

        # Assert that the scale_coords function was called with the correct parameters
        mock_scale_coords.assert_called()

        # Assert that the result is an Image object
        self.assertIsInstance(result, Image.Image)


if __name__ == '__main__':
    unittest.main()