import torch
from PIL import Image
import warnings


class ObjectDetector:
    def __init__(self, model_path='yolov5s.pt', device=None):
        """
        Initialize the object detector with a YOLOv5 model.

        :param model_path: Path to the YOLOv5 model file.
        :param device: Device to run the model on (e.g., 'cuda' for GPU).
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")

    def _filter_results(self, results, target_label):
        """
        Filter the detection results by target label.

        :param results: Detection results.
        :param target_label: Label of the object to detect.
        :return: Filtered results.
        """
        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[filtered_results['name'] == target_label]
        return filtered_results

    def detect_object(self, image_path, target_label):
        """
        Detect a specific type of object in the given image.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        :return: Detection results for the target label.
        """
        try:
            img = Image.open(image_path)
        except IOError:
            warnings.warn(f"Failed to open the image at {image_path}. Please check the path.")
            return None

        # Perform inference
        results = self.model(img)

        return self._filter_results(results, target_label)

    def show_detections(self, image_path, target_label):
        """
        Display the image with bounding boxes around detected objects.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        """
        results = self.detect_object(image_path, target_label)
        if results is not None:
            results.show()

    def save_detections(self, image_path, target_label, output_path):
        """
        Save the image with bounding boxes around detected objects to the specified path.

        :param image_path: Path to the input image file.
        :param target_label: Label of the object to detect.
        :param output_path: Path to save the output image.
        """
        results = self.detect_object(image_path, target_label)
        if results is None:
            return

        if not any(result['name'] == target_label for result in results.xyxy[0]):
            warnings.warn(f"No detections for target label {target_label}. Skipping rendering.")
            return

        try:
            # Render detections on the original image
            results.render()

            # Save the image with detections
            Image.fromarray(results.ims[0]).save(output_path)
        except Exception as e:
            warnings.warn(f"Failed to save the detections to {output_path}: {e}")
