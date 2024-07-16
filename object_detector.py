import torch
from PIL import Image
import numpy as np
import pandas as pd


class YOLOv5Detector:
    def __init__(self, model_path='yolov5l.pt', device=None):
        """
        Initialize the object detector with a YOLOv5 model.

        :param model_path: Path to the YOLOv5 model file.
        :param device: Device to run the model on (e.g., 'cuda' for GPU).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not model_path.endswith('.pt'):
            raise ValueError("Invalid model path, must be a .pt file.")

        self.device = device
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")


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
            print(f"Error opening image at {image_path}")
            return pd.DataFrame()

        try:
            results = self.model(img)
        except Exception as e:
            print(f"Error during inference: {e}")
            return pd.DataFrame()

        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[filtered_results['name'] == target_label]

        return filtered_results


    def show_detections(self, image_path, target_label):
        """
        Display the image with bounding boxes around detected objects.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        """
        results = self.detect_object(image_path, target_label)
        if results.empty:
            print("No detections found.")
            return

        try:
            results.show()
        except Exception as e:
            print(f"Error showing detections: {e}")


    def save_detections(self, image_path, target_label, output_path):
        """
        Save the image with bounding boxes around detected objects to the specified path,
        without modifying the original image.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        :param output_path: Path to save labelled image
        """
        results = self.detect_object(image_path, target_label)
        if results.empty:
            print("No detections found. Skipping save.")
            return

        try:
            results = self.model(image_path)
            filtered_results = results.pandas().xyxy[0]
            filtered_results = filtered_results[filtered_results['name'] == target_label]

            # Render detections on the original image
            results.render()

            # Save the image with detections
            Image.fromarray(results.ims[0]).save(output_path)
        except Exception as e:
            print(f"Error saving detections to {output_path}: {e}")
