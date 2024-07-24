import torch
from PIL import Image
import warnings
from ultralytics import YOLO
import io
from confluent_kafka import Consumer, KafkaError

class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt', device=None):
        """
        Initialize the object detector with a YOLOv5 model.

        :param model_path: Path to the YOLOv5 model file.
        :param device: Device to run the model on (e.g., 'cuda' for GPU).
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = YOLO(model_path)
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
        results = self.model(image_path)
        return self._filter_results(results, target_label)

    def show_detections(self, image_path, target_label):
        """
        Display the image with bounding boxes around detected objects.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        """
        results = self.detect_object(image_path, target_label)
        if results is not None:
            results[0].plot()

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

    def detect_from_kafka_message(self, kafka_message, target_label):
        """
        Decode the image from a Kafka message and perform object detection.

        :param kafka_message: A Kafka message containing the image data.
        :param target_label: Label of the object to detect.
        :return: Detection results for the target label.
        """
        try:
            # Extract the image data from the Kafka message
            image_data = kafka_message.value()

            # Decode the JPEG image
            image = Image.open(io.BytesIO(image_data))

            # Perform detection
            return self.detect_object(image, target_label)
        except Exception as e:
            warnings.warn(f"Failed to process Kafka message: {e}")
            return None


def receive_kafka_messages(kafka_brokers, topic):
    consumer = Consumer({
        'bootstrap.servers': '192.168.31.112:9092',
        'group.id': 'my_group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([topic])
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print('End of partition reached')
            else:
                print('Error: {}'.format())
                continue
                yield msg.value()
                consumer.close()
                return
            yield msg.value()
            consumer.close()
            return
        yield msg.value()
        consumer.close()
        return


if __name__ == '__main__':
    consumer = receive_kafka_messages('192.168.31.112:9092', 'test')
    for message in consumer:
        print(message)



