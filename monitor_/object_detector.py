import torch
import warnings
from ultralytics import YOLO
import io
from confluent_kafka import Consumer, KafkaError
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
        # boxes = results.boxes
        # filtered_boxes = boxes.to_pandas()
        #
        # # Filter the results by the target label.
        # filtered_results = filtered_boxes[filtered_boxes['name'] == target_label]
        #
        # return filtered_results
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        names = results.names

        # Create a DataFrame from the data.
        df = pd.DataFrame({
            'xmin': xyxy[:, 0],
            'ymin': xyxy[:, 1],
            'xmax': xyxy[:, 2],
            'ymax': xyxy[:, 3],
            'confidence': conf,
            'class': cls,
            'name': [names[int(c)] for c in cls]
        })

        # Filter the results by the target label.
        filtered_results = df[df['name'] == target_label]

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
        print(results)
        return self._filter_results(results[0], target_label)

    def show_detections(self, image_path, target_label):
        img = Image.open(image_path)
        results = self.model(image_path)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Detections for {target_label}")
        plt.axis('off')
        plt.show()

    def save_detections(self, image_path, target_label, output_path):
        img = Image.open(image_path)
        results = self.model(image_path)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        img.save(output_path)

    def _draw_boxes(self, img, filtered_results):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", size=15)  # You might need to specify the correct font path.
        for index, row in filtered_results.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            confidence = row['confidence']
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=10)
            draw.text((xmin, ymin - 10), f"{row['name']} {confidence:.2f}", fill="red", font=font)


    # def show_detections(self, image_path, target_label):
    #     """
    #     Display the image with bounding boxes around detected objects.
    #
    #     :param image_path: Path to the image file.
    #     :param target_label: Label of the object to detect.
    #     """
    #     results = self.detect_object(image_path, target_label)
    #     if results is not None:
    #         results[0].plot()
    #
    # def save_detections(self, image_path, target_label, output_path):
    #     """
    #     Save the image with bounding boxes around detected objects to the specified path.
    #
    #     :param image_path: Path to the input image file.
    #     :param target_label: Label of the object to detect.
    #     :param output_path: Path to save the output image.
    #     """
    #     results = self.detect_object(image_path, target_label)
    #     if results is None:
    #         return
    #
    #     if not any(result['name'] == target_label for result in results.xyxy[0]):
    #         warnings.warn(f"No detections for target label {target_label}. Skipping rendering.")
    #         return
    #
    #     try:
    #         # Render detections on the original image
    #         results.render()
    #
    #         # Save the image with detections
    #         Image.fromarray(results.ims[0]).save(output_path)
    #     except Exception as e:
    #         warnings.warn(f"Failed to save the detections to {output_path}: {e}")

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
        'bootstrap.servers': kafka_brokers,
        'group.id': 'my_group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([topic])

    try:
        while True:
            try:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print('End of partition reached')
                    else:
                        print(f'Error: {msg.error()}')
                    continue
                yield msg
            except KeyboardInterrupt:
                print("Received keyboard interrupt, exiting...")
                break
    finally:
        consumer.close()


if __name__ == '__main__':
    # #TEST: Consumer
    # consumer = receive_kafka_messages('192.168.31.112:9092', 'video_stream')
    # for msg in consumer:
    #     if msg.error():
    #         print(f"Consumer error: {msg.error()}")
    #         continue
    #
    #     message_key = msg.key().decode('utf-8') if msg.key() is not None else None
    #     message_value = msg.value()
    #     video_info = json.loads(message_key) if message_key is not None else {}
    #     video_name = video_info.get('stream')
    #     video_url = video_info.get('url')
    #     print(f'Received message: key={video_info}, value={message_value[:10]} bytes...')

    # TEST: detector
    detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
    target_label = 'person'
    image_path = 'example.jpg'
    detection_results = detector.detect_object(image_path, target_label)
    if detection_results is not None:
        detector.show_detections(image_path, target_label)
    output_path = 'output.jpg'
    detector.save_detections(image_path, target_label, output_path)

