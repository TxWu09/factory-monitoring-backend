import torch
import warnings
from ultralytics import YOLO
import io
from confluent_kafka import Consumer, KafkaError
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import minio
from minio.error import S3Error
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


    def detect_object_path(self, image_path, target_label):
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

        try:
            results = self.model(image_path)
            return self._filter_results(results[0], target_label)
        except Exception as e:
            warnings.warn(f"Failed to process the image: {e}")
            return None

    def detect_object(self, img, target_label):
        """
        Detect a specific type of object in the given image.

        :param img: A PIL Image object or JPEG image data.
        :param target_label: Label of the object to detect.
        :return: Detection results for the target label.
        """
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))

        try:
            results = self.model(img)
            return self._filter_results(results[0], target_label)
        except Exception as e:
            warnings.warn(f"Failed to process the image: {e}")
            return None

    def show_detections_path(self, image_path, target_label):
        """
        Show detections on the given image.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        """
        img = Image.open(image_path)
        results = self.model(image_path)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Detections for {target_label}")
        plt.axis('off')
        plt.show()

    def show_detections(self, img, target_label):
        """
        Show detections on the given image.

        :param img: A PIL Image object or JPEG image data.
        :param target_label: Label of the object to detect.
        """
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))

        results = self.model(img)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Detections for {target_label}")
        plt.axis('off')
        plt.show()

    def _draw_boxes(self, img, filtered_results):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", size=15)  # You might need to specify the correct font path.
        for index, row in filtered_results.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            confidence = row['confidence']
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=10)
            draw.text((xmin, ymin - 10), f"{row['name']} {confidence:.2f}", fill="red", font=font)

    def save_detections_path(self, image_path, target_label, output_path):
        """
        Save the image with detections to the specified path.

        :param image_path: Path to the image file.
        :param target_label: Label of the object to detect.
        :param output_path: Path to save the annotated image.
        """
        img = Image.open(image_path)
        results = self.model(image_path)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        img.save(output_path)

    def save_detections(self, img, target_label, output_path):
        """
        Save the image with detections to the specified path.

        :param img: A PIL Image object or JPEG image data.
        :param target_label: Label of the object to detect.
        :param output_path: Path to save the annotated image.
        """
        if isinstance(img, bytes):
            # Convert JPEG data to PIL Image
            img = Image.open(io.BytesIO(img))

        results = self.model(img)
        filtered_results = self._filter_results(results[0], target_label)
        self._draw_boxes(img, filtered_results)
        img.save(output_path)

    def save_detections_to_minio(self, video_info, img, minio_config):
        """
        Save the image with detections and related URL information as metadata to Minio.

        :param video_info: A dictionary containing video information including 'stream', 'url', and 'capture_time'.
        :param img: The image data (either as a PIL Image object or JPEG bytes).
        :param minio_config: A dictionary containing Minio configuration including 'endpoint', 'access_key', 'secret_key', and 'bucket_name'.
        """
        # Extract necessary information from video_info
        video_name = video_info['stream']
        capture_time_str = video_info['capture_time']
        video_url = video_info['url']

        # Initialize Minio client
        minio_client = minio.Minio(
            minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            secure=False # Set to False if your Minio server is not using SSL
        )

        # Construct the image file name
        image_name = f"{video_name}.jpg"

        # Convert image data to bytes if it's a PIL Image object
        if isinstance(img, Image.Image):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img = img_byte_arr.getvalue()

        try:
            # Prepare the metadata
            metadata = {
                'capture-time': capture_time_str
            }

            # Upload the image to Minio with metadata
            minio_client.put_object(
                minio_config['bucket_name'],
                image_name,
                io.BytesIO(img),
                len(img),
                content_type='image/jpeg',
                metadata=metadata
            )
        except S3Error as e:
            print(f"Error occurred while uploading to Minio: {e}")

    # def load_detections(self):
#命名需要包含摄像头名称/截图的时间
#修改消息队列，传入截图时间

class KafkaMessageReceiver:
    def __init__(self, brokers, topic):
        """
        Initialize the Kafka message receiver.

        :param brokers: Kafka broker addresses.
        :param topic: Kafka topic name.
        """
        self.brokers = brokers
        self.topic = topic

        # Initialize the Kafka consumer
        self.consumer = Consumer({
            'bootstrap.servers': self.brokers,
            'group.id': 'my_group',
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([self.topic])

    def receive_message(self):
        msg = self.consumer.poll(1.0)
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print('End of partition reached')
            else:
                print(f'Error: {msg.error()}')
            return None

        # Decode the message
        video_info = eval(msg.key().decode('utf-8'))
        jpeg_data = msg.value()

        # Return the decoded message
        return (video_info, jpeg_data)

    def close(self):
        self.consumer.close()


# if __name__ == '__main__':
#     kafka_receiver = KafkaMessageReceiver(
#         brokers='192.168.31.112:9092',
#         topic='video_stream'
#     )
#     # Minio configuration
#     minio_config = {
#         'endpoint': '127.0.0.1:9000',
#         'access_key': 'minio',
#         'secret_key': 'miniosecret',
#         'bucket_name': 'videos'
#     }
#
#     message = kafka_receiver.receive_message()
#
#     # Check if a message was received
#     if message is not None:
#         video_info, jpeg_data = message
#         print("Received message:")
#         print("Video Info:", video_info)
#         print("Jpeg Data Length:", len(jpeg_data))
#
#
#         # TEST: detector
#         detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
#         target_label = 'person'
#         detection_results = detector.detect_object(jpeg_data, target_label)
#
#         if detection_results is not None:
#             detector.show_detections(jpeg_data, target_label)
#             output_path = 'output.jpg'
#             detector.save_detections(jpeg_data, target_label, output_path)
#             detector.save_detections_to_minio(video_info, jpeg_data, minio_config)
#             print("end")
#     else:
#         print("No message received.")
#
#     kafka_receiver.close()






