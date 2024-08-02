from image_comparator import *
from object_detector import *
import time
from minio_manager import *


class FullMachineConsumer:
    def __init__(self, kafka_config, minio_config, comparing_threshold):
        self.kafka_config = kafka_config
        self.minio_config = minio_config
        self.comparing_threshold = comparing_threshold
        self.detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

        self.kafka_receiver_video_info = KafkaMessageReceiver(kafka_config['brokers'], kafka_config['topic1'])
        self.minio = MinioImageManager(minio_config['endpoint'], minio_config['access_key'], minio_config['secret_key'], minio_config['secure'])

    def run(self):
        while True:
            message = self.kafka_receiver_video_info.receive_message()
            if message is not None:
                video_info, jpeg_data = message
                if video_info['type'] == 1:
                    print(video_info['type'])
                    print(video_info['stream'])
                    print("detection-type1")
                    if not self.minio.check_image_exists(self.minio_config['bucket_name1'], video_info['stream']):
                        self.minio.put_image(self.minio_config['bucket_name1'], video_info['stream'], jpeg_data, video_info['capture_time'])

                    target_label = 'person'
                    detection_results_person = self.detector.detect_object(jpeg_data, target_label)
                    if detection_results_person is not None:
                        # self.detector.show_detections(jpeg_data, target_label)
                        # output_path = 'output.jpg'
                        # self.detector.save_detections(jpeg_data, target_label, output_path)
                        self.detector.save_detections_to_minio(video_info, jpeg_data, self.minio_config)
                        # self.kafka_producer_alarm.send_alarm(stream_name, 1)

                    target_label = 'car'
                    detection_results_car = self.detector.detect_object(jpeg_data, target_label)
                    if detection_results_car is not None:
                        self.detector.show_detections(jpeg_data, target_label)
                        self.detector.save_detections_to_minio(video_info, jpeg_data, self.minio_config)
                        # self.kafka_producer_alarm.send_alarm(stream_name, 2)

                elif video_info['type'] == 2:
                    print("comparasion-type2")
                    print(video_info['type'])
                    print(video_info['stream'])
                    if self.minio.check_image_exists(self.minio_config['bucket_name2'], video_info['stream']):
                        image1 = self.minio.get_image(self.minio_config['bucket_name2'], video_info['stream'])
                        image2 = jpeg_data
                        comparator = ImageComparator()
                        if comparator.compare_images(image1, image2, self.comparing_threshold):
                            continue
                        else:
                            self.minio.put_image(self.minio_config['bucket_name2'], video_info['stream'], image2, video_info['capture_time'])
                            # self.kafka_producer_alarm.send_alarm(stream_name, 3)
                    else:
                        self.minio.put_image(self.minio_config['bucket_name2'], video_info['stream'], jpeg_data, video_info['capture_time'])

                else:
                    print("No message received.")

    # if no message, sleep()
    # try catch keyboard interrupt