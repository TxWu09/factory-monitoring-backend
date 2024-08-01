from image_comparator import *
from object_detector import *
from video_processor import *
import time


# 传带框的图
# 中止进程
# 清空队列

class FullMachine:
    def __init__(self, db_config, kafka_config, minio_config, polling_time, comparing_threshold):
        self.kafka_config = kafka_config
        self.minio_config = minio_config
        self.running = True
        self.comparing_threshold = comparing_threshold

        # fetch all streams from db
        self.video_provider = VideoStreamInfoProvider(db_config)
        self.video_streams = self.video_provider.get_all_video_streams_info()

        # set switch interval
        self.switch_interval = polling_time/len(self.video_streams)
        self.kafka_producer_video_info = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic1'])
        self.kafka_receiver_video_info = KafkaMessageReceiver(kafka_config['brokers'], kafka_config['topic1'])
        # self.kafka_producer_alarm = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic2'])
        self.detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.minio = MinioImageManager(minio_config['endpoint'], minio_config['access_key'], minio_config['secret_key'], minio_config['secure'])

    def run(self):
        while self.running:
            for video_stream in self.video_streams:
                stream_name = video_stream['stream_name']
                stream_url = video_stream['stream_url']
                video_type = video_stream['type']
                frame_extractor = VideoFrameExtractor()
                capture_time, encoded_frame = frame_extractor.get_encoded_frame(stream_url)
                self.kafka_producer_video_info.send_video_info(stream_name, stream_url, video_type, capture_time, encoded_frame)
                # 判断类型，进行compare/detect
                message = self.kafka_receiver_video_info.receive_message()
                if message is not None:
                    video_info, jpeg_data = message
                    if video_info['type'] == 1:
                        print(video_info['type'])
                        print(video_info['stream'])
                        print("detection-type1")
                        print(self.minio.check_image_exists(self.minio_config['bucket_name1'], video_info['stream']))
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
                        print(self.minio.check_image_exists(self.minio_config['bucket_name2'], video_info['stream']))
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

                time.sleep(self.switch_interval)
            time.sleep(self.switch_interval)

    def stop(self):
        self.running = False
        self.kafka_producer_video_info.close()
        self.kafka_receiver_video_info.close()
        # self.kafka_producer_alarm.close()

if __name__ == '__main__':
    db_config = {
        'host': '192.168.31.112',
        'user': 'root',
        'password': 'my-secret-pw',
        'database': 'mydb1'
    }


    kafka_config = {
        'brokers': '192.168.31.112:9092',
        'topic1': 'video_stream',
        'topic2': 'alarm'
    }

    minio_config = {
        'endpoint': '127.0.0.1:9000',
        'access_key': 'minio',
        'secret_key': 'miniosecret',
        'secure': False,
        'bucket_name1': 'videotype1',
        'bucket_name2': 'videotype2'
    }

    polling_time = 60
    comparing_threshold = 5
    full_machine = FullMachine(db_config, kafka_config, minio_config, polling_time, comparing_threshold)
    # full_machine.stop()
    full_machine.run()
    time.sleep(300)
    full_machine.stop()










