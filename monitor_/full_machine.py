from image_comparator import *
from object_detector import *
from video_processor import *
import time



class FullMachine:
    def __init__(self, db_config, kafka_config, minio_config, polling_time):
        self.kafka_config = kafka_config
        self.minio_config = minio_config
        self.running = False

        # fetch all streams from db
        self.video_provider = VideoStreamInfoProvider(db_config)
        self.video_streams = self.video_provider.get_all_video_streams_info()
        # set switch interval
        self.switch_interval = polling_time/len(self.video_streams)
        self.kafka_producer = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic'])w
        self.kafka_receiver = KafkaMessageReceiver(kafka_config['brokers'], kafka_config['topic'])
        self.detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        while self.running:
            for video_stream in self.video_streams:
                stream_name = video_stream['stream_name']
                stream_url = video_stream['stream_url']
                video_type = video_stream['video_type']
                frame_extractor = VideoFrameExtractor()
                capture_time, encoded_frame = frame_extractor.get_encoded_frame(stream_url)
                self.kafka_producer.send_video_info(stream_name, stream_url, capture_time, encoded_frame)
                # 判断类型，进行compare/detect
                message = self.kafka_receiver.receive_message()
                if message is not None:
                    video_info, jpeg_data = message
                    # video_info = {'stream': video_name, 'url': video_url, 'type' : video_type, 'capture_time': capture_time.strftime("%Y-%m-%d %H:%M:%S")}
                    if video_info['type'] == '1':
                        target_label = 'person'
                        detection_results_person = self.detector.detect_object(jpeg_data, target_label)
                        if detection_results_person is not None:
                            self.detector.show_detections(jpeg_data, target_label)
                            # output_path = 'output.jpg'
                            # self.detector.save_detections(jpeg_data, target_label, output_path)
                            self.detector.save_detections_to_minio(video_info, jpeg_data, self.minio_config)

                        target_label = 'car'
                        detection_results_car = self.detector.detect_object(jpeg_data, target_label)
                        if detection_results_car is not None:
                            self.detector.show_detections(jpeg_data, target_label)
                            # output_path = 'output.jpg'
                            # self.detector.save_detections(jpeg_data, target_label, output_path)
                            self.detector.save_detections_to_minio(video_info, jpeg_data, self.minio_config)

                        #发送警报
                    elif video_info['type'] == '2':
                        #从minio获取上一张，比较


                else:
                    print("No message received.")

                time.sleep(self.switch_interval)
            time.sleep(self.switch_interval)

    def stop(self):
        self.running = False
        self.kafka_producer.close()
        self.kafka_receiver.close()







