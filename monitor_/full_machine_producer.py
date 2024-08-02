from object_detector import *
from video_processor import *
import time


class FullMachineProducer:
    def __init__(self, db_config, kafka_config, minio_config, comparing_threshold):
        self.kafka_config = kafka_config
        self.minio_config = minio_config
        self.comparing_threshold = comparing_threshold

        # fetch all streams from db
        self.video_provider = VideoStreamInfoProvider(db_config)
        self.video_streams = self.video_provider.get_all_video_streams_info()

        # set switch interval
        self.switch_interval = 60/len(self.video_streams)
        self.kafka_producer_video_info = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic1'])
        self.detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        while True:
            for video_stream in self.video_streams:
                stream_name = video_stream['stream_name']
                stream_url = video_stream['stream_url']
                video_type = video_stream['type']
                frame_extractor = VideoFrameExtractor()
                capture_time, encoded_frame = frame_extractor.get_encoded_frame(stream_url)
                self.kafka_producer_video_info.send_video_info(stream_name, stream_url, video_type, capture_time, encoded_frame)
                time.sleep(self.switch_interval)