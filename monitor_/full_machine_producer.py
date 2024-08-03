from object_detector import *
from video_processor import *
import time
from minio_manager import *
import logging

logging.basicConfig(level=logging.DEBUG)#,filename='full_machine_consumer.log',filemode='w')
logger=logging.getLogger(__name__)


class FullMachineProducer:
    def __init__(self, db_config, kafka_config, minio_config, polling):
        self.kafka_config = kafka_config

        # fetch all streams from db
        self.video_provider = VideoStreamInfoProvider(db_config)
        self.video_streams = self.video_provider.get_all_video_streams_info()

        # set switch interval
        self.switch_interval = polling/len(self.video_streams)
        self.kafka_producer_video_info = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic1'])
        self.minio = MinioImageManager(minio_config['endpoint'], minio_config['access_key'],
                                       minio_config['secret_key'], minio_config['secure'])

    def run(self):
        for video_stream in self.video_streams:
            frame_extractor = VideoFrameExtractor()
            capture_time, encoded_frame = frame_extractor.get_encoded_frame(video_stream['stream_url'])
            if video_stream['type'] == 1:
                self.minio.put_image('videotype1', video_stream['stream_name'], encoded_frame, capture_time.strftime("%Y-%m-%d %H:%M:%S"))
            elif video_stream['type'] == 2:
                self.minio.put_image('videotype2', video_stream['stream_name'], encoded_frame, capture_time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            while True:
                for video_stream in self.video_streams:
                    stream_name = video_stream['stream_name']
                    stream_url = video_stream['stream_url']
                    video_type = video_stream['type']
                    frame_extractor = VideoFrameExtractor()
                    capture_time, encoded_frame = frame_extractor.get_encoded_frame(stream_url)
                    self.kafka_producer_video_info.send_video_info(stream_name, stream_url, video_type, capture_time, encoded_frame)
                    time.sleep(self.switch_interval)
        except KeyboardInterrupt:
            logger.info('Received keyboard interrupt signal, exiting...')

if __name__ == '__main__':
    logger.info('start')
    db_config = {
         'host': '192.168.31.112',
         'user': 'root',
         'password': 'my-secret-pw',
         'database': 'mydb1'
     }

    kafka_config = {
         'brokers': '192.168.31.112:9092',
         'topic1' : 'video_stream',
         'topic2' : 'alarm'
    }

    minio_config = {
        'endpoint': '127.0.0.1:9000',
        'access_key': 'minio',
        'secret_key': 'miniosecret',
        'secure': False,
        'bucket_name1': 'videotype1',
        'bucket_name2': 'videotype2'
    }

    producer = FullMachineProducer(db_config, kafka_config, minio_config, 60)
    producer.run()
