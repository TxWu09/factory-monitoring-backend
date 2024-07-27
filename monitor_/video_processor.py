#视频拉流
#参数 (kafka_brokers topic

#先截屏再传，yolo使用示例


#producer传过去/consumer接收
#mySQL  docker-->端口


import cv2
import json
from confluent_kafka import Producer, KafkaError
import logging
import mysql.connector
import numpy as np



class VideoStreamInfoProvider:
    def __init__(self, db_config):
        self.db_config = db_config

    def get_video_stream_info(self, video_name):
        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()
        query = "SELECT stream_name, stream_url, type FROM videos WHERE stream_name=%s"
        cursor.execute(query, (video_name,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        if result:
            stream_name, stream_url, video_type = result
            return stream_name, stream_url, video_type
        else:
            raise ValueError(f"No video stream information found for: {video_name}")


class VideoFrameExtractor:
    def get_encoded_frame(self, stream_url):
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {stream_url}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame from video stream.")

        _, encoded_frame = cv2.imencode('.jpg', frame)
        cap.release()
        return encoded_frame.tobytes()


class VideoStreamProducer:
    def __init__(self, kafka_brokers, topic, config=None):
        producer_config = {
            'bootstrap.servers': kafka_brokers,
            'acks': 'all',
            'retries': 3,
        }
        if config:
            producer_config.update(config)

        self.producer = Producer(producer_config)
        self.topic = topic
        self.logger = logging.getLogger(__name__)

    def send_video_info(self, video_name, video_url, encoded_frame):
        try:
            video_info = {'stream': video_name, 'url': video_url}
            message_key = json.dumps(video_info)
            self.producer.produce(self.topic, value=encoded_frame, key=message_key)
            self.producer.flush()
        except KafkaError as e:
            self.logger.error(f"Failed to send video info: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

    def close(self):
        self.producer.flush()



logging.basicConfig(level=logging.INFO)



if __name__ == '__main__':

    DB_CONFIG = {
        'host': '192.168.31.112',
        'user': 'root',
        'password': 'my-secret-pw',
        'database': 'mydb1'
    }

    video_name = 'ParkingLot_Copy1'

    info_provider = VideoStreamInfoProvider(DB_CONFIG)
    stream_name, stream_url, video_type = info_provider.get_video_stream_info(video_name)
    print(stream_url)

    frame_extractor = VideoFrameExtractor()
    encoded_frame = frame_extractor.get_encoded_frame(stream_url)

    print("Encoded frame:", encoded_frame[:10])
    decoded_frame = cv2.imdecode(np.frombuffer(encoded_frame, np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow('Captured Frame', decoded_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Producer
    producer = VideoStreamProducer('192.168.31.112:9092', 'video_stream')
    producer.send_video_info(stream_name, stream_url, encoded_frame)
    print("sent")
    producer.close()



