#视频拉流
#参数 (kafka_brokers topic

#先截屏再传，yolo使用示例
import json
from confluent_kafka import Producer, KafkaError
import logging
import cv2


class VideoStreamProducer:
    def __init__(self, kafka_brokers, topic, config=None):
        """
        初始化视频流生产者。

        :param kafka_brokers: Kafka代理的地址。
        :param topic: 发送消息的目标主题。
        :param config: Kafka Producer的配置字典，可选。
        """
        # 使用提供的配置参数，如果没有提供，则使用默认配置
        producer_config = {
            'bootstrap.servers': kafka_brokers,
            'acks': 'all',  # 确保所有ISR都确认收到消息
            'retries': 3,  # 在发送失败时重试多次
            # 更多配置根据需要添加...
        }
        if config:
            producer_config.update(config)

        self.producer = Producer(producer_config)
        self.topic = topic
        self.logger = logging.getLogger(__name__)

    def encode_image(self, frame):
        """
        将OpenCV图像帧编码为JPEG格式的字节串。
        """
        _, encoded_frame = cv2.imencode('.jpg', frame)
        return encoded_frame.tobytes()

    def send_video_info(self, video_name, stream, video_type):
        """
        构建并发送视频信息和当前帧到指定的主题。
        """
        try:
            cap = cv2.VideoCapture(stream)
            if not cap.isOpened():
                raise ValueError(f"Could not open video stream: {stream}")

            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame from video stream: {video_name}")

            encoded_frame = self.encode_image(frame)
            video_info = {'name': video_name, 'stream': stream, 'type': video_type, 'frame': encoded_frame}
            message = json.dumps(video_info).encode('utf-8')

            self.producer.produce(self.topic, value=message)
            self.producer.flush()
            cap.release()
        except KafkaError as e:
            self.logger.error(f"Failed to send video info: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

    def close(self):
        """
        关闭生产者实例。
        """
        self.producer.flush()


# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 使用示例
# producer = VideoStreamProducer('localhost:9092', 'video_stream')
# producer.send_video_info('test_video', 'test_stream', 'test_type')
# producer.close()
