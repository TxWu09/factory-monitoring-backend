from image_comparator import *
from object_detector import *
import time
from minio_manager import *
import logging
from video_processor import *


logging.basicConfig(level=logging.DEBUG)#,filename='full_machine_consumer.log',filemode='w')
logger=logging.getLogger(__name__)


class FullMachineConsumer:
    def __init__(self, kafka_config, minio_config, comparing_threshold):
        self.kafka_config = kafka_config
        self.minio_config = minio_config
        self.comparing_threshold = comparing_threshold
        self.detector = YOLOv5Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

        self.kafka_producer_alarm = VideoStreamProducer(kafka_config['brokers'], kafka_config['topic2'])
        self.kafka_receiver_video_info = KafkaMessageReceiver(kafka_config['brokers'], kafka_config['topic1'])
        self.minio = MinioImageManager(minio_config['endpoint'], minio_config['access_key'],
                                       minio_config['secret_key'], minio_config['secure'])

    def run(self):
        try:
            while True:
                message = self.kafka_receiver_video_info.receive_message()
                if message is not None:
                    video_info, jpeg_data = message
                    if video_info['type'] == 1:
                        target_label = ['person', 'car']
                        detection_results, detected_labels, annotated_image = self.detector.detect_object(jpeg_data, target_label)
                        if detected_labels:
                            self.detector.save_detections_to_minio(video_info, annotated_image, detected_labels, self.minio_config)
                            if detected_labels == ['person']:
                                self.kafka_producer_alarm.send_alarm(video_info['stream'], video_info['capture_time'], 1)
                            elif detected_labels == ['car']:
                                self.kafka_producer_alarm.send_alarm(video_info['stream'], video_info['capture_time'],2)
                            elif sorted(detected_labels) == sorted(['person', 'car']):
                                self.kafka_producer_alarm.send_alarm(video_info['stream'], video_info['capture_time'],3)

                    elif video_info['type'] == 2:
                        logger.debug("comparasion-type2")
                        logger.debug(video_info['type'])
                        logger.debug(video_info['stream'])

                        image1 = self.minio.get_image(self.minio_config['bucket_name2'], video_info['stream'])
                        image2 = jpeg_data
                        comparator = ImageComparator()
                        logger.debug("comparestarts")
                        if comparator.compare_images(image1, image2, self.comparing_threshold):
                            logger.debug("no difference")
                            continue
                        else:
                            logger.debug("difference")
                            self.minio.put_different_image(self.minio_config['bucket_name2'], video_info['stream'], image2, video_info['capture_time'])
                            self.kafka_producer_alarm.send_alarm(video_info['stream'], video_info['capture_time'],4)
                else:
                    logger.debug("No message received.")
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info('Received keyboard interrupt signal, exiting...')


if __name__ == '__main__':
    minio_config = {
        'endpoint': '127.0.0.1:9000',
        'access_key': 'minio',
        'secret_key': 'miniosecret',
        'secure': False,
        'bucket_name1': 'videotype1',
        'bucket_name2': 'videotype2'
    }
    kafka_config = {
        'brokers': '192.168.31.112:9092',
        'topic1': 'video_stream',
        'topic2': 'alarm'
    }

    consumer = FullMachineConsumer(kafka_config, minio_config, 0.5)
    consumer.run()