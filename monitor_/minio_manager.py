from minio import Minio
from minio.error import S3Error
import io

class MinioImageManager:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def get_image(self, bucket_name, object_name):
        """
        从MinIO服务器上下载图片并返回字节流。
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :return: 图像的字节流
        """
        filename = object_name + '.jpg'
        try:
            response = self.client.get_object(bucket_name, filename)
            image_data = response.data
            response.close()
            return image_data
        except S3Error as err:
            print("Error occurred:", err)
            raise

    def put_image(self, bucket_name, object_name, byte_data, capture_time_str):
        """
        向MinIO服务器上传图片，覆盖同名对象。
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :param byte_data: 图片的字节数据
        :param capture_time_str: 获取时间
        """
        try:
            image_name = f"{object_name}.jpg"
            metadata = {
                'capture-time': capture_time_str
            }

            self.client.put_object(
                bucket_name=bucket_name,
                object_name=image_name,
                data=io.BytesIO(byte_data),
                length=len(byte_data),
                content_type='image/jpeg',
                metadata=metadata
            )
            print(f"Image {object_name} uploaded successfully.")
        except S3Error as err:
            print("Error occurred:", err)

    def check_image_exists(self, bucket_name, object_name):
        """
        检查MinIO中是否已存在指定的对象。
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :return: 对象存在返回True，否则返回False
        """
        print(bucket_name)
        print(object_name)
        filename = object_name + '.jpg'
        try:
            self.client.stat_object(bucket_name, filename)
            return True
        except S3Error as err:
            print("no such key")
            if err.code == 'NoSuchKey':
                return False
            else:
                raise

    def put_different_image(self, bucket_name, object_name, byte_data, capture_time_str):
        try:
            capture_time_str = capture_time_str.replace(':', '').replace(' ', '_')
            image_name = f"{object_name}_{capture_time_str}.jpg"
            metadata = {
                'capture-time': capture_time_str
            }

            self.client.put_object(
                bucket_name=bucket_name,
                object_name=image_name,
                data=io.BytesIO(byte_data),
                length=len(byte_data),
                content_type='image/jpeg',
                metadata=metadata
            )
            print(f"Image {object_name} uploaded successfully.")
        except S3Error as err:
            print("Error occurred:", err)