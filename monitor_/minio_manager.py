from minio import Minio
from minio.error import S3Error
import io

class MinioImageManager:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def get_image(self, bucket_name, object_name):
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