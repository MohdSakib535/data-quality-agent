import logging
import os
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


class ObjectStorageError(RuntimeError):
    pass


def raw_upload_key(job_id: str) -> str:
    return f"uploads/raw_data/{job_id}.csv"


def cleaned_output_key(job_id: str) -> str:
    return f"outputs/{job_id}_cleaned.csv"


class ObjectStorageService:
    def __init__(self) -> None:
        self.bucket_name = settings.S3_BUCKET_NAME
        self.endpoint_url = settings.S3_ENDPOINT_URL
        self.region_name = settings.S3_REGION
        self.enabled = bool(self.bucket_name)
        self._client = None

    def _get_client(self):
        if not self.enabled:
            return None

        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.client import Config
        except ImportError as exc:
            raise ObjectStorageError(
                "Object storage is configured, but boto3 is not installed."
            ) from exc

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            region_name=self.region_name,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        return self._client

    def ensure_bucket(self) -> None:
        if not self.enabled:
            return

        client = self._get_client()
        try:
            client.head_bucket(Bucket=self.bucket_name)
            return
        except Exception:
            pass

        create_kwargs = {"Bucket": self.bucket_name}
        if self.region_name and self.region_name != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": self.region_name
            }

        try:
            client.create_bucket(**create_kwargs)
            logger.info("Created object storage bucket '%s'", self.bucket_name)
        except Exception as exc:
            raise ObjectStorageError(
                f"Failed to ensure object storage bucket '{self.bucket_name}': {exc}"
            ) from exc

    def upload_file(self, local_path: str, key: str) -> str:
        if not self.enabled:
            return local_path

        client = self._get_client()
        try:
            client.upload_file(local_path, self.bucket_name, key)
        except Exception as exc:
            raise ObjectStorageError(
                f"Failed to upload '{local_path}' to object storage key '{key}': {exc}"
            ) from exc
        return self.object_uri(key)

    def download_file(self, key: str, local_path: str) -> bool:
        if not self.enabled:
            return False

        client = self._get_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            client.download_file(self.bucket_name, key, local_path)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to download object storage key '%s' to '%s': %s",
                key,
                local_path,
                exc,
            )
            return False

    def object_uri(self, key: str) -> str:
        return f"s3://{self.bucket_name}/{key}"


@lru_cache(maxsize=1)
def get_object_storage_service() -> ObjectStorageService:
    return ObjectStorageService()
