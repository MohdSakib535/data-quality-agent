"""
S3 / MinIO storage service.
Presigned URLs (PUT/GET), head_object, delete_object, multipart upload, streaming read.
All boto3 exceptions handled explicitly.
"""
import io
import logging
from datetime import datetime, timezone, timedelta
from typing import BinaryIO, Iterator

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Minimum S3 multipart part size (5 MB)
MIN_PART_SIZE = 5 * 1024 * 1024


def _get_s3_client():
    """Create a boto3 S3 client pointing at the configured endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        region_name=settings.S3_REGION,
    )


# ── Presigned URLs ───────────────────────────────────────────

def generate_presigned_put_url(s3_key: str, ttl: int | None = None) -> tuple[str, datetime]:
    """Generate a presigned PUT URL for uploading a file to S3."""
    ttl = ttl or settings.S3_PRESIGNED_URL_TTL_UPLOAD
    client = _get_s3_client()
    try:
        url = client.generate_presigned_url(
            "put_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=ttl,
        )
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        logger.info("Generated presigned PUT URL", extra={"s3_key": s3_key, "ttl": ttl})
        return url, expires_at
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("Failed to generate presigned PUT URL", extra={"s3_key": s3_key, "error": str(exc)})
        raise


def generate_presigned_get_url(s3_key: str, ttl: int | None = None) -> tuple[str, datetime]:
    """Generate a presigned GET URL for downloading a file from S3."""
    ttl = ttl or settings.S3_PRESIGNED_URL_TTL_DOWNLOAD
    client = _get_s3_client()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=ttl,
        )
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        logger.info("Generated presigned GET URL", extra={"s3_key": s3_key, "ttl": ttl})
        return url, expires_at
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("Failed to generate presigned GET URL", extra={"s3_key": s3_key, "error": str(exc)})
        raise


# ── Head / Exists / Delete ───────────────────────────────────

def head_object(s3_key: str) -> dict | None:
    """Return S3 object metadata or None if not found."""
    client = _get_s3_client()
    try:
        response = client.head_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        return {
            "content_length": response["ContentLength"],
            "content_type": response.get("ContentType"),
            "last_modified": response.get("LastModified"),
        }
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return None
        logger.error("head_object failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise
    except (NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("head_object connection error", extra={"s3_key": s3_key, "error": str(exc)})
        raise


def delete_object(s3_key: str) -> None:
    """Delete an S3 object. No-op if object does not exist."""
    client = _get_s3_client()
    try:
        client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        logger.info("Deleted S3 object", extra={"s3_key": s3_key})
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("delete_object failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise


# ── Streaming Read ───────────────────────────────────────────

def stream_object(s3_key: str, chunk_size: int = 8192) -> Iterator[bytes]:
    """Stream an S3 object body in chunks."""
    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        body = response["Body"]
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            logger.error("Object not found for streaming", extra={"s3_key": s3_key})
            raise FileNotFoundError(f"S3 object not found: {s3_key}")
        logger.error("stream_object failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise
    except (NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("stream_object connection error", extra={"s3_key": s3_key, "error": str(exc)})
        raise


def download_object_to_bytes(s3_key: str) -> bytes:
    """Download an entire S3 object into memory. Use only for small files."""
    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"S3 object not found: {s3_key}")
        raise
    except (NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("download_object_to_bytes failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise


# ── Upload ───────────────────────────────────────────────────

def upload_bytes(s3_key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload bytes directly to S3."""
    client = _get_s3_client()
    try:
        client.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("Uploaded bytes to S3", extra={"s3_key": s3_key, "size": len(data)})
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("upload_bytes failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise


def upload_fileobj(s3_key: str, fileobj: BinaryIO, content_type: str = "application/octet-stream") -> None:
    """Upload a file-like object to S3."""
    client = _get_s3_client()
    try:
        client.upload_fileobj(
            fileobj,
            settings.S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Uploaded file object to S3", extra={"s3_key": s3_key})
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("upload_fileobj failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise


# ── Multipart Upload ────────────────────────────────────────

def multipart_upload_from_chunks(
    s3_key: str,
    chunk_iterator: Iterator[bytes],
    content_type: str = "application/octet-stream",
) -> None:
    """
    Upload data to S3 using multipart upload from an iterator of byte chunks.
    Buffers chunks until MIN_PART_SIZE (5MB) is reached before uploading each part.
    """
    client = _get_s3_client()
    mpu = None
    try:
        mpu = client.create_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            ContentType=content_type,
        )
        upload_id = mpu["UploadId"]
        parts: list[dict] = []
        part_number = 1
        buffer = io.BytesIO()

        for chunk in chunk_iterator:
            buffer.write(chunk)
            if buffer.tell() >= MIN_PART_SIZE:
                buffer.seek(0)
                resp = client.upload_part(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=s3_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=buffer.read(),
                )
                parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
                part_number += 1
                buffer = io.BytesIO()

        # Upload remaining bytes
        if buffer.tell() > 0:
            buffer.seek(0)
            resp = client.upload_part(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=buffer.read(),
            )
            parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})

        client.complete_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        logger.info(
            "Multipart upload complete",
            extra={"s3_key": s3_key, "parts_count": len(parts)},
        )

    except Exception as exc:
        # Abort multipart upload on any failure
        if mpu:
            try:
                client.abort_multipart_upload(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=s3_key,
                    UploadId=mpu["UploadId"],
                )
            except Exception:
                logger.warning("Failed to abort multipart upload", extra={"s3_key": s3_key})
        logger.error("Multipart upload failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise


# ── Magic byte validation ────────────────────────────────────

def validate_file_magic_bytes(s3_key: str) -> str | None:
    """
    Read the first 8 bytes of an S3 object and check for known file types.
    Returns: 'csv', 'xlsx', or None (unknown/invalid).
    """
    client = _get_s3_client()
    try:
        response = client.get_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Range="bytes=0-7",
        )
        header = response["Body"].read()
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.error("validate_file_magic_bytes failed", extra={"s3_key": s3_key, "error": str(exc)})
        raise

    # XLSX / XLS files start with PK (zip) or XLS OLE header
    if header[:2] == b"PK":
        return "xlsx"
    # OLE Compound File (older .xls)
    if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "xls"
    # CSV: treat as text if it's mostly printable ASCII/UTF-8
    try:
        header.decode("utf-8")
        return "csv"
    except UnicodeDecodeError:
        return None
