import os, hashlib, mimetypes
import boto3
from botocore.config import Config

def get_s3():
  return boto3.client(
    "s3",
    endpoint_url=os.environ.get("MINIO_ENDPOINT"),
    aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
    region_name=os.environ.get("MINIO_REGION","us-east-1"),
    config=Config(s3={"addressing_style":"path"}, signature_version="s3v4"),
  )

def put_bytes(s3, bucket, key, data, content_type="application/octet-stream"):
  s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
  return hashlib.sha256(data).hexdigest()
