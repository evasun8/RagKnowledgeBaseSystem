# Import core dependencies: dataclasses, environment variable parser, and path utilities
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (ensures os.getenv can retrieve MinIO-related configurations)
load_dotenv()


# Define the service configuration for MinIO Object Storage (aligning with the architectural style of LLMConfig)
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO service address (including protocol http/https and port)
    access_key: str  # MinIO access key (corresponding to MINIO_ACCESS_KEY)
    secret_key: str  # MinIO secret key (corresponding to MINIO_SECRET_KEY)
    bucket_name: str # MinIO default bucket name (dedicated for knowledge base documents)
    minio_img_dir: str # MinIO directory folder for storing images
    minio_secure: bool # Whether to use SSL encryption (HTTP vs HTTPS)


# Instantiate the MinIO configuration object, automatically reading and binding from .env
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)