# Import Python built-in modules
import os
import json
import urllib3
# Import the core class of the official MinIO Python SDK
from minio import Minio
# Project internal configurations and logger
from app.conf.minio_config import minio_config
from app.core.logger import logger

# Global MinIO client object, initialized to None
minio_client = None

def get_minio_client():
    """
    Lazily initializes and retrieves the MinIO client instance.
    (Moves the connection and bucket validation inside the function to completely resolve import-blocking issues)
    :return: Initialized Minio object / None (if initialization failed)
    """
    global minio_client
    
    # If already successfully initialized, return the cached instance directly
    if minio_client is not None:
        return minio_client

    try:
        # Core debugging print: force-verify exactly what configuration was read
        print("\n" + "="*50)
        print("[DEBUG] MinIO Client initializing inside function...")
        print(f"[DEBUG] Endpoint: {minio_config.endpoint}")
        print(f"[DEBUG] Bucket Name: {minio_config.bucket_name}")
        print("="*50 + "\n")

        # Create an HttpClient with a timeout to prevent the underlying network socket from blocking indefinitely (throws an error if connection fails within 5 seconds)
        http_client = urllib3.PoolManager(timeout=5.0)

        # Initialize MinIO client instance
        client_instance = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=False,  # Ensure local testing uses HTTP
            http_client=http_client
        )
        bucket_name = minio_config.bucket_name

        # Check if the bucket exists; if not, create it automatically
        if not client_instance.bucket_exists(bucket_name):
            logger.info(f"MinIO bucket [{bucket_name}] does not exist; starting creation.")
            client_instance.make_bucket(bucket_name)
            logger.info(f"MinIO bucket [{bucket_name}] created successfully.")
        else:
            logger.info(f"MinIO bucket [{bucket_name}] already exists; skipping creation.")

        # Configure the public read-only policy for the bucket
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
            }]
        }
        client_instance.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
        logger.info(f"MinIO bucket [{bucket_name}] has been configured with a public read-only policy.")

        # Assign to the global variable upon successful completion
        minio_client = client_instance
        return minio_client

    except Exception as e:
        # With the timeout configured, connection failures will immediately catch exceptions and print, avoiding any silent hanging
        logger.error(f"MinIO client initialization failed; error message: {str(e)}", exc_info=True)
        print(f"\n[CRITICAL ERROR] Failed to connect to MinIO: {e}\n")
        minio_client = None
        return None