# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import boto3
import io
import os
import lipforcing.utils.logging_utils as logger


def set_env_vars(credentials_path: str = None) -> None:
    """
    Set the environment variables for LipForcing

    Args:
      credentials_path: The path to the JSON file containing AWS credentials and region information.
    """

    # Reads AWS credentials and configuration from a JSON file and sets them as environment variables.
    if credentials_path is not None and os.path.isfile(credentials_path):
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            key_map = {
                "AWS_ACCESS_KEY_ID": "aws_access_key_id",
                "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
                "AWS_DEFAULT_REGION": "region_name",
                "AWS_ENDPOINT_URL": "endpoint_url",
                "S3_ENDPOINT_URL": "endpoint_url",
            }
            for env_key, config_key in key_map.items():
                if config_key in config:
                    os.environ[env_key] = config[config_key]
                else:
                    logger.warning(f"Missing key {config_key} in {credentials_path}, skipping env variable {env_key}.")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {credentials_path}, skip loading AWS credentials.")
        else:
            logger.success(f"AWS credentials loaded from {credentials_path} and set as environment variables.")

    # Set Hugging Face cache directory
    os.environ["HF_HOME"] = os.getenv(
        "HF_HOME", os.path.join(os.getenv("LIPFORCING_OUTPUT_ROOT", "outputs"), ".cache")
    )


def latest_checkpoint(path: str) -> str:
    """Get the latest checkpoint with the largest iteration number from S3 bucket"""

    if path.startswith("s3://"):
        # Get the list of objects in the s3 container

        s3_client = boto3.client("s3")

        objects = s3_client.list_objects_v2(Bucket=path)["Contents"]
        # Filter for .pth files and extract iteration numbers
        model_files = [obj["Key"] for obj in objects if obj["Key"].endswith(".pth")]
    elif os.path.exists(path):
        # Get the list of files in the local directory
        model_files = os.listdir(path)
    else:
        # no model files found
        model_files = []

    iterations = []
    for file in model_files:
        try:
            # Assuming file names are like '123.pth'
            iterations.append(int(file.split(".")[0]))
        except ValueError:
            pass  # Skip files with invalid names

    if not iterations:
        logger.error(f"No model files found in {path}")
        return ""

    # Find the highest iteration number
    latest_iteration = max(iterations)
    latest_model_path = os.path.join(path, f"{latest_iteration:07d}")

    return latest_model_path


def s3_load(s3_path: str) -> io.BytesIO:
    """Load a file from S3 bucket and return the content as bytes"""

    bucket = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])

    s3_client = boto3.client("s3")
    obj = s3_client.get_object(Bucket=bucket, Key=key)

    return io.BytesIO(obj["Body"].read())


def s3_save(s3_path: str, data: bytes) -> None:
    """Save a file to S3 bucket"""

    bucket = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])

    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
