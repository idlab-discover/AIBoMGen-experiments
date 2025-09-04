import boto3
import os
import time


def initialize_s3_client(endpoint_url, access_key, secret_key):
    """
    Initialize the S3 client for MinIO.
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def is_job_finished(s3_client, bucket_name, folder_name):
    """
    Check if the job is finished by looking for the 'output' folder in the specified job folder.
    """
    print(
        f"Checking if job is finished by looking for required files in '{folder_name}/output/'...")
    required_files = [
        "logs.log",
        "cyclonedx_bom.json",
        "metrics.json",
        "trained_model.keras"
    ]
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=bucket_name, Prefix=f"{folder_name}/output/")

    found_files = set()
    found_link_file = False
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                filename = os.path.basename(obj["Key"])
                if filename in required_files:
                    found_files.add(filename)
                if filename.endswith('.link'):
                    found_link_file = True
    missing_files = set(required_files) - found_files
    if missing_files or not found_link_file:
        print(
            f"Still missing files: {missing_files if missing_files else ''}{' and no .link file found' if not found_link_file else ''}")
        return False
    return True


def download_folder_from_minio(s3_client, bucket_name, folder_name, download_dir):
    """
    Download a folder from MinIO to the specified local directory.
    """
    print(f"Downloading folder '{folder_name}' from bucket '{bucket_name}'...")
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_name)

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                file_key = obj["Key"]
                file_path = os.path.join(download_dir, file_key)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                s3_client.download_file(bucket_name, file_key, file_path)
                print(f"Downloaded: {file_key} -> {file_path}")


def wait_for_job_and_download(s3_client, bucket_name, folder_name, download_dir, check_interval=10):
    """
    Wait until the job is finished (i.e., 'output' folder exists) and then download the folder from MinIO.
    """
    while not is_job_finished(s3_client, bucket_name, folder_name):
        print("Job is not yet finished. Retrying in 10 seconds...")
        # Wait for the specified interval before retrying
        time.sleep(check_interval)

    # Download the folder from MinIO
    download_folder_from_minio(
        s3_client, bucket_name, folder_name, download_dir)
    print(f"Artifacts downloaded to: {download_dir}")
