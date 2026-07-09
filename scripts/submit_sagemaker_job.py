# scripts/submit_sagemaker_job.py
# Submits SageMaker Training Job using boto3 directly
# No sagemaker SDK needed — boto3 is already installed

import boto3
import json
import time
import tarfile
import os

# ── Configuration ─────────────────────────────────────────────────
ROLE_ARN    = "arn:aws:iam::475790160491:role/SageMakerExecutionRole-Week11"
BUCKET      = "martin-mlops-models"
PREFIX      = "week11"
REGION      = "eu-central-1"
JOB_NAME    = f"deforestation-risk-{int(time.time())}"
INSTANCE    = "ml.m5.large"

# ── Upload training script to S3 ──────────────────────────────────
print("Step 1: Packaging training script...")
with tarfile.open("/tmp/sourcedir.tar.gz", "w:gz") as tar:
    tar.add("scripts/sagemaker_train.py", arcname="sagemaker_train.py")

s3 = boto3.client("s3", region_name=REGION)
s3.upload_file(
    "/tmp/sourcedir.tar.gz",
    BUCKET,
    f"{PREFIX}/code/sourcedir.tar.gz"
)
print(f"Training script uploaded to s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz")

# ── Submit Training Job ───────────────────────────────────────────
print(f"\nStep 2: Submitting Training Job: {JOB_NAME}")

sm = boto3.client("sagemaker", region_name=REGION)

response = sm.create_training_job(
    TrainingJobName=JOB_NAME,
    AlgorithmSpecification={
        "TrainingImage": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/sklearn:1.2-1-cpu-py3",
        "TrainingInputMode": "File",
    },
    RoleArn=ROLE_ARN,
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output/"
    },
    ResourceConfig={
        "InstanceType": INSTANCE,
        "InstanceCount": 1,
        "VolumeSizeInGB": 5,
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 3600
    },
    HyperParameters={
        "sagemaker_program": "sagemaker_train.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz",
    },
)

print(f"Job submitted successfully!")
print(f"Job Name: {JOB_NAME}")
print(f"Input: s3://{BUCKET}/{PREFIX}/data/")
print(f"Output: s3://{BUCKET}/{PREFIX}/output/")

# ── Monitor job ───────────────────────────────────────────────────
print(f"\nStep 3: Monitoring job status...")
while True:
    status = sm.describe_training_job(TrainingJobName=JOB_NAME)
    job_status = status["TrainingJobStatus"]
    print(f"Status: {job_status}")

    if job_status in ["Completed", "Failed", "Stopped"]:
        break
    time.sleep(30)

if job_status == "Completed":
    model_uri = status["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_uri}")
else:
    print(f"\nJob {job_status}")
    if "FailureReason" in status:
        print(f"Reason: {status['FailureReason']}")
