import boto3
import os
from uuid import uuid4
from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-southeast-2"
)

BUCKET_NAME = "sceneweaver" 

def upload_image_to_s3(image_bytes: bytes, filename: str, folder: str = "images"):
    key = f"{folder}/{uuid4().hex}_{filename}"
    s3.upload_fileobj(
        BytesIO(image_bytes),
        BUCKET_NAME,
        key,
        ExtraArgs={
            "ContentType": "image/jpeg",
            "ContentDisposition": f'attachment; filename="{filename}"'
        }
    )
    return f"https://{BUCKET_NAME}.s3.ap-southeast-2.amazonaws.com/{key}"

def delete_image_from_s3(image_url: str):
   
    try:
        prefix = f"https://{BUCKET_NAME}.s3.ap-southeast-2.amazonaws.com/"
        if not image_url.startswith(prefix):
            raise ValueError("Invalid image URL format")

        key = image_url.replace(prefix, "")
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
        return True

    except ClientError as e:
        print(f"S3 deletion error: {e}")
        return False

    except Exception as e:
        print(f"Unexpected error during S3 deletion: {e}")
        return False