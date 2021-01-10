import boto3

from tejas.core.config import settings
from botocore.client import Config

ddb_client = boto3.client("dynamodb")
ddb = boto3.resource("dynamodb")
tasks_table = ddb.Table(settings.TASKS_TABLE)

s3_client = boto3.client("s3", config=Config(signature_version='s3v4'))
