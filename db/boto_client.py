import boto3

from core.config import settings

ddb_client = boto3.client("dynamodb")
ddb = boto3.resource("dynamodb")
tasks_table = ddb.Table(settings.TASKS_TABLE)
