import json
from loguru import logger


def train_model(event, context):

    logger.info(f"Started Lambda Request: {context.aws_request_id}")

    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event,
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
