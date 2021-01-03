# this should always be at the top of the root handler
try:
    import sys
    import os

    sys.path.insert(0, os.environ["TEJAS_LIBS_PATH"])

    print("Successfully Imported")
except Exception as ex:
    print(f"error: {ex}")
    raise ex

from typing import Dict, List

from torch.jit import ScriptModule

from core.config import TrainerState
from db.boto_client import tasks_table
from ml.trainer import LambdaTrainer

from .core.config import settings


import json

from loguru import logger


def train_model(event, context):

    logger.info(f"Started Lambda Request: {context.aws_request_id}")

    task_id: str = event["taskId"]
    task_args: Dict[str, str] = event["args"]

    dataset_zip: str = task_args["datasetZip"]
    model_name: str = task_args["modelName"]

    # instantiate the task in db
    new_task = {
        "taskId": task_id,
        "taskArgs": task_args,
        "taskStatus": TrainerState.INITIALIZING,
        "taskResult": "",
    }
    tasks_table.put_item(Item=new_task)

    # instantiate the trainer
    trainer = LambdaTrainer(dataset_zip=dataset_zip, model_name=model_name)

    # start training
    tasks_table.update_item(
        Key={
            "taskId": task_id,
        },
        AttributeUpdates={
            "taskStatus": TrainerState.TRAINING
        }
    )

    logger.info(f"Training Started for {task_id}")

    traced_model: ScriptModule
    train_stats: List[str]
    traced_model, train_stats = trainer.train(n_epochs=1)

    # trace and save the model
    saved_model_path = str(settings.MODELS_PATH / f"{task_id}.traced.pt")
    traced_model.save(saved_model_path)

    logger.info(f"Training Completed for {task_id}")

    # update the state and meta in db
    tasks_table.update_item(
        Key={
            "taskId": task_id,
        },
        AttributeUpdates={
            "taskStatus": TrainerState.COMPLETED,
            "taskResult": {
                "modelPath": saved_model_path,
                "trainingLogs": train_stats
            }
        }
    )

    task_return = tasks_table.get_item(
        Key={
            "taskId": task_id
        }
    )['Item']

    body = {
        "message": "Model Training Function Completed",
        "task": json.dumps(task_return),
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
