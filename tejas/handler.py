def train_model(event, context):
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

    from tejas.core.config import TrainerState
    from tejas.db.boto_client import tasks_table
    from tejas.ml.trainer import LambdaTrainer

    from tejas.core.config import settings

    import json

    from loguru import logger

    logger.info(f"Started Lambda Request: {context.aws_request_id}")

    task_id: str = event["taskId"]
    task_args: Dict[str, str] = event["args"]

    dataset_zip: str = task_args["datasetZip"]
    model_name: str = task_args["modelName"]

    # instantiate the task in db
    new_task = {
        "taskId": task_id,
        "taskArgs": task_args,
        "taskStatus": TrainerState.INITIALIZING.value,
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
        UpdateExpression="set taskStatus=:status",
        ExpressionAttributeValues={
            ":status": TrainerState.TRAINING.value
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
    logger.info(f"Model for task_id: {task_id} and dataset: {dataset_zip} saved in {saved_model_path}")

    idx_to_classname: Dict[int, str] = {v: k for k, v in trainer.dataset.class_to_idx.iteritems()}

    # dump the idx_to_classname dict to a json file
    idx_to_classname_path = str(settings.MODELS_PATH / f'{task_id}.json')
    with (settings.MODELS_PATH / f'{task_id}.json').open('w') as f:
        json.dump(idx_to_classname, f)

    # update the state and meta in db
    tasks_table.update_item(
        Key={
            "taskId": task_id,
        },
        UpdateExpression="set taskStatus=:status, taskResult=:result",
        ExpressionAttributeValues={
            ":status": TrainerState.COMPLETED.value,
            ":result": {
                "modelPath": saved_model_path,
                "idxToClassnamePath": idx_to_classname_path,
                "trainingLogs": train_stats
            }
        },
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

    # for cloudwatch logs and debug
    logger.info(body)

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
