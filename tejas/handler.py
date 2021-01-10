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
    from pathlib import Path

    from torch.jit import ScriptModule

    from tejas.core.config import TrainerState
    from tejas.db.boto_client import tasks_table
    from tejas.ml.trainer import LambdaTrainer

    from tejas.core.config import settings

    import json
    from datetime import datetime

    import boto3

    s3 = boto3.client('s3')

    from loguru import logger

    # we'll only process the first uploaded file (we dont support multiple datasets training parallel) (for now)
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']
    response = s3.head_object(Bucket=bucket, Key=key)

    # logger.info(response)
    #
    # logger.info(str(event))
    # logger.info(str(context))
    #
    # return {"statusCode": 200}

    logger.info(f"Started Lambda Request: {context.aws_request_id}")

    # download the dataset
    dataset_zip = str(Path("/tmp") / Path(key))
    s3.download_file(bucket, key, dataset_zip)

    train_meta = response['Metadata']
    train_meta_args = json.loads(train_meta['args'])

    task_id: str = train_meta_args["taskId"]
    task_args: Dict[str, str] = train_meta_args["args"]

    # dataset_zip: str = task_args["datasetZip"]
    model_name: str = task_args["modelName"]

    # instantiate the task in db
    new_task = {
        "taskId": task_id,
        "taskArgs": task_args,
        "taskStatus": TrainerState.INITIALIZING.value,
        "taskResult": "",
        "timestamp": datetime.now().isoformat()
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
    traced_model, train_stats = trainer.train(n_epochs=settings.MAX_EPOCHS)

    # trace and save the model
    saved_model_path = str(settings.MODELS_PATH / f"{task_id}.traced.pt")
    traced_model.save(saved_model_path)

    logger.info(f"Training Completed for {task_id}")
    logger.info(f"Model for task_id: {task_id} and dataset: {dataset_zip} saved in {saved_model_path}")

    idx_to_classname: Dict[int, str] = {v: k for k, v in trainer.dataset.class_to_idx.items()}

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
