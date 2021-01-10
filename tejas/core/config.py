from enum import Enum
from pydantic import BaseSettings
from pathlib import Path

import os


class TrainerState(Enum):
    UNKNOWN = "UNKNOWN"
    INITIALIZING = "INITIALIZING"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Settings(BaseSettings):
    PROJECT_NAME: str = "TejasAI"

    TASKS_TABLE = os.environ["TASKS_TABLE"]
    MODELS_PATH = Path(os.environ["TEJAS_MODELS_PATH"])
    # DATASETS_PATH = Path(os.environ["TEJAS_DATASETS_PATH"])
    PRETRAINED_PATH = Path(os.environ["TEJAS_PRETRAINED_PATH"])

    DATASETS_BUCKET: str = os.environ["TEJAS_DATASETS_BUCKET"]

    MAX_EPOCHS: int = int(os.environ["TEJAS_MAX_EPOCHS"])


settings = Settings()
