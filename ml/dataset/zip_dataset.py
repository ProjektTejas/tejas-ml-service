import os

from zipfile import ZipFile
from loguru import logger
from pathlib import Path

from torchvision.datasets import ImageFolder

from typing import Any


class ZipDataset(ImageFolder):
    def __init__(self, zip_file: str, **kwargs: Any) -> None:
        self.zip_file = Path(zip_file)
        root = self.zip_file

        self.extract_zip(root=root)

        super(ZipDataset, self).__init__(root=root, **kwargs)

    def extract_zip(self, root: str) -> None:
        logger.info("extracting dataset zip file")
        zipf: ZipFile = ZipFile(self.zip_file, "r")
        zipf.extractall(root)
