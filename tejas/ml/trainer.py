from torch.jit import RecursiveScriptModule, ScriptModule

from .models import MobileNetV2
from .dataset import ZipDataset

import torch

import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from loguru import logger

from typing import Callable, Dict, Any, Tuple, List


class LambdaTrainer:
    def __init__(self, dataset_zip: str, model_name: str) -> None:
        ModelClass, transform = get_model_and_transform(model_name=model_name)

        self.dataset: ZipDataset = ZipDataset(zip_file=dataset_zip, transform=transform)

        num_classes: int = len(self.dataset.classes)

        self.model = ModelClass(num_classes=num_classes)

        self.train_loader = DataLoader(
            self.dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, n_epochs: int) -> Tuple[Any, List[str]]:
        train_stats: List[str] = []
        for epoch in range(n_epochs):
            results = self.train_epoch(epoch)

            train_stats.append(
                f"Epoch: {epoch+1}/{n_epochs}, Train Acc: {results['acc']}, Train Loss: {results['loss']}"
            )

        logger.info(f"Finished training {n_epochs} Epochs")

        return self.traced_model, train_stats

    @property
    def traced_model(self) -> ScriptModule:
        traced_model = torch.jit.trace(self.model, (torch.randn(1, 3, 224, 224),))
        return traced_model

    def train_epoch(self, epoch_num: int) -> Dict[str, Any]:
        self.model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataset)
        epoch_acc = running_corrects.double() / len(self.dataset)

        return {"loss": epoch_loss, "acc": epoch_acc}


def get_model_and_transform(model_name: str) -> Tuple[Callable, T.Compose]:
    if model_name == "mobilenet_v2":
        model_class: Callable = MobileNetV2
        transform: T.Compose = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return model_class, transform
