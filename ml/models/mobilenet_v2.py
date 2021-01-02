import torch.nn as nn

from torchvision import models


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super(MobileNetV2, self).__init__()

        self.model_ft = models.mobilenet_v2(pretrained=True)
        num_ftrs = self.model_ft.classifier[1].out_features
        self.classifier_layer = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        intermediate = self.model_ft(x)
        return self.classifier_layer(intermediate)
