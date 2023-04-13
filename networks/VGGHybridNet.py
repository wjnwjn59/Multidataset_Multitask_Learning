import torch.nn as nn

from utils.utils import *
from torchvision import models

class VGGHybridNet(nn.Module):
    def __init__(
            self, 
            task1_n_classes=12,
            task2_n_classes=2
        ):
        super(VGGHybridNet, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        unfreeze_pretrained(vgg16)

        self.encoder = nn.Sequential(*list(vgg16.features.children()))

        self.task1_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, task1_n_classes)
        )

        self.task2_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, task2_n_classes)
        )

    def forward(self, x, task):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        if task == 0:
            x = self.task1_classifier(x)
        elif task == 1: 
            x = self.task2_classifier(x)

        return x