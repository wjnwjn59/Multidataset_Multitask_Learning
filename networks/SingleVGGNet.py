import torch.nn as nn

from utils.utils import *
from torchvision import models

class SingleVGGNet(nn.Module):
    def __init__(
            self, 
            n_classes=2
        ):
        super(SingleVGGNet, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        unfreeze_pretrained(vgg16)

        self.encoder = nn.Sequential(*list(vgg16.features.children()))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        ) 

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x