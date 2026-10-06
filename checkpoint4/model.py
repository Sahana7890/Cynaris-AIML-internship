import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .config import NUM_CLASSES


def create_model(pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Replace ImageNet's 1000-class head with our 3-class head.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
