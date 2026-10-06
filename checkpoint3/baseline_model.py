import torch
from torchvision import models


NUM_CLASSES = 3


def create_model():
    model = models.resnet18(weights="DEFAULT")

    # Replace the final classification layer
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


if __name__ == "__main__":
    model = create_model()

    print("MedScan ResNet-18 baseline model created")
    print(model.fc)
