import torch
import torch.nn as nn
import torchvision

def build_model(num_classes):
    backbone = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    # Freeze pretrained backbone
    for param in backbone.parameters():
        param.requires_grad = False

    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes+1)
    )
    return backbone
