import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision
import torch
from PIL import Image
from torch.optim import optimizer
from torchvision import datasets, transforms,models




resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])
])
