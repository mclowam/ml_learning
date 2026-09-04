import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision
import torch
from PIL import Image
from torch.optim import optimizer
from torchvision import datasets, transforms,models


model = models.resnet18(pretrained=True)


resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])
])

image = Image.open("pes.jfif")

image_trans = resnet_transform(image)
# print(np.array(image).shape)
# print("*" * 10)
# print(image_trans.shape)

model.eval()

with torch.no_grad():
    model_output = model(image_trans.reshape((1,3,224,224)))
    np.argmax(model_output.data.cpu().numpy())

t = torch.tensor([1,2,3,4,5,6,7,8])
t.view()

# plt.imshow(image_trans.permute(1,2,0).data.cpu().numpy())
# plt.show()