import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from tqdm import tqdm

resnet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('./archive/train', transform=resnet_transforms)
val_data = datasets.ImageFolder('./archive/valid', transform=resnet_transforms)
test_data = datasets.ImageFolder('./archive/test', transform=resnet_transforms)

classes = train_data.classes
num_classes = len(classes)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


def create_model(num_freeze_layers, num_out_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_out_classes)

    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_freeze_layers=9, num_out_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(model, data_loader, loss_fn):
    losses = []
    num_current = 0
    num_elements = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            losses.append(loss.item())
            y_pred = torch.argmax(logits, dim=1)
            num_current += torch.sum(y_pred == y_batch).item()
            num_elements += len(y_batch)

    accuracy = num_current / num_elements
    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch=5):
    for epoch in range(n_epoch):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epoch}")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc, val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{n_epoch} -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    return model


model = train(model, loss_fn, optimizer, n_epoch=5)

test_accuracy, test_loss = evaluate(model, test_loader, loss_fn)
print(f"Test Accuracy: {test_accuracy:.4f}")


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


model.eval()
X_batch, y_batch = next(iter(test_loader))

with torch.no_grad():
    logits = model(X_batch.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for i, ax in enumerate(axes.flat):
    if i >= len(X_batch):
        break

    img = denormalize(X_batch[i])
    ax.imshow(img)

    pred_class = classes[preds[i]]
    true_class = classes[y_batch[i].item()]

    color = "green" if preds[i] == y_batch[i].item() else "red"
    ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
    ax.axis("off")

plt.tight_layout()
plt.show()