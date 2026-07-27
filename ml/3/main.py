import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


full_train_data = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transforms.ToTensor())

train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()

        self.fc_in = nn.Linear(28*28, 256)
        self.dop1 = nn.Linear(256, 256)
        self.dop2 = nn.Linear(256, 256)
        self.dop3 = nn.Linear(256, 256)

        self.fc_out = nn.Linear(256, 10)

    def forward(self, x):
        #переводим входной объект из картинки в вектор
        x = self.flatten(x)

        #умножение матрицы весов 1 слоя и применения функции активации
        x = F.relu(self.fc_in(x))

        x = self.dop1(x)
        x = self.dop2(x)
        x = self.dop3(x)

        # умножение матрицы весов 2 слоя и применения функции активации
        x = self.fc_out(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, loss_fn):
    model.eval()
    y_pred_list = []
    y_true_list = []
    losses = []

    for i, batch in enumerate(tqdm(dataloader)):
        X_batch, y_batch = batch

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            loss = loss.item()

            losses.append(loss)

            y_pred = torch.argmax(logits, dim=1)

        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

    accuracy = accuracy_score(y_true_list, y_pred_list)

    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch=6):
    data = {
        "acc_train": [],
        "loss_train": [],
        "acc_val": [],
        "loss_val": []
    }

    for epoch in range(n_epoch):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            X_batch, y_batch = batch

            logits = model(X_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))

            if i % 50 == 0:
                print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch end", epoch)

        acc_train_epoch, loss_train_epoch = evaluate(model, train_loader, loss_fn)
        print("Train acc:", acc_train_epoch, "Train loss:", loss_train_epoch)

        acc_val_epoch, loss_val_epoch = evaluate(model, val_loader, loss_fn)
        print("val acc:", acc_val_epoch, "val loss:", loss_val_epoch)

        data["acc_train"].append(acc_train_epoch)
        data["loss_train"].append(loss_train_epoch)
        data["acc_val"].append(acc_val_epoch)
        data["loss_val"].append(loss_val_epoch)

    return model, data


model = SimpleNet().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model, data = train(model, loss_fn, optimizer, n_epoch=1)



test_acc, test_loss = evaluate(model, test_loader, loss_fn)
print(f"\nTest Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(data["loss_train"], label="Train Loss")
axs[0].plot(data["loss_val"], label="Val Loss")
axs[0].set_title("Loss")
axs[0].set_xlabel("Epoch")
axs[0].legend()

axs[1].plot(data["acc_train"], label="Train Acc")
axs[1].plot(data["acc_val"], label="Val Acc")
axs[1].set_title("Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].legend()

plt.show()

model.eval()
X_batch, y_batch = next(iter(test_loader))

with torch.no_grad():
    logits = model(X_batch.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_batch[i].squeeze(), cmap="gray")
    pred_label = preds[i]
    true_label = y_batch[i].item()

    color = "green" if pred_label == true_label else "red"
    ax.set_title(f"Pred: {pred_label} (True: {true_label})", color=color)
    ax.axis("off")

plt.tight_layout()
plt.show()
