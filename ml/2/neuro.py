import time

import tqdm
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch.nn as nn
import torch


def train(model, X, y, criterion, optimizer, num_epoch):
    for t in tqdm(range(num_epoch)):
        y_pred = model(X)

        loss = criterion(y_pred, y)

        # print(f"Ошибка {loss.item()}")
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model


NN = nn.Sequential(nn.Linear(1, 100, bias=True),
                   nn.Tanh(),
                   nn.Linear(100, 100, bias=True),
                   nn.Tanh(),
                   nn.Linear(100, 1, bias=True),
                   nn.Tanh())


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(NN.parameters(), lr=1e-2)


X = torch.normal(mean=torch.zeros((1000, 1)), std=10)
Y = torch.cos(X)
X_test = torch.linspace(-6,6,1000)
Y_test = torch.cos(X_test)




for i in range(100):
    print(i)
    NN = train(NN, X, Y, criterion, optimizer, 30)
    print(NN[0].bias)

    nn_prediction = NN(X_test.view(-1,1))
    nn_prediction = nn_prediction.detach().numpy()

    plt.figure(figsize=(20,7))

    plt.scatter(x=X_test, y=Y_test, label="True Cosine")
    plt.scatter(x=X_test, y=nn_prediction, label="NN predictions")



plt.show()
