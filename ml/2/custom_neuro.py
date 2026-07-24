import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, dim=1, neuros=105):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(dim, neuros)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(neuros, neuros)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(neuros, 1)
        self.tanh3 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)

        x = self.fc2(x)
        x = self.tanh2(x)

        x = self.fc3(x)
        x = self.tanh3(x)

        return x


def train(model, X, y, criterion, optimizer, num_epoch):
    for t in tqdm(range(num_epoch)):
        y_pred = model(X)

        loss = criterion(y_pred, y)

        # print(f"Ошибка {loss.item()}")
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model


NN = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(NN.parameters(), lr=1e-2)

X = torch.normal(mean=torch.zeros((1000, 1)), std=10)
Y = torch.cos(X)
X_test = torch.linspace(-6, 6, 1000)
Y_test = torch.cos(X_test)

for i in range(200):
    print(i)
    NN = train(NN, X, Y, criterion, optimizer, 30)

    nn_prediction = NN(X_test.view(-1, 1))
    nn_prediction = nn_prediction.detach().numpy()

    plt.figure(figsize=(20, 7))

    plt.scatter(x=X_test, y=Y_test, label="True Cosine")
    plt.scatter(x=X_test, y=nn_prediction, label="NN predictions")

plt.show()
