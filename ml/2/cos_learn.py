import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


# X = torch.normal(mean=torch.zeros((1000,1)), std=2)
# Y = torch.cos(X)

sns.set_theme()

# plt.figure(figsize=(20,7))
# plt.scatter(x=X, y=Y)


NN = torch.nn.Sequential(nn.Linear(1,5, bias=True),
                         nn.Tanh(),
                         nn.Linear(5,5, bias=True),
                         nn.Tanh(),
                         nn.Linear(5,1, bias=True),
                         nn.Tanh())

X_test = torch.linspace(-6,6,1000)
Y_test = torch.cos(X_test)

nn_prediction = NN(X_test.view(-1, 1))
nn_prediction = nn_prediction.detach().numpy()

plt.figure(figsize=(20,7))
plt.scatter(x=X_test, y=Y_test, label="True Cosine")
plt.scatter(x=X_test, y=nn_prediction, label="NN predictions")
plt.legend()
plt.show()
