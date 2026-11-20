from pathlib import Path
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hide = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.hide(x)))
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(input_dim=384).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


MODEL_PATH = Path(__file__).with_name("sentiment_model.pt")


def save_weights(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)


def load_weights(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    return model
