from model import Net, device, loss_fn, save_weights
from embeddings import model, train_loader, val_loader
import torch
from train import train


net = Net(input_dim=model.vector_size).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
history = train(net, train_loader, val_loader, loss_fn, optimizer, n_epochs=30)
save_weights(net)
