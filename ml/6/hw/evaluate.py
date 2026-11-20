import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader, loss_fn):
    model.eval()
    losses = []
    num_correct = 0
    num_elements = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            losses.append(loss_fn(logits, y_batch).item())

            y_pred = torch.argmax(logits, dim=1)
            num_correct += (y_pred == y_batch).sum().item()
            num_elements += len(X_batch)

    return num_correct / num_elements, np.mean(losses)


