import torch
from embeddings import get_phrase_embedding, model
from model import Net, load_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(input_dim=model.vector_size).to(device)

def predict(net, phrase):
    net.eval()
    vec = torch.tensor(get_phrase_embedding(phrase), dtype=torch.float32)
    with torch.no_grad():
        logits = net(vec.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    label = int(torch.argmax(probs))
    class_name = "positive" if label == 1 else "negative"

    print(f"Фраза: {phrase}")
    print(f"Вероятности: negative={probs[0]:.3f}, positive={probs[1]:.3f}")
    print(f"Класс: {label} ({class_name})\n")
    return label

load_weights(net)

predict(net, "I loved this movie, it was amazing")
predict(net, "The food was cold and the service was terrible")
predict(net, "The movie was not good")
predict(net, "Amina is dumb")

