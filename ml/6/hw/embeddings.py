import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gensim.downloader as api
from nltk.tokenize import WordPunctTokenizer
from load import train, val, test

model = api.load("glove-twitter-100")
tokenizer = WordPunctTokenizer()


def get_phrase_embedding(phrase):
    vectors = np.zeros([model.vector_size], dtype="float32")
    tokens = tokenizer.tokenize(phrase.lower())

    vectors = [model[x] for x in tokens if model.has_index_for(x)]

    if vectors:
        vector = np.mean(vectors, axis=0)

    return vector

def embed(texts):
    return np.stack([get_phrase_embedding(t) for t in texts])

def make_loader(df, shuffle):
    X = torch.tensor(embed(df["text"]), dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=shuffle)


train_loader = make_loader(train, shuffle=True)
val_loader = make_loader(val, shuffle=False)
test_loader = make_loader(test, shuffle=True)