import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv("sentiment_dataset.csv")

df["label"] = df["label"].map({"positive": 1, "negative": 0})

def clean(s: pd.Series) -> pd.Series:
    return (s.str.lower()
             .str.strip()
             .str.replace(r"\s+", " ", regex=True))

df["text"] = clean(df["text"])


train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=SEED
)
train, val = train_test_split(
    train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=SEED
)
