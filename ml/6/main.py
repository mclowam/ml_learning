import string
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from matplotlib import pyplot as plt
from IPython.display import clear_output
from gensim.models import Word2Vec
import gensim.downloader as api


# data = list(open("quora.txt", encoding="utf-8"))
#
# tokenizer = WordPunctTokenizer()
#
# print(tokenizer.tokenize(data[55]))
#
# data_tok = [tokenizer.tokenize(x.lower()) for x in data]
#
# len(data)
#
# model_obj = Word2Vec(data_tok,
#                      vector_size=32,
#                      min_count=5,
#                      window=5)
#
# model = model_obj.wv
#
# ex = model.get_vector("cat")
# print(ex)

model = api.load("glove-twitter-25")

print(model.most_similar(positive=["лето"]))