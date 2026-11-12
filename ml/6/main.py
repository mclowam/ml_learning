import string
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from matplotlib import pyplot as plt
from IPython.display import clear_output
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import output_notebook
from sklearn.manifold import TSNE


data = list(open("quora.txt", encoding="utf-8"))
#
tokenizer = WordPunctTokenizer()
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
words = list(model.key_to_index.keys())[:1000]
words_vectors = np.asarray([model[x] for x in words])

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
embedding = tsne.fit_transform(words_vectors)

# data_source = bm.ColumnDataSource({
#     "x": embedding[:, 0],
#     "y": embedding[:, 1],
#     "token": words,
#     "color": ["blue"] * len(words)
# })
#
# fig = pl.figure(active_scroll="wheel_zoom", width=700, height=500)
# fig.scatter("x", "y", size=7, color="color", alpha=0.6, marker="circle", source=data_source)
# fig.add_tools(bm.HoverTool(tooltips=[("token", "@token")]))
# pl.show(fig)


def get_phrase_embedding(phrase):
    vector = np.zeros([model.vector_size], dtype="float32")
    phrase_tokenized = tokenizer.tokenize(phrase.lower())
    phrase_vectors = [model[x] for x in phrase_tokenized if model.has_index_for(x)]

    if len(phrase_vectors) != 0:
        vector = np.mean(phrase_vectors, axis=0)

    return vector

vector = get_phrase_embedding("Hello, today we speak about word vector!")
print(vector)
