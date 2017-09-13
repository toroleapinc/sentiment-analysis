"""Feature extraction: BoW, TF-IDF, Word2Vec."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def bow_features(texts, max_features=10000):
    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    return vec, vec.fit_transform(texts)

def tfidf_features(texts, max_features=10000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    return vec, vec.fit_transform(texts)

def word2vec_features(texts, vector_size=300, window=5, epochs=20):
    tokenized = [t.split() for t in texts]
    model = Word2Vec(tokenized, vector_size=vector_size, window=window,
                     min_count=2, workers=4, epochs=epochs)
    def doc_vector(tokens):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)
    return model, np.array([doc_vector(t) for t in tokenized])
