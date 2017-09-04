# Sentiment Analysis

Comparing BoW, TF-IDF, and Word2Vec features with SVM classifiers for binary sentiment analysis. Got top 5% on the class leaderboard with TF-IDF + RBF kernel SVM.

### Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

Put your dataset (TSV with `review` and `sentiment` columns) in `data/`.

### Usage

```bash
python -m src.train --data data/train.tsv
python -m src.evaluate --data data/test.tsv --model models/best_svm.pkl
```

### Results

Best single model: TF-IDF + SVM (RBF) -> 95.3% accuracy, 0.953 F1
