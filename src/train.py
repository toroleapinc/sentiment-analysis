"""Main training script."""
import argparse
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocess import clean_text
from .features import bow_features, tfidf_features, word2vec_features
from .models import train_svm, train_logreg, train_nb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep='\t')
    df['clean'] = df['review'].apply(clean_text)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean'].values, df['sentiment'].values,
        test_size=args.test_size, stratify=df['sentiment'], random_state=42
    )

    print("\n=== TF-IDF features ===")
    tfidf_vec, X_train_tfidf = tfidf_features(X_train_text)
    svm_rbf = train_svm(X_train_tfidf, y_train, kernel='rbf')
    svm_linear = train_svm(X_train_tfidf, y_train, kernel='linear')

    print("\n=== BoW features ===")
    bow_vec, X_train_bow = bow_features(X_train_text)
    svm_bow = train_svm(X_train_bow, y_train)
    nb = train_nb(X_train_bow, y_train)

    # Word2Vec (experimental)
    # print("\n=== Word2Vec features ===")
    # w2v_model, X_train_w2v = word2vec_features(X_train_text)
    # TODO: word2vec eval - it's slower and didn't beat TF-IDF

    os.makedirs('models', exist_ok=True)
    with open('models/best_svm.pkl', 'wb') as f:
        pickle.dump({'model': svm_rbf, 'vectorizer': tfidf_vec}, f)
    print("\nSaved best model to models/best_svm.pkl")

if __name__ == '__main__':
    main()
