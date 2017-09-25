"""Evaluate models on test data."""
import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .preprocess import clean_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='models/best_svm.pkl')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        saved = pickle.load(f)
    model, vec = saved['model'], saved['vectorizer']
    df = pd.read_csv(args.data, sep='\t')
    df['clean'] = df['review'].apply(clean_text)
    X = vec.transform(df['clean'].values)
    y_pred = model.predict(X)
    y_true = df['sentiment'].values
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    main()
