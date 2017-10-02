"""Visualization."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, save='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save)
    plt.close()

def plot_model_comparison(results, save='comparison.png'):
    names = list(results.keys())
    accs = list(results.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, accs)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0.85, 1.0)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
