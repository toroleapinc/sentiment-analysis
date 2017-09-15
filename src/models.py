"""SVM and baseline classifiers."""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def train_svm(X, y, kernel='rbf', cv=5):
    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100]}
    else:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01]}
    grid = GridSearchCV(SVC(kernel=kernel), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f"SVM ({kernel}): best CV={grid.best_score_:.4f}, params={grid.best_params_}")
    return grid.best_estimator_

def train_logreg(X, y):
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)
    return lr

def train_nb(X, y):
    nb = MultinomialNB()
    nb.fit(X, y)
    return nb
