import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = self._train_weak_classifier(X, y, w)
            predictions = model.predict(X)
            error = np.sum(w * (predictions != y)) / np.sum(w)

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            self.models.append(model)

            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_predictions += alpha * model.predict(X)
        return np.sign(final_predictions)

    def _train_weak_classifier(self, X, y, w):
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y, sample_weight=w)
        return model

# Example usage:
# X = np.array([[...], [...], ...])
# y = np.array([...])
# adaboost = AdaBoost(n_estimators=50)
# adaboost.fit(X, y)
# predictions = adaboost.predict(X)