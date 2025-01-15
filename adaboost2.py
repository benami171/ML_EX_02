import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize distribution D₁(xᵢ) = 1/n
        D_t = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            model = self._train_weak_classifier(X, y, D_t)
            predictions = model.predict(X)
            
            # Compute weighted error εₜ(h)
            epsilon_t = np.sum(D_t * (predictions != y)) / np.sum(D_t)

            # Skip if classifier is too weak or perfect
            if epsilon_t >= 0.5 or epsilon_t == 0:
                continue

            # Calculate classifier weight αₜ
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            self.alphas.append(alpha_t)
            self.models.append(model)

            # Update distribution D_(t+1)
            D_t_plus_1 = D_t * np.exp(-alpha_t * y * predictions)
            Z_t = np.sum(D_t_plus_1)  # Normalization constant
            D_t = D_t_plus_1 / Z_t

    def predict(self, X):
        # Implement F(x) = Σ αₜhₜ(x)
        F_x = np.zeros(X.shape[0])
        for alpha_t, h_t in zip(self.alphas, self.models):
            F_x += alpha_t * h_t.predict(X)
        # Implement H(x) = sign[F(x)]
        return np.sign(F_x)

    def _train_weak_classifier(self, X, y, D_t):
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y, sample_weight=D_t)
        return model