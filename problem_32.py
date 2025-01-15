import numpy as np
from sklearn.datasets import load_iris
from itertools import combinations
import math

def load_versicolor_virginica():
    """Load only Versicolor and Virginica samples from Iris dataset."""
    iris = load_iris()
    indices = np.where((iris.target == 1) | (iris.target == 2))[0]
    
    X = iris.data[indices, 1:3]
    y = np.where(iris.target[indices] == 1, -1, 1)
    
    return X, y

def create_line_classifiers(points, labels):
    """Create weak classifiers based on lines between all pairs of points."""
    lines = []
    n_points = len(points)
    
    for i, j in combinations(range(n_points), 2):
        p1, p2 = points[i], points[j]
        
        if np.allclose(p1, p2):
            continue
            
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        
        norm = np.sqrt(a*a + b*b)
        a, b, c = a/norm, b/norm, c/norm
        
        lines.append((a, b, c))
    
    return lines

def classify_point(point, line):
    """Classify a point based on which side of the line it falls."""
    a, b, c = line
    return np.sign(a*point[0] + b*point[1] + c)

def get_predictions(points, line):
    """Get predictions for all points using a line classifier."""
    return np.array([classify_point(point, line) for point in points])

def adaboost_train(X, y, lines, T=8):
    """Train AdaBoost for T iterations using line classifiers."""
    n_samples = len(X)
    # Initialize D₁(xᵢ) = 1/n
    D_t = np.ones(n_samples) / n_samples
    
    alphas = []
    selected_classifiers = []
    
    for t in range(T):
        min_error = float('inf')
        best_h_t = None
        best_predictions = None
        
        # Step 3: Find classifier with minimum weighted error
        for h in lines:
            predictions = get_predictions(X, h)
            epsilon_t = np.sum(D_t * (predictions != y))
            
            if epsilon_t < min_error:
                min_error = epsilon_t
                best_h_t = h
                best_predictions = predictions
        
        # Normalize error by current distribution
        epsilon_t = min_error / np.sum(D_t)
        
        if epsilon_t >= 0.5 or epsilon_t == 0:
            continue
            
        # Step 5: Calculate classifier weight αₜ
        alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        
        # Step 6: Update distribution D_(t+1)
        D_t_plus_1 = D_t * np.exp(-alpha_t * y * best_predictions)
        Z_t = np.sum(D_t_plus_1)
        D_t = D_t_plus_1 / Z_t
        
        alphas.append(alpha_t)
        selected_classifiers.append(best_h_t)
        
        if len(selected_classifiers) == T:
            break
            
    return selected_classifiers, alphas

def adaboost_predict(X, classifiers, alphas, k):
    """Make predictions using the first k weak classifiers."""
    # Calculate F(x) = Σᵏₜ₌₁ αₜhₜ(x)
    F_x = np.zeros(len(X))
    
    for alpha_t, h_t in zip(alphas[:k], classifiers[:k]):
        F_x += alpha_t * get_predictions(X, h_t)
    
    # Calculate H(x) = sign[F(x)]    
    return np.sign(F_x)

def compute_error(y_true, y_pred):
    """Compute classification error."""
    return np.mean(y_true != y_pred)

def run_experiment():
    """Run the complete AdaBoost experiment with progress tracking."""
    print("Starting Adaboost experiment with 100 runs...")
    X, y = load_versicolor_virginica()
    n_runs = 100
    k_values = range(1, 9)
    
    train_errors = np.zeros((n_runs, 8))
    test_errors = np.zeros((n_runs, 8))
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/100")
        indices = np.random.permutation(len(X))
        split = len(X) // 2
        
        train_idx = indices[:split]
        test_idx = indices[split:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        lines = create_line_classifiers(X_train, y_train)
        classifiers, alphas = adaboost_train(X_train, y_train, lines)
        
        for k in k_values:
            if k <= len(classifiers):
                y_train_pred = adaboost_predict(X_train, classifiers, alphas, k)
                y_test_pred = adaboost_predict(X_test, classifiers, alphas, k)
                
                train_errors[run, k-1] = compute_error(y_train, y_train_pred)
                test_errors[run, k-1] = compute_error(y_test, y_test_pred)
    
    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)
    
    print("\nFinal Results after 100 runs:")
    print("\nAverage True Errors:")
    for k, error in enumerate(avg_test_errors, 1):
        print(f"H{k}: {error:.4f}")

    print("\nAverage Empirical Errors:")
    for k, error in enumerate(avg_train_errors, 1):
        print(f"H{k}: {error:.4f}")

if __name__ == "__main__":
    run_experiment()