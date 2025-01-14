import numpy as np
from sklearn.datasets import load_iris
from itertools import combinations
import math

def load_versicolor_virginica():
    """Load only Versicolor and Virginica samples from Iris dataset."""
    iris = load_iris()
    # Get indices for Versicolor (1) and Virginica (2)
    indices = np.where((iris.target == 1) | (iris.target == 2))[0]
    
    # Get features 2 and 3 only (as per homework requirements)
    X = iris.data[indices, 1:3]
    # Convert Versicolor to -1 and Virginica to 1
    y = np.where(iris.target[indices] == 1, -1, 1)
    
    return X, y

def create_line_classifiers(points, labels):
    """Create weak classifiers based on lines between all pairs of points."""
    lines = []
    n_points = len(points)
    
    # Generate all pairs of points
    for i, j in combinations(range(n_points), 2):
        p1, p2 = points[i], points[j]
        
        # Skip if points are too close to avoid numerical issues
        if np.allclose(p1, p2):
            continue
            
        # Calculate line parameters (ax + by + c = 0)
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        
        # Normalize parameters
        norm = np.sqrt(a*a + b*b)
        a, b, c = a/norm, b/norm, c/norm
        
        lines.append((a, b, c))
    
    return lines

def classify_point(point, line):
    """Classify a point based on which side of the line it falls."""
    a, b, c = line
    # ax + by + c gives signed distance to line
    return np.sign(a*point[0] + b*point[1] + c)

def get_predictions(points, line):
    """Get predictions for all points using a line classifier."""
    return np.array([classify_point(point, line) for point in points])

def adaboost_train(X, y, lines, T=8):
    """Train Adaboost for T iterations with progress tracking."""
    print(f"  Training Adaboost for {T} iterations...")
    n_samples = len(X)
    # Initialize weights uniformly
    weights = np.ones(n_samples) / n_samples
    
    alphas = []
    selected_classifiers = []
    
    for t in range(T):
        print(f"    Iteration {t + 1}/{T}: Finding best weak classifier...")
        min_error = float('inf')
        best_predictions = None
        best_line = None
        
        for line in lines:
            predictions = get_predictions(X, line)
            error = np.sum(weights * (predictions != y))
            
            if error < min_error:
                min_error = error
                best_predictions = predictions
                best_line = line
        
        # Calculate alpha (classifier weight)
        epsilon = min_error / np.sum(weights)
        if epsilon >= 0.5 or epsilon == 0:  # Skip if classifier is too weak or perfect
            continue
            
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        
        # Update weights
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights)  # Normalize
        
        alphas.append(alpha)
        selected_classifiers.append(best_line)
        print(f"      Selected classifier with error: {epsilon:.4f}, alpha: {alpha:.4f}")
        
        if len(selected_classifiers) == T:
            break
            
    return selected_classifiers, alphas

def adaboost_predict(X, classifiers, alphas, k):
    """Make predictions using the first k weak classifiers."""
    predictions = np.zeros(len(X))
    
    for alpha, classifier in zip(alphas[:k], classifiers[:k]):
        predictions += alpha * get_predictions(X, classifier)
        
    return np.sign(predictions)

def compute_error(y_true, y_pred):
    """Compute classification error."""
    return np.mean(y_true != y_pred)

def run_experiment():
    """Run the complete Adaboost experiment with progress tracking."""
    print("Starting Adaboost experiment with 100 runs...")
    X, y = load_versicolor_virginica()
    n_runs = 100
    k_values = range(1, 9)  # k from 1 to 8
    
    train_errors = np.zeros((n_runs, 8))
    test_errors = np.zeros((n_runs, 8))
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/100")
        # Randomly split data into train and test
        indices = np.random.permutation(len(X))
        split = len(X) // 2
        
        train_idx = indices[:split]
        test_idx = indices[split:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Generate lines from training points
        lines = create_line_classifiers(X_train, y_train)
        
        # Train Adaboost
        classifiers, alphas = adaboost_train(X_train, y_train, lines)
        
        # Compute errors for different k values
        for k in k_values:
            if k <= len(classifiers):
                y_train_pred = adaboost_predict(X_train, classifiers, alphas, k)
                y_test_pred = adaboost_predict(X_test, classifiers, alphas, k)
                
                train_errors[run, k-1] = compute_error(y_train, y_train_pred)
                test_errors[run, k-1] = compute_error(y_test, y_test_pred)
    
    # Compute average errors
    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)
    
    print("\nFinal Results after 100 runs:")
    print("\nAverage Training Errors:")
    for k, error in enumerate(avg_train_errors, 1):
        print(f"H{k}: {error:.4f}")
    
    print("\nAverage Test Errors:")
    for k, error in enumerate(avg_test_errors, 1):
        print(f"H{k}: {error:.4f}")

if __name__ == "__main__":
    run_experiment()