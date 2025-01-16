'''
Using Python 3.13.1
'''
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

def create_weak_classifiers(points, labels):
    """
    Create set H of T weak classifiers h_j: S → {-1,1}
    Each classifier is a line that divides the feature space.
    """
    H = []
    n_points = len(points)
    
    for i, j in combinations(range(n_points), 2):
        p1, p2 = points[i], points[j]
        
        if np.allclose(p1, p2):
            continue
            
        # Line equation: ax + by + c = 0
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        
        # Normalize parameters
        norm = np.sqrt(a*a + b*b)
        a, b, c = a/norm, b/norm, c/norm
        
        H.append((a, b, c))
    
    return H

def h_j(point, classifier):
    """
    Weak classifier h_j: S → {-1,1}
    Classifies a point based on its position relative to the line.
    """
    a, b, c = classifier
    return np.sign(a*point[0] + b*point[1] + c)

def evaluate_weak_classifier(points, classifier):
    """Evaluate h_j(x) for all points x in S."""
    return np.array([h_j(point, classifier) for point in points])

def adaboost_train(X, y, H, T=8):
    """
    AdaBoost training algorithm as presented in class.
    
    Input:
    - Set S of points x_i ∈ S with labels y_i
    - Number of iterations T
    - Set H of weak classifiers h_j: S → {-1,1}
    
    Output:
    - Weights α_j for each selected classifier h_j
    """
    n_samples = len(X)
    
    # Step 1: Initialize D₁(xᵢ) = 1/n
    D_t = np.ones(n_samples) / n_samples
    
    alphas = []          # Store α_t values
    selected_h_t = []    # Store selected classifiers
    
    for t in range(T):
        min_error = float('inf')
        best_h_t = None
        best_predictions = None
        
        # Step 3: Compute weighted error for each h ∈ H
        for h in H:
            predictions = evaluate_weak_classifier(X, h)
            # εₜ(h) = Σᵢ₌₁ⁿ Dₜ(xᵢ)[h(xᵢ) ≠ yᵢ]
            epsilon_t = np.sum(D_t * (predictions != y))
            
            # Step 4: Select classifier with minimum weighted error
            if epsilon_t < min_error:
                min_error = epsilon_t
                best_h_t = h
                best_predictions = predictions
        
        epsilon_t = min_error / np.sum(D_t)
        
        if epsilon_t >= 0.5 or epsilon_t == 0:
            continue
            
        # Step 5: Set classifier weight αₜ = ½ln((1-εₜ)/εₜ)
        alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        
        # Step 6: Update distribution
        # D_(t+1)(xᵢ) = (1/Zₜ) * Dₜ(xᵢ) * exp(-αₜyᵢhₜ(xᵢ))
        D_t_plus_1 = D_t * np.exp(-alpha_t * y * best_predictions)
        Z_t = np.sum(D_t_plus_1)  # Normalization constant
        D_t = D_t_plus_1 / Z_t
        
        alphas.append(alpha_t)
        selected_h_t.append(best_h_t)
        
        if len(selected_h_t) == T:
            break
            
    return selected_h_t, alphas

def F_x(X, classifiers, alphas, k):
    """
    Compute F(x) = Σᵏₜ₌₁ αₜhₜ(x)
    """
    result = np.zeros(len(X))
    for alpha_t, h_t in zip(alphas[:k], classifiers[:k]):
        result += alpha_t * evaluate_weak_classifier(X, h_t)
    return result

def H_x(X, classifiers, alphas, k):
    """
    Compute H(x) = sign[F(x)]
    """
    return np.sign(F_x(X, classifiers, alphas, k))

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
        
        H = create_weak_classifiers(X_train, y_train)
        classifiers, alphas = adaboost_train(X_train, y_train, H)
        
        for k in k_values:
            if k <= len(classifiers):
                y_train_pred = H_x(X_train, classifiers, alphas, k)
                y_test_pred = H_x(X_test, classifiers, alphas, k)
                
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