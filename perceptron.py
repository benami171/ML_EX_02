"""
Using Python 3.11.9
"""
import numpy as np
from sklearn.datasets import load_iris

def perceptron(X, y):
    """
    Implements the Perceptron algorithm
    
    Parameters:
    X: array of shape (n_samples, n_features)
    y: array of labels (-1 or 1)
    
    Returns:
    w: final weight vector
    mistakes: number of mistakes made
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize w1 = 0 as seen in class.
    mistakes = 0
    
    while True:  # Run indefinitely until perfect separation is found
        made_mistake = False
        
        # Iterate over all points xi
        for i in range(n_samples):
            # Calculate dot product w_t · x_i
            prediction = np.dot(w, X[i])
            
            # If we made a mistake
            if (prediction <= 0 and y[i] == 1) or (prediction > 0 and y[i] == -1):
                # Update weights according to slides
                if y[i] == 1:
                    w = w + X[i]  # w_{t+1} ← w_t + x_i
                else:
                    w = w - X[i]  # w_{t+1} ← w_t - x_i
                    
                mistakes += 1
                made_mistake = True
        
        # If no mistakes in this round, we found a perfect separator
        if not made_mistake:
            break
            
    return w, mistakes


# Load and prepare the Iris data
def prepare_iris_data(iris_file, class1, class2):
    """
    Prepare Iris data for binary classification with Perceptron
    Only uses second and third features
    """
    # Read the iris data
    data = []
    with open(iris_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            # Take only second and third features
            features = [float(values[1]), float(values[2])]
            species = values[4]
            data.append((features, species))
    
    # Filter only the two classes we want
    filtered_data = [(features, 1 if species == class1 else -1) 
                    for features, species in data 
                    if species in [class1, class2]]
    
    # Convert to numpy arrays
    X = np.array([features for features, _ in filtered_data])
    y = np.array([label for _, label in filtered_data])
    
    return X, y