```markdown:README.md
# Machine Learning Assignment - Perceptron and AdaBoost

## Overview
This project implements and analyzes two fundamental machine learning algorithms:
1. Perceptron algorithm for binary classification
2. AdaBoost algorithm using line classifiers as weak learners

## Files Structure
- `problem_2.py`: Perceptron implementation and experiments with true margin calculation
- `perceptron.py`: Core perceptron algorithm and data preparation functions
- `true_margin.py`: True margin calculation utilities
- `problem_3.py`: AdaBoost implementation and experiments
- `iris.txt`: Iris dataset

## Problem 2: Perceptron
### Implementation
- Binary classification on Iris dataset pairs:
  - Setosa vs Versicolor
  - Setosa vs Virginica
- Features used: sepal width and petal length
- Calculates true margin between classes
- Runs until perfect separation is found (no iteration limit)

### Usage
```python
python problem_2.py
```

### Output
For each pair of classes:
- Final weight vector
- Number of mistakes
- True margin

## Problem 3: AdaBoost
### Implementation
- Uses line classifiers as weak learners
- Runs for T=8 iterations
- Performs 100 experimental runs
- 50-50 train-test split on Versicolor vs Virginica

### Usage
```python
python problem_3.py
```

### Output
For each classifier H₁ through H₈:
- Average True Error (test error)
- Average Empirical Error (training error)

## Dependencies
- Python 3.10.12
- NumPy
- scikit-learn (for loading Iris dataset)

## Technical Details
### Perceptron
- Continues until perfect separation is found
- No maximum iteration limit
- Updates weights when classification mistakes are made

### AdaBoost
- Creates weak classifiers using lines between point pairs
- Updates distribution weights according to classification errors
- Combines weak classifiers with learned weights (α values)

### True Margin
- Calculates maximum possible margin between linearly separable classes
- Uses geometric approach with point-to-line distances
- Considers all valid three-point combinations

## Running the Experiments
1. Ensure all dependencies are installed
2. Place iris.txt in the same directory
3. Run either problem_2.py or problem_3.py

## Notes
- Perceptron will run indefinitely for non-linearly-separable classes
- AdaBoost shows characteristic resistance to overfitting despite increasing model complexity
```
